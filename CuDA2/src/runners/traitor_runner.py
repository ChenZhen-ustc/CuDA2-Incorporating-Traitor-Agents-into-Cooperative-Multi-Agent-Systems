from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import random
import torch

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, scheme_base, groups, groups_base, preprocess, mac, mac_base, rnd):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.new_batch_base = partial(EpisodeBatch, scheme_base, groups_base, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)                         
        self.mac = mac
        self.mac_base = mac_base
        self.rnd = rnd

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.batch_base = self.new_batch_base()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()
        t = 0
        victim_agent_num = self.get_env_info()["n_agents"] - self.args.traitor_num
        traitor_agent_num = self.args.traitor_num
        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.mac_base.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()[victim_agent_num:victim_agent_num+traitor_agent_num]],
                "obs": [self.env.get_obs()[victim_agent_num:victim_agent_num+traitor_agent_num]]
            }
            obs_base = []
            for i in range(len(self.env.get_obs())-traitor_agent_num):
                obs_base += [np.delete(self.env.get_obs()[i],slice(-2-5*traitor_agent_num,-2))]
            pre_transition_data_base = {
                "state": [np.delete(self.env.get_state(),slice(victim_agent_num*4,(victim_agent_num+traitor_agent_num)*4))],
                "avail_actions": [self.env.get_avail_actions()[:victim_agent_num]],
                "obs": [obs_base]
            }

            

            self.batch.update(pre_transition_data, ts=self.t)
            self.batch_base.update(pre_transition_data_base, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            if self.args.action_type == "stop":
                for al_id, al_unit in self.env.agents.items():
                    for k in range(self.args.traitor_num):
                        if al_id == victim_agent_num+k:
                            if al_unit.health > 0:
                                actions[0][k] = 1
                            else:
                                actions[0][k] = 0
            elif self.args.action_type == "random":
                for k in range(self.args.traitor_num):
                    actions[0][k] = random.choice(np.nonzero(self.env.get_avail_actions()[victim_agent_num:victim_agent_num+traitor_agent_num][k])[0])
            actions_base = self.mac_base.select_actions(self.batch_base, t_ep=self.t, t_env=self.t_env, test_mode=True)
            actions_all = torch.cat([actions_base, actions], 1)
            state_rnd = np.zeros(2*victim_agent_num)
            for i in range(victim_agent_num):
                state_rnd[i*2:i*2+2] = self.env.get_state()[i*4+2:i*4+4]
            state_rnd[::2]=state_rnd[::2]*self.env.max_distance_x+self.env.map_x/2
            state_rnd[1::2]=state_rnd[1::2]*self.env.max_distance_y+self.env.map_y/2
            state_rnd = np.around(state_rnd,2)
            Ri = self.rnd.get_reward(torch.tensor(state_rnd, dtype=torch.float))
            reward, terminated, env_info = self.env.step(actions_all[0])
            
            if t == 0 and self.args.action_type == "PBRS_RND":
                self.rnd.update(Ri)
                t += 1
            state_next_rnd = np.zeros(2*victim_agent_num)
            for i in range(victim_agent_num):
                state_next_rnd[i*2:i*2+2] = self.env.get_state()[i*4+2:i*4+4]
            state_next_rnd[::2]=state_next_rnd[::2]*self.env.max_distance_x+self.env.map_x/2
            state_next_rnd[1::2]=state_next_rnd[1::2]*self.env.max_distance_y+self.env.map_y/2
            state_next_rnd = np.around(state_next_rnd,2)
            Ri_next = self.rnd.get_reward(torch.tensor(state_next_rnd, dtype=torch.float))
            if self.args.action_type == "random":
                self.rnd.update(Ri)
            if self.args.action_type == "PBRS_RND":
                self.rnd.update(Ri_next)
                

            Ri = Ri.item()
            Ri_next = Ri_next.item()*0.99
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward
            traitor_reward = 0
            if self.args.action_type == "minus_r":
                traitor_reward = -reward
            elif self.args.action_type == "PBRS_RND":
                traitor_reward = -reward+(Ri_next-Ri)*10
            post_transition_data = {
                "actions": actions,
                "reward": [(traitor_reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            post_transition_data_base = {
                "actions": actions_base,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)
            self.batch_base.update(post_transition_data_base, ts=self.t)

            self.t += 1

        # last_data = {
        #     "state": [self.env.get_state()],
        #     "avail_actions": [self.env.get_avail_actions()],
        #     "obs": [self.env.get_obs()]
        # }
        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()[victim_agent_num:victim_agent_num+traitor_agent_num]],
            "obs": [self.env.get_obs()[victim_agent_num:victim_agent_num+traitor_agent_num]]
        }
        obs_base = []
        for i in range(len(self.env.get_obs())-traitor_agent_num):
            obs_base += [np.delete(self.env.get_obs()[i],slice(-2-5*traitor_agent_num,-2))]
        last_data_base = {
            "state": [np.delete(self.env.get_state(),slice(victim_agent_num*4,(victim_agent_num+traitor_agent_num)*4))],
            "avail_actions": [self.env.get_avail_actions()[:victim_agent_num]],
            "obs": [obs_base]
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)
        self.batch_base.update(last_data_base, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        actions_base = self.mac_base.select_actions(self.batch_base, t_ep=self.t, t_env=self.t_env, test_mode=True)
        self.batch.update({"actions": actions}, ts=self.t)
        self.batch_base.update({"actions": actions_base}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
