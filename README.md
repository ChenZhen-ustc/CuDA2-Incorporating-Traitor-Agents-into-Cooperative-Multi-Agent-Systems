# CuDA2-Incorporating-Traitor-Agents-into-Cooperative-Multi-Agent-Systems
CuDA2 is  an extension of [EPyMARL](https://github.com/uoe-agents/epymarl), and includes
- Add traitors into CMARL
- Additional maps for test (you need to put the maps into the smac environment you installed and you also need to change the files "smac_maps.py" e.g. xxxxx/smac/env/starcraft2/maps/smac_maps.py)
- Two traitor runners for episode runner and parallel runner

# Installation & Run instructions

For information on installing and using this codebase with SMAC, we suggest visiting and reading the original [EPyMARL](https://github.com/uoe-agents/epymarl) README. Here, we maintain information on using the extra features CuDA2 offers.
To install the codebase, clone this repo and install the `requirements.txt`.  

Example of running origin algorithm without traitors:
```sh
python3 src/main.py --config=qmix action_type=origin traitor_num=0 --env-config=sc2 with env_args.map_name=6m_vs_6m
```
Example of running traitors that choice stop/random/minus_r/PBRS_RND actions:
```sh
python3 src/main.py --config=qmix action_type=stop traitor_num=1 --env-config=sc2 with env_args.map_name=7m_vs_6m
python3 src/main.py --config=qmix action_type=random traitor_num=1 --env-config=sc2 with env_args.map_name=7m_vs_6m
python3 src/main.py --config=qmix action_type=minus_r traitor_num=1 --env-config=sc2 with env_args.map_name=7m_vs_6m
python3 src/main.py --config=qmix action_type=PBRS_RND traitor_num=1 --env-config=sc2 with env_args.map_name=7m_vs_6m
```
Note that you have to run "origin" before you can run one with traitors. And you also have to run "random" before running "PBRS_RND"
