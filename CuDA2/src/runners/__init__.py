REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .traitor_runner import EpisodeRunner
REGISTRY["traitor"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner
