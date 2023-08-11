from dataclasses import dataclass

from gempy.config import AvailableBackends


@dataclass
class GemPyEngineConfig:
    backend: AvailableBackends = AvailableBackends.numpy # ? This can be grabbed from gempy.config file?
    use_gpu: bool = False