from dataclasses import dataclass

from config import AvailableBackends


@dataclass
class GempyEngineConfig:
    backend: AvailableBackends = AvailableBackends.numpy # ? This can be grabbed from gempy.config file?
    use_gpu: bool = False
    pykeops_enabled: bool = False