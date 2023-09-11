from dataclasses import dataclass
from typing import Optional

from gempy_engine.config import AvailableBackends


@dataclass
class GemPyEngineConfig:
    backend: AvailableBackends = AvailableBackends.numpy # ? This can be grabbed from gempy.config file?
    use_gpu: bool = False
    dtype: Optional[str] = None  #: The data type used in the engine. If None, the default data type of the backend is used.
    