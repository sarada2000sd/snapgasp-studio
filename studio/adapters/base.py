from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

class ModelAdapter(ABC):
    def __init__(self, spec: dict[str, Any], local_path: Path):
        self.spec, self.local_path, self.pipeline = spec, Path(local_path), None
    @abstractmethod
    def load(self, runtime: dict[str, Any]) -> None: ...
    @abstractmethod
    def capabilities(self) -> dict[str, bool]: ...
    @abstractmethod
    def generate(self, **kwargs: Any) -> Any: ...
    def unload(self):
        self.pipeline = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception: pass
