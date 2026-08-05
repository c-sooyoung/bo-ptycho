from abc import ABC, abstractmethod
from typing import Any


class PtychoEngine(ABC):
    name = None

    def __init__(self, config):
        self.config = config
        self._verbosity = self.config['io'].get('verbosity', 0)
        self._output = None

    @abstractmethod
    def run(self, run_id="") -> None:
        pass

    @abstractmethod
    def metric(self, names: str | list[str]) -> float | list[float]:
        pass
