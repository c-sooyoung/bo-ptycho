from abc import ABC, abstractmethod
from typing import Any


class PtychoEngine(ABC):
    name = None

    def __init__(self, config):
        self.config = config
        self._output = None
        self._recon_object = None
        self._recon_probe = None
        self._metric = None


    @abstractmethod
    def run(self) -> None:
        pass

    @abstractmethod
    def output(self) -> Any:
        pass

    @abstractmethod
    def recon_object(self) -> Any:
        pass

    @abstractmethod
    def recon_probe(self) -> Any:
        pass

    @abstractmethod
    def metric(self) -> float:
        pass
