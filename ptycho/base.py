from abc import ABC, abstractmethod
from typing import Any


class PtychoEngine(ABC):
    name = None

    def __init__(self, config):
        self.config = config


    @abstractmethod
    def run(self, run_id="") -> None:
        pass
