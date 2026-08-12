from abc import ABC, abstractmethod


class BOEngine(ABC):
    def __init__(self, config):
        self.config = config
        self.state = None

    @abstractmethod
    def ask(self, n: int = 1) -> list[dict]:
        pass

    @abstractmethod
    def tell(self, job_config, y_value):
        pass
