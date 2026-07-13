from abc import ABC, abstractmethod


class BOEngine(ABC):
    name = None

    def __init__(self, config):
        self.config = config
        self.state = None

    @abstractmethod
    def initialize(self):
        pass

    @abstractmethod
    def ask(self):
        pass

    @abstractmethod
    def tell(self, job_config, y_value):
        pass
