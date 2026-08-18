import copy
import numpy as np
from samplers.base import Sampler


class RandomSampler(Sampler):

    def __init__(self, config):
        super().__init__(config)


    def ask(self, n = 1):
        next_configs = []

        for _ in range(n):
            next_config = copy.deepcopy(self.config)
            for param in self.params:
                radius = self.config['bo']['params'][param]['radius']
                center = self.config['ptycho']['params'][param]
                modulation = radius * (np.random.rand() - 0.5) * 2
                next_value = center + modulation
                if self.param_types[param] == 'int':
                    next_value = round(next_value)
                next_config['ptycho']['params'][param] = next_value
            next_configs.append(next_config)

        return next_configs
