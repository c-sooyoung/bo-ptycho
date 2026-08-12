import os
import copy
import numpy as np
from bo.base import BOEngine


class RandomBOEngine(BOEngine):

    def __init__(self, config):
        super().__init__(config)

        self.params = [key for key, spec in config["bo"]["params"].items() if spec is not None]
        self.param_types = {key: config["bo"]["params"][key].get("type", "float") for key in self.params}
        self.integer_indices = [i for i, param in enumerate(self.params) if self.param_types[param] == 'int']
        self.bounds = np.empty((2, len(self.params)))

        for i, param in enumerate(self.params):
            center = config["ptycho"]["params"][param]
            radius = config["bo"]["params"][param]["radius"]
            self.bounds[0, i] = center - radius
            self.bounds[1, i] = center + radius

        self.train_x = np.empty((0, len(self.params)))  # shape: (BOiter, BOparam)
        self.train_y = np.empty((0,))                   # shape: (BOiter,)

        train_x_path = config["bo"].get("train_x")
        train_y_path = config["bo"].get("train_y")
        if train_x_path is not None and train_y_path is not None:
            if os.path.exists(train_x_path) and os.path.exists(train_y_path):
                train_x = np.load(train_x_path)
                train_y = np.load(train_y_path)
                assert train_x.ndim == 2,                    "loaded train_x must be 2D"
                assert train_x.shape[1] == len(self.params), "loaded train_x shape(1) does not match number of variable parameters"
                assert train_y.ndim == 1,                    "loaded train_y must be 1D"
                assert train_y.shape[0] == train_x.shape[0], "loaded train_x and train_y shape(0) have unequal iterations"
                self.train_x = train_x
                self.train_y = train_y
        

    def ask(self, n = 1):
        config = self.config
        next_configs = []

        for _ in range(n):
            next_config = copy.deepcopy(config)
            for param in self.params:
                radius = config['bo']['params'][param]['radius']
                center = config['ptycho']['params'][param]
                modulation = radius * (np.random.rand() - 0.5) * 2
                next_value = center + modulation
                if self.param_types[param] == 'int':
                    next_value = round(next_value)
                next_config['ptycho']['params'][param] = next_value
            next_configs.append(next_config)

        return next_configs


    def tell(self, job_config, y_value):
        config = self.config

        x_value = []
        for param in self.params:
            x_value.append(job_config['ptycho']['params'][param])

        self.train_x = np.vstack([
            self.train_x,
            np.array(x_value).reshape(1, -1)
        ])

        self.train_y = np.concatenate([
            self.train_y,
            np.array([y_value])
        ])


        train_x_path = config['bo'].get('train_x')
        train_y_path = config['bo'].get('train_y')
        if train_x_path is not None and train_y_path is not None:
            np.save(train_x_path, self.train_x)
            np.save(train_y_path, self.train_y)
        else:
            result_dir = config['io']['result_dir']
            np.save(os.path.join(result_dir, 'train_x.npy'), self.train_x)
            np.save(os.path.join(result_dir, 'train_y.npy'), self.train_y)
