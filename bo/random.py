import os
import copy
import numpy as np
from bo.bo_base import BOEngine


class RandomBOEngine(BOEngine):

    def __init__(self, config):
        super().__init__(config)

        bo_params = [
            key for key, spec in config["bo"]["params"].items() if spec is not None
        ]

        bo_param_types = {
            key: config["bo"]["params"][key].get("type", "float") for key in bo_params
        }

        integer_params = [
            key for key in bo_params if bo_param_types[key] == "int"
        ]

        integer_indices = [
            bo_params.index(key) for key in integer_params
        ]

        bounds = np.empty((2, len(bo_params)))

        for i, param in enumerate(bo_params):
            center = config["ptycho"]["params"][param]
            radius = config["bo"]["params"][param]["radius"]
            bounds[0, i] = center - radius
            bounds[1, i] = center + radius

        state = {
            "method": "random",
            "acquisition": "",
            "params": bo_params,
            "param_types": bo_param_types,
            "integer_params": integer_params,
            "integer_indices": integer_indices,
            "bounds": bounds,                          # shape: (2, BOparam)
            "train_x": np.empty((0, len(bo_params))),  # shape: (BOiter, BOparam)
            "train_y": np.empty((0,)),                 # shape: (BOiter,)
            "train_info": []                           # shape: (BOiter,)
        }

        train_x_path = config["bo"].get("train_x")
        train_y_path = config["bo"].get("train_y")

        if train_x_path is not None and train_y_path is not None:
            if os.path.exists(train_x_path) and os.path.exists(train_y_path):
                train_x = np.load(train_x_path)
                train_y = np.load(train_y_path)
                assert train_x.ndim == 2,                    "loaded train_x must be 2D"
                assert train_x.shape[1] == len(bo_params),   "loaded train_x shape(1) does not match number of variable parameters"
                assert train_y.ndim == 1,                    "loaded train_y must be 1D"
                assert train_y.shape[0] == train_x.shape[0], "loaded train_x and train_y shape(0) have unequal iterations"
                state["train_x"] = train_x
                state["train_y"] = train_y

        self.state = state


    def ask(self):
        config = self.config
        state = self.state
        next_config = copy.deepcopy(config)

        for param in state['params']:
            radius = config['bo']['params'][param]['radius']
            center = config['ptycho']['params'][param]
            modulation = radius * (np.random.rand() - 0.5) * 2
            next_value = center + modulation
            if state['param_types'][param] == 'int':
                next_value = round(next_value)
            next_config['ptycho']['params'][param] = next_value

        return next_config


    def tell(self, job_config, y_value):
        config = self.config
        state = self.state

        x_value = []
        for param in state['params']:
            x_value.append(job_config['ptycho']['params'][param])

        state['train_x'] = np.vstack([
            state['train_x'],
            np.array(x_value).reshape(1, -1)
        ])

        state['train_y'] = np.concatenate([
            state['train_y'],
            np.array([y_value])
        ])

        state['train_info'].append(state['method'])


        train_x_path = config['bo'].get('train_x')
        train_y_path = config['bo'].get('train_y')
        if train_x_path is not None and train_y_path is not None:
            np.save(train_x_path, state['train_x'])
            np.save(train_y_path, state['train_y'])
        else:
            result_dir = config['io']['result_dir']
            np.save(os.path.join(result_dir, 'train_x.npy'), state['train_x'])
            np.save(os.path.join(result_dir, 'train_y.npy'), state['train_y'])
