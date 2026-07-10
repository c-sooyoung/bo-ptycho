import numpy as np
import copy


def initialize(config):
    bo_params = []

    for key, value in config['bo']['params'].items():
        if value is not None:
            bo_params.append(key)

    train_x = np.empty((0, len(bo_params)))
    train_y = np.empty((0,))

    bo_state = {
        'params': bo_params,
        'train_x': train_x,
        'train_y': train_y,
    }

    return bo_state


def ask(config, bo_state):
    next_config = copy.deepcopy(config)

    for param in bo_state['params']:
        max_modulation = config['bo']['params'][param]
        center_value = config['ptycho']['params'][param]

        modulation = max_modulation * (np.random.rand() - 0.5) * 2
        next_config['ptycho']['params'][param] = center_value + modulation

    return next_config


def tell(job_config, bo_state, y_value):
    x_value = []

    for param in bo_state['params']:
        x_value.append(job_config['ptycho']['params'][param])

    x_value = np.array(x_value).reshape(1, -1)
    y_value = np.array([y_value])

    bo_state['train_x'] = np.vstack([
        bo_state['train_x'],
        x_value,
    ])

    bo_state['train_y'] = np.concatenate([
        bo_state['train_y'],
        y_value,
    ])

    return bo_state