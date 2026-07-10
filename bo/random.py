import os
import copy
import numpy as np


def initialize(config):

    bo_params = [
        key
        for key, value in config['bo']['params'].items()
        if value is not None
    ]

    bo_state = {
        'algorithm': 'random',
        'params': bo_params,
        'train_x': np.empty((0, len(bo_params))),
        'train_y': np.empty((0,)),
    }

    train_x_path = config['bo'].get('train_x')
    train_y_path = config['bo'].get('train_y')

    if train_x_path is not None and train_y_path is not None:
        if os.path.exists(train_x_path) and os.path.exists(train_y_path):
            train_x = np.load(train_x_path)
            train_y = np.load(train_y_path)

            if (
                train_x.ndim == 2
                and train_x.shape[1] == len(bo_params)
                and train_y.ndim == 1
                and train_y.shape[0] == train_x.shape[0]
            ):
                bo_state['train_x'] = train_x
                bo_state['train_y'] = train_y

    return bo_state


def ask(config, bo_state):
    next_config = copy.deepcopy(config)

    for param in bo_state['params']:
        max_modulation = config['bo']['params'][param]
        center_value = config['ptycho']['params'][param]

        modulation = max_modulation * (np.random.rand() - 0.5) * 2
        next_config['ptycho']['params'][param] = center_value + modulation

    return next_config


def tell(config, job_config, bo_state, y_value):
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

    result_dir = config['io']['result_dir']
    np.save(os.path.join(result_dir, 'train_x.npy'), bo_state['train_x'])
    np.save(os.path.join(result_dir, 'train_y.npy'), bo_state['train_y'])

    
    return bo_state