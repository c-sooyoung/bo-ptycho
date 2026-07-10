import os
import sys
import yaml

def ptycho_run(config):
    ptycho_engine = config['ptycho']['engine']

    if ptycho_engine == 'fold_slice':
        from ptycho.fold_slice import run
        run(config)

    # Add more ptycho engines here as needed
    # write their respective run functions and import them above


def ptycho_error(config):
    ptycho_engine = config['ptycho']['engine']

    if ptycho_engine == 'fold_slice':
        from ptycho.fold_slice import error
        return error(config)






def bo_initialize(config):
    bo_algorithm = config['bo']['algorithm']

    if bo_algorithm == 'ucb':
        from bo.ucb import initialize
        bo_state = initialize(config)
    
    else:
        from bo.random import initialize
        bo_state = initialize(config)
    
    return bo_state


def bo_ask(config, bo_state):
    bo_engine = config['bo']['algorithm']

    if bo_engine == 'ucb':
        from bo.ucb import ask
        next_config = ask(config, bo_state)
    
    else:
        from bo.random import ask
        next_config = ask(config, bo_state)

    # Add more BO engines here as needed
    # write their respective ask functions and import them above

    return next_config


def bo_tell(config, job_config, bo_state, y_value):
    bo_engine = config['bo']['algorithm']

    if bo_engine == 'ucb':
        from bo.ucb import tell
        bo_config = tell(job_config, bo_state, y_value)
    
    else:
        from bo.random import tell
        bo_config = tell(job_config, bo_state, y_value)
    
    return bo_config






if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bo-ptycho.py <config_yaml>")
        sys.exit(1)

    config_yaml = sys.argv[1]
    with open(config_yaml, 'r') as f:
        config = yaml.safe_load(f)

    # ptycho_run(config_yaml)
    # results = ptycho_results(config_yaml)
    # new_config = bo_loop(config)
    # print(new_config) # type: ignore
    
