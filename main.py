import os
import sys
import yaml

import bo
import ptycho

def main(config):

    bo_engine = bo.RandomBOEngine(config)

    max_iterations = config['bo']['max_iterations']

    for _ in range(max_iterations):
        job_config = bo_engine.ask()

        ptycho_engine = ptycho.FoldSlicePtychoEngine(job_config)
        ptycho_engine.run()
        y_value = ptycho_engine.metric()

        bo_engine.tell(job_config, y_value)

        print(bo_engine.state['train_x'])
        print(bo_engine.state['train_y'])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bo-ptycho.py <config_yaml>")
        sys.exit(1)

    config_yaml = sys.argv[1]
    with open(config_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)
