import os
import sys
import yaml

import bo
import ptycho

def main(config):

    randombo = bo.RandomBOEngine(config)

    max_iterations = config['bo']['max_iterations']

    for j in range(10):
        job_config = randombo.ask()

        ptycho_engine = ptycho.FoldSlicePtychoEngine(job_config)
        ptycho_engine.run()
        y_value = ptycho_engine.metric()

        randombo.tell(job_config, y_value)

        print('[random] TRAIN_X\n', randombo.state['train_x'])
        print('[random] TRAIN_Y\n', randombo.state['train_y'])


    sobo = bo.SingleObjectiveBOEngine(config)

    for j in range(max_iterations):
        job_config = sobo.ask()

        ptycho_engine = ptycho.FoldSlicePtychoEngine(job_config)
        ptycho_engine.run()
        y_value = ptycho_engine.metric()

        sobo.tell(job_config, y_value)

        print('[sobo] TRAIN_X\n', sobo.train_x)
        print('[sobo] TRAIN_Y\n', sobo.train_y)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bo-ptycho.py <config_yaml>")
        sys.exit(1)

    config_yaml = sys.argv[1]
    with open(config_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    main(config)
