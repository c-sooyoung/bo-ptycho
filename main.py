import os
import sys
import yaml
import shutil
import numpy as np

import bo
import ptycho

def main(config_yaml):

    with open(config_yaml, 'r') as f:
        config = yaml.safe_load(f)

    result_dir = config['io']['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    shutil.copy(config_yaml, os.path.join(result_dir, os.path.basename(config_yaml)))

    randombo = bo.RandomBOEngine(config)

    for j in range(10):
        job_config = randombo.ask()
        ptycho_engine = ptycho.FoldSlicePtychoEngine(job_config)
        ptycho_engine.run()
        y_value = -np.log(ptycho_engine.metric())
        randombo.tell(job_config, y_value)
        print('[random] TRAIN_X\n', randombo.state['train_x'])
        print('[random] TRAIN_Y\n', randombo.state['train_y'])

    sobo = bo.SingleObjectiveBOEngine(config)
    sobo.train_x = randombo.state['train_x']
    sobo.train_y = randombo.state['train_y']

    for j in range(config['bo']['max_iterations']):
        job_config = sobo.ask()
        ptycho_engine = ptycho.FoldSlicePtychoEngine(job_config)
        ptycho_engine.run(header=f"[BO {j:03d}] ")
        y_value = -np.log(ptycho_engine.metric())
        sobo.tell(job_config, y_value)
        print('[sobo] TRAIN_X\n', sobo.train_x)
        print('[sobo] TRAIN_Y\n', sobo.train_y)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bo-ptycho.py <config_yaml>")
        sys.exit(1)
    config_yaml = sys.argv[1]
    main(config_yaml)
