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

    # RANDOM BO SAMPLING PREPARATIONS; 20 SAMPLES
    randombo = bo.RandomBOEngine(config)
    for j in range(20):
        print(f"RANDOM sampling iteration {j+1}")
        job_config = randombo.ask()
        ptycho_engine = ptycho.FoldSlicePtychoEngine(job_config)
        ptycho_engine.run()
        y_value = -np.log(ptycho_engine.metric())
        randombo.tell(job_config, y_value)

    # MAIN SINGLE OBJECTIVE BAYESIAN OPTIMIZATION
    sobo = bo.SingleObjectiveBOEngine(config)
    sobo.train_x = randombo.train_x
    sobo.train_y = randombo.train_y
    for j in range(config['bo']['max_iterations']):
        print(f"SOBO sampling iteration {j+1}")
        job_config = sobo.ask()
        ptycho_engine = ptycho.FoldSlicePtychoEngine(job_config)
        ptycho_engine.run(header=f"[BO {j:03d}] ")
        y_value = -np.log(ptycho_engine.metric())
        sobo.tell(job_config, y_value)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bo-ptycho.py <config_yaml>")
        sys.exit(1)
    config_yaml = sys.argv[1]
    main(config_yaml)
