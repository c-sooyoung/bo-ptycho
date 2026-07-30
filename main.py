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
    J = 20
    randombo = bo.RandomBOEngine(config)

    bo_txt = os.path.join(result_dir, "bo.txt")
    with open(bo_txt, "w") as f:
        f.write(f"    iter\tmetric\t{"\t".join([p[:7] for p in randombo.params])}\n")

    for j in range(J):
        print(f"RANDOM sampling iteration {j}")
        job_config = randombo.ask()
        ptycho_engine = ptycho.FoldSlicePtychoEngine(job_config)
        ptycho_engine.run(run_id=f"bo-{j:03d}")
        y_value = ptycho_engine._metric
        randombo.tell(job_config, y_value)
        with open(bo_txt, "a") as f:
            p = [f'{job_config['ptycho']['params'][key]:.2f}' for key in randombo.params]
            f.write(f"{j: 8d}\t{y_value:.4f}\t{"\t".join(p)}\n")

    # MAIN SINGLE OBJECTIVE BAYESIAN OPTIMIZATION
    sobo = bo.SingleObjectiveBOEngine(config)
    sobo.train_x = randombo.train_x
    sobo.train_y = randombo.train_y
    for j in range(J, J + config['bo']['max_iterations'] + 1):
        print(f"SOBO sampling iteration {j}")
        job_config = sobo.ask()
        ptycho_engine = ptycho.FoldSlicePtychoEngine(job_config)
        ptycho_engine.run(run_id=f"bo-{j:03d}")
        y_value = ptycho_engine._metric
        sobo.tell(job_config, y_value)
        with open(bo_txt, "a") as f:
            p = [f'{job_config['ptycho']['params'][key]:.2f}' for key in sobo.params]
            f.write(f"{j: 8d}\t{y_value:.4f}\t{"\t".join(p)}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bo-ptycho.py <config_yaml>")
        sys.exit(1)
    config_yaml = sys.argv[1]
    main(config_yaml)
