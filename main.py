import os
import sys
import yaml
import shutil

import pipelines

def main(config_yaml):
    with open(config_yaml, 'r') as f:
        config = yaml.safe_load(f)

    result_dir = config["io"]["result_dir"]
    if os.path.exists(result_dir):
        if sys.stdin.isatty():
            answer = input(f"Will delete {result_dir}: [y/N]\n> ").strip().lower() == 'y'
            if not answer:
                print("Aborted.")
                return         
        shutil.rmtree(result_dir)
    
    os.makedirs(result_dir, exist_ok=True)
    shutil.copy(config_yaml, os.path.join(result_dir, os.path.basename(config_yaml)))
    pipelines.job_types[config['job']['type']](config)
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bo-ptycho.py <config_yaml>")
        sys.exit(1)
    config_yaml = sys.argv[1]
    main(config_yaml)
