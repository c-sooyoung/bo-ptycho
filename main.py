import os
import sys
import yaml

import pipelines

def main(config_yaml):

    with open(config_yaml, 'r') as f:
        config = yaml.safe_load(f)


    job_types = {
        'random+sobo': pipelines.sobo_pipeline
    }

    job_types[config['job']['type']](config)
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python bo-ptycho.py <config_yaml>")
        sys.exit(1)
    config_yaml = sys.argv[1]
    main(config_yaml)
