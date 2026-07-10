import os
import sys
import shutil
import subprocess
import numpy as np
from scipy.io import loadmat

def fold_slice_translator(config):
    fold_slice_result_dir = os.path.join(config['io']['result_dir'], 'fold_slice')

    fold_slice_dict = {}
    fold_slice_dict['raw_data'] = config['io']['input_data_path']
    fold_slice_dict['result_dir'] = os.path.join(fold_slice_result_dir, '')
    fold_slice_dict.update(config['ptycho']['params'])

    if os.path.exists(os.path.join(fold_slice_result_dir)):
        shutil.rmtree(os.path.join(fold_slice_result_dir))
    os.makedirs(os.path.join(fold_slice_result_dir))

    setup_txt = os.path.join(fold_slice_result_dir, 'setup.txt')
    with open(setup_txt, 'w') as f:
        f.write('\n\n')
        for key, value in fold_slice_dict.items():
            f.write(f"{key} {value}\n")
    
    return setup_txt


def run(config):
    setup_txt = fold_slice_translator(config)
    fold_slice_path = config['ptycho']['path']
    verbosity = config['io'].get('verbosity', 0)

    matlab_commands = [
        f"cd('{fold_slice_path}');",
        "cd('ptycho');",
        f"prepare_data('{setup_txt}');"
        f"run_multislice_new('{setup_txt}');"
    ]
    
    p = subprocess.Popen(
        ['matlab', '-batch', ' '.join(matlab_commands)],
        stdout=subprocess.PIPE if verbosity > 0 else subprocess.DEVNULL,
        stderr=subprocess.STDOUT if verbosity > 0 else subprocess.DEVNULL,
        text=True
    )

    if not verbosity == 0:
        header = '[fold slice]'
        for line in p.stdout: # type: ignore
            sys.stdout.write(f'{header} {line}')
        p.stdout.close() # type: ignore
    else:
        print(f"fold_slice running. Set verbosity>0 for full fold_slice output.")

    p.wait()


def error(config):
    fold_slice_result_dir = os.path.join(config['io']['result_dir'], 'fold_slice')
    roi_dir = os.path.join(
        fold_slice_result_dir,
        f"{config['ptycho']['params']['scan_number']}",
        f"roi{config['ptycho']['params']['roi_label']}"
    )
    output_dir = os.path.join(roi_dir, next(os.walk(roi_dir))[1][0])
    # image_path = os.path.join(output_dir, 'obj_phase_roi_sum', next(os.walk(os.path.join(output_dir, 'obj_phase_roi_sum')))[2][0])
    result_mat = os.path.join(output_dir, f"Niter{config['ptycho']['params']['Niter']}.mat")
    if not os.path.exists(result_mat):
        raise FileNotFoundError(f"Result directory {result_mat} does not exist. Please check the fold_slice output.")
    
    # return loadmat(result_mat)
    return 1