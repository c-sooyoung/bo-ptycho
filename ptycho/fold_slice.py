import os
import sys
import shutil
import subprocess
import re
from typing import Any
from scipy.io import loadmat
import numpy as np
from ptycho.ptycho_base import PtychoEngine


class FoldSlicePtychoEngine(PtychoEngine):

    def __init__(self, config):
        super().__init__(config)
        self._output_dir = os.path.join(config['io']['result_dir'], 'fold_slice')
        self._fold_slice_path = self.config['ptycho']['path']
        self._setup_txt_path = os.path.join(self._output_dir, 'setup.txt')
        self._verbosity = self.config['io'].get('verbosity', 0)


    def run(self, run_id="") -> None:
        fold_slice_dict = {}
        fold_slice_dict['raw_data'] = self.config['io']['input_data_path']
        fold_slice_dict['result_dir'] = os.path.join(self._output_dir, '')
        fold_slice_dict.update(self.config['ptycho']['params'])
        for key, value in fold_slice_dict.items():
            if type(value) == bool:
                fold_slice_dict[key] = str(value).lower()

        if os.path.exists(os.path.join(self._output_dir)):
            shutil.rmtree(os.path.join(self._output_dir))
        os.makedirs(os.path.join(self._output_dir))

        with open(os.path.join(self._output_dir, 'setup.txt'), 'w') as f:
            f.write('\n\n')
            for key, value in fold_slice_dict.items():
                f.write(f"{key} {value}\n")

        matlab_commands = [
            f"cd('{self._fold_slice_path}');",
            "cd('ptycho');",
            f"prepare_data('{self._setup_txt_path}');",
            f"run_multislice_new('{self._setup_txt_path}');"
        ]
        
        p = subprocess.Popen(
            ['matlab', '-batch', ' '.join(matlab_commands)],
            stdout=subprocess.PIPE   if self._verbosity > 0 else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if self._verbosity > 0 else subprocess.DEVNULL,
            text=True
        )

        if self._verbosity > 0:
            for line in p.stdout: # type: ignore
                sys.stdout.write(f'[{run_id}] [fold_slice] {line}')
            p.stdout.close() # type: ignore
        else:
            print(f"fold_slice running. Set verbosity > 0 for full fold_slice output.")

        p.wait()

        roi_dir = os.path.join(
            self._output_dir,
            f"{self.config['ptycho']['params']['scan_number']}",
            f"roi{self.config['ptycho']['params']['roi_label']}"
        )
        output_dir = os.path.join(roi_dir, next(os.walk(roi_dir))[1][0])
        mat_path = os.path.join(output_dir, f"Niter{self.config['ptycho']['params']['Niter']}.mat")
        image_path = os.path.join(output_dir, 'obj_phase_roi_sum', f"obj_phase_roi_sum_Niter{self.config['ptycho']['params']['Niter']}.tiff")

        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"{mat_path} does not exist. Please check the fold_slice output.")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"{image_path} does not exist. Please check the fold_slice output.")

        self._output = loadmat(mat_path)
        self._metric = -np.log(float(self._output['outputs']['fourier_error_out'][0][0].squeeze()[-1]))

        os.makedirs(os.path.join(self.config['io']['result_dir'], "mat"), exist_ok=True)
        os.makedirs(os.path.join(self.config['io']['result_dir'], "tiff"), exist_ok=True)
        shutil.copy(mat_path, os.path.join(self.config['io']['result_dir'], "mat", f"{self._metric:.4f}_{run_id}.mat"))
        shutil.copy(image_path, os.path.join(self.config['io']['result_dir'], "tiff", f"{self._metric:.4f}_{run_id}.tiff"))
