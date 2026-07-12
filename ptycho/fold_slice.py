import os
import sys
import shutil
import subprocess
from typing import Any
from scipy.io import loadmat
import numpy as np
from ptycho.ptycho_base import PtychoEngine


class FoldSlicePtychoEngine(PtychoEngine):

    def __init__(self, config):
        super().__init__(config)
        self.fold_slice_output_dir = os.path.join(config['io']['result_dir'], 'fold_slice')


    def fold_slice_prepare(self) -> None:
        config = self.config
        fold_slice_output_dir = self.fold_slice_output_dir

        fold_slice_dict = {}
        fold_slice_dict['raw_data'] = config['io']['input_data_path']
        fold_slice_dict['result_dir'] = os.path.join(fold_slice_output_dir, '')
        fold_slice_dict.update(config['ptycho']['params'])

        if os.path.exists(os.path.join(fold_slice_output_dir)):
            shutil.rmtree(os.path.join(fold_slice_output_dir))
        os.makedirs(os.path.join(fold_slice_output_dir))

        with open(os.path.join(fold_slice_output_dir, 'setup.txt'), 'w') as f:
            f.write('\n\n')
            for key, value in fold_slice_dict.items():
                f.write(f"{key} {value}\n")
    

    def run(self) -> None:
        self.fold_slice_prepare()

        config = self.config
        fold_slice_path = config['ptycho']['path']
        setup_txt = os.path.join(self.fold_slice_output_dir, 'setup.txt')
        verbosity = config['io'].get('verbosity', 0)

        matlab_commands = [
            f"cd('{fold_slice_path}');",
            "cd('ptycho');",
            f"prepare_data('{setup_txt}');",
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
            print(f"fold_slice running. Set verbosity > 0 for full fold_slice output.")

        p.wait()


    def get_output_mat_path(self) -> str:
        config = self.config
        fold_slice_output_dir = self.fold_slice_output_dir
        roi_dir = os.path.join(
            fold_slice_output_dir,
            f"{config['ptycho']['params']['scan_number']}",
            f"roi{config['ptycho']['params']['roi_label']}"
        )
        output_dir = os.path.join(roi_dir, next(os.walk(roi_dir))[1][0])
        # image_path = os.path.join(output_dir, 'obj_phase_roi_sum', next(os.walk(os.path.join(output_dir, 'obj_phase_roi_sum')))[2][0])
        output_mat_path = os.path.join(output_dir, f"Niter{config['ptycho']['params']['Niter']}.mat")
        return output_mat_path
    

    def output(self) -> Any:
        if self._output is None:
            output_mat_path = self.get_output_mat_path()
            if not os.path.exists(output_mat_path):
                raise FileNotFoundError(f"Result directory {output_mat_path} does not exist. Please check the fold_slice output.")
            self._output = loadmat(output_mat_path)
        return self._output
    

    def recon_object(self) -> np.ndarray:
        if self._recon_object is None:
            output = self.output()
            self._recon_object = output['object'] # type: ignore
        return self._recon_object


    def recon_probe(self) -> np.ndarray:
        if self._recon_probe is None:
            output = self.output()
            self._recon_probe = output['probe'] # type: ignore
        return self._recon_probe


    def metric(self) -> float:
        if self._metric is None:
            output = self.output()
            self._metric = float(np.asarray(output['outputs']['fourier_error_out'][0]).squeeze()) # type: ignore
        return self._metric
