import os
import sys
import shutil
import subprocess
from scipy.io import loadmat
import numpy as np
from ptycho.base import PtychoEngine


class FoldSlicePtychoEngine(PtychoEngine):

    def __init__(self, config):
        super().__init__(config)
        self._output_dir = os.path.join(config['io']['result_dir'], 'fold_slice')
        self._fold_slice_path = self.config['ptycho']['path']
        self._setup_txt_path = os.path.join(self._output_dir, 'setup.txt')
        self.metric_methods = {
            'log_fourier': self._log_fourier_metric,
        }


    def run(self, run_id="") -> None:

        # generate setup.txt for fold_slice input
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

        # execute matlab
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

        # get output of fold_slice
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

        log_fourier_error = self._log_fourier_metric()
        os.makedirs(os.path.join(self.config['io']['result_dir'], "mat"), exist_ok=True)
        os.makedirs(os.path.join(self.config['io']['result_dir'], "tiff"), exist_ok=True)
        shutil.copy(mat_path, os.path.join(self.config['io']['result_dir'], "mat", f"{log_fourier_error:.4f}_{run_id}.mat"))
        shutil.copy(image_path, os.path.join(self.config['io']['result_dir'], "tiff", f"{log_fourier_error:.4f}_{run_id}.tiff"))


    def metric(self, names):
        requested_names = [names] if isinstance(names, str) else list(names)
        unsupported = [n for n in requested_names if n not in self.metric_methods]
        if unsupported: raise ValueError(f"Metric(s) {unsupported} not implemented. Available metrics: {self.metric_methods.keys()}")

        if isinstance(names, (list, tuple)):
            return [self.metric_methods[name]() for name in names]  # MOBO
        else:
            return self.metric_methods[names]()                     # SOBO


    def _log_fourier_metric(self) -> float:
        if self._output is None:
            raise RuntimeError("Must call run() before requesting metrics.")
        return -np.log(float(self._output['outputs']['fourier_error_out'][0][0].squeeze()[-1]))
