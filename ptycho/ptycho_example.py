from typing import Any
import time
import random
from ptycho.ptycho_base import PtychoEngine

class ExamplePtychoEngine(PtychoEngine):

    # initialize job/engine-specific variables here
    # e.g. self.output_dir
    
    def __init__(self, config):
        super().__init__(config)

    # run single ptychography job based on `config`
    def run(self) -> None:
        print("[ExamplePtychoEngine] Sleeping for 0.1 second.")
        time.sleep(0.1)
    
    # save outputs to self._output
    def output(self) -> Any:
        if self._output is None:
            pass
        return self._output
    
    # return reconstructed object (numpy array)
    def recon_object(self) -> Any:
        if self._recon_object is None:
            output = self.output()
            pass
        return self._recon_object

    # return reconstructed probe (numpy array)
    def recon_probe(self) -> Any:
        if self._recon_probe is None:
            output = self.output()
            pass
        return self._recon_probe
    
    # return metric value for Bayesian optimization
    def metric(self) -> float:
        if self._metric is None:
            output = self.output()
            self._metric = random.uniform(0, 1)
        return self._metric
