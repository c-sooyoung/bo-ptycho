from typing import Any
import time
import random
from ptycho.base import PtychoEngine

class ExamplePtychoEngine(PtychoEngine):

    # initialize job/engine-specific variables here
    # e.g. self.output_dir
    
    def __init__(self, config):
        super().__init__(config)
        self._metric = 0.0

    # run single ptychography job based on `config`
    def run(self, run_id="") -> None:
        print(f"[{run_id}] [ExamplePtychoEngine] Sleeping for 0.1 second.")
        time.sleep(0.1)
