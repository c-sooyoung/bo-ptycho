from typing import Any
import time
import random
from ptycho.base import PtychoEngine

class ExamplePtychoEngine(PtychoEngine):
    
    def __init__(self, config):
        super().__init__(config)

    # run single ptychography job based on `config`
    def run(self, run_id="") -> None:
        print(f"[{run_id}] [ExamplePtychoEngine] Sleeping for 0.1 second.")
        time.sleep(0.1)

    def metric(self, names):
        if isinstance(names, (list, tuple)):
            return [0.0] * len(names)
        else:
            return 0.0
