# TODO:

- fold_slice:
    - load diffractions / hdf5 files; change only param per job
    - restructure `FoldSlicePtychoEngine.__init__()` to load data but not params
        - write new `prepare_data.m`
    - start fold_slice from config.yaml instead of setup.txt
- BO train_x/y transfer between engines for single job
- metric() function(s) for each ptycho engine
    - FRC score
- separate `config` into `bo_config` and `ptycho_config`; let `BOEngine` have no knowledge of ptychography and vice versa.
- add GPU version of ExamplePtychoEngine
- change `PtychoEngine.metric()` to accept list of names and return dict
- `GridSampler`
    - `.__init__()` should create a grid
    - `.ask()` should remove those items from the grid
- change job.sub to job.sh
    - dynamic jobname
    - pre-check result directory
- clean up examples for Si-FIB paper version

# TODAY:
- rename `BOEngine` to `Sampler`
- separation of available GPUs and parallel sample batches
- prepare next batch for efficient GPU use

