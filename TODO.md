# TODO:

- unify `BOEngine.__init__()`
- BO train_x/y transfer between engines for single job
- multi-GPU dispatcher
    - synchronous batched BO
    - asynchronous BO
- template job sequences / yamls
    - mobo
- metric() function(s) for each ptycho engine
    - FRC score
- separate `config` into `bo_config` and `ptycho_config`; let `BOEngine` have no knowledge of ptychography and vice versa.
- add GPU version of ExamplePtychoEngine
- change `PtychoEngine.metric()` to accept list of names and return dict


# TODAY:

- fold_slice: load diffractions / hdf5 files; change only param per job
    - restructure `FoldSlicePtychoEngine.__init__()` to load data but not params
        - write new `prepare_data.m`
