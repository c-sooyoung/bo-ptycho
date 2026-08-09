TODO:

- Clean up BO states
    - unify `BOEngine.__init__()`
- BO train_x/y transfer between engines for single job
- multi-GPU dispatcher
- template job sequences / yamls
    - mobo
- metric() function(s) for each ptycho engine
    - FRC score
- fold_slice: load diffractions / hdf5 files; change only param per job
    - restructure `FoldSlicePtychoEngine.__init__()` to load data but not params
        - write new `prepare_data.m`
