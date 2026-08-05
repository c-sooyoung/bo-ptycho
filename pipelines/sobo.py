import os
import bo
import ptycho

def sobo_pipeline(config):

    result_dir = config['io']['result_dir']
    os.makedirs(result_dir, exist_ok=True)

    RANDOM_ITERS = config['job'].get('random_iters', 0)
    SOBO_ITERS   = config['job'].get('sobo_iters')
    METRIC = config['bo']['metric']
    PTYCHOENGINE = ptycho.engines[config['ptycho']['engine']]


    randombo = bo.RandomBOEngine(config)

    bo_txt = os.path.join(result_dir, "bo.txt")
    with open(bo_txt, "w") as f:
        f.write(f"    iter\tmetric\t{"\t".join([p[:7] for p in randombo.params])}\n")

    for j in range(RANDOM_ITERS):
        print(f"RANDOM sampling; iteration {j}")
        job_config = randombo.ask()
        ptycho_engine = PTYCHOENGINE(job_config)
        ptycho_engine.run(run_id=f"bo-{j:03d}")
        y_value = ptycho_engine.metric(METRIC)
        randombo.tell(job_config, y_value)
        with open(bo_txt, "a") as f:
            p = [f'{job_config['ptycho']['params'][key]:.2f}' for key in randombo.params]
            f.write(f"{j: 8d}\t{y_value:.4f}\t{"\t".join(p)}\n")


    sobo = bo.SingleObjectiveBOEngine(config)
    sobo.train_x = randombo.train_x
    sobo.train_y = randombo.train_y

    for j in range(SOBO_ITERS):
        print(f"SOBO sampling; iteration {RANDOM_ITERS + j}")
        job_config = sobo.ask()
        ptycho_engine = PTYCHOENGINE(job_config)
        ptycho_engine.run(run_id=f"bo-{RANDOM_ITERS + j:03d}")
        y_value = ptycho_engine.metric(METRIC)
        sobo.tell(job_config, y_value)
        with open(bo_txt, "a") as f:
            p = [f'{job_config['ptycho']['params'][key]:.2f}' for key in sobo.params]
            f.write(f"{RANDOM_ITERS + j: 8d}\t{y_value:.4f}\t{"\t".join(p)}\n")
