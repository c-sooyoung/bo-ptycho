import os
import traceback
import multiprocessing as mp
import shutil

import bo
import ptycho

def run_ptycho_worker(
    worker_id,
    gpu_token,
    job_config,
    metric,
    run_id,
    result_queue,
):
    # This process, and MATLAB launched from it,
    # can see exactly one GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_token

    try:
        PTYCHOENGINE = ptycho.engines[job_config["ptycho"]["engine"]]

        ptycho_engine = PTYCHOENGINE(job_config)
        ptycho_engine.run(run_id=run_id)
        y_value = ptycho_engine.metric(metric)

        result_queue.put(
            (worker_id, y_value, None)
        )

    except Exception:
        result_queue.put(
            (worker_id, None, traceback.format_exc())
        )


def run_batch(
    ctx,
    gpu_tokens,
    job_configs,
    metric,
    iteration,
):
    result_queue = ctx.Queue()
    processes = []

    for i, job_config in enumerate(job_configs):
        # Important: unique run_id for simultaneous jobs.
        run_id = f"bo-{iteration:03d}-{i:02d}"

        p = ctx.Process(
            target=run_ptycho_worker,
            args=(
                i,
                gpu_tokens[i],
                job_config,
                metric,
                run_id,
                result_queue,
            ),
        )

        p.start()
        processes.append(p)

    # Four jobs are now running concurrently.

    results = [
        result_queue.get()
        for _ in processes
    ]

    # Synchronization barrier.
    for p in processes:
        p.join()

    # Completion order is arbitrary.
    results.sort(key=lambda x: x[0])

    for worker_id, _, error in results:
        if error is not None:
            raise RuntimeError(
                f"Ptycho worker {worker_id} failed:\n{error}"
            )

    return [y_value for _, y_value, _ in results]


def sobo_pipeline(config):

    result_dir = config["io"]["result_dir"]
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    RANDOM_ITERS = config["job"].get("random_iters", 0)
    SOBO_ITERS = config["job"].get("sobo_iters", 0)
    METRIC = config["bo"]["metric"]
    BO_BATCH = config["bo"]["batch"]

    # SLURM should expose the four GPUs allocated to this job.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")

    if visible is None:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES is not set"
        )

    gpu_tokens = [
        token.strip()
        for token in visible.split(",")
        if token.strip()
    ]

    if len(gpu_tokens) < BO_BATCH:
        raise RuntimeError(
            f"Expected {BO_BATCH} allocated GPUs, got {len(gpu_tokens)}"
        )

    # Explicitly use spawn for CUDA / MATLAB isolation.
    ctx = mp.get_context("spawn")

    # ---------------------------------------------------------
    # Random sampling
    # ---------------------------------------------------------

    randombo = bo.RandomBOEngine(config)

    for j in range(RANDOM_ITERS):
        print(
            f"RANDOM sampling; iteration {j}",
            flush=True,
        )

        job_configs = randombo.ask(n=BO_BATCH)

        y_values = run_batch(
            ctx=ctx,
            gpu_tokens=gpu_tokens,
            job_configs=job_configs,
            metric=METRIC,
            iteration=j,
        )

        # Only the parent touches BO state / train_x / train_y.
        for job_config, y_value in zip(
            job_configs,
            y_values,
        ):
            randombo.tell(
                job_config,
                y_value,
            )

    # ---------------------------------------------------------
    # SOBO
    # ---------------------------------------------------------

    sobo = bo.SingleObjectiveBOEngine(config)

    sobo.train_x = randombo.train_x
    sobo.train_y = randombo.train_y

    for j in range(SOBO_ITERS):
        iteration = RANDOM_ITERS + j

        print(
            f"SOBO sampling; iteration {iteration}",
            flush=True,
        )

        job_configs = sobo.ask(n=BO_BATCH)

        y_values = run_batch(
            ctx=ctx,
            gpu_tokens=gpu_tokens,
            job_configs=job_configs,
            metric=METRIC,
            iteration=iteration,
        )

        for job_config, y_value in zip(
            job_configs,
            y_values,
        ):
            sobo.tell(
                job_config,
                y_value,
            )
