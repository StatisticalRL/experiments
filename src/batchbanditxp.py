"""
main_batch.py — entry point for batch-bandit experiments
=========================================================
Two experiment families:

1. FIXED-SCHEDULE experiments  (run_fixed_schedule_experiment)
   All baselines are compared on a predetermined batch-size schedule
   (constant / linear / quadratic / cubic / exp / sup-exp).

2. BABA-SCHEDULE experiments   (run_baba_schedule_experiment)
   All baselines **and** BABA itself run on the BABA static time grid.
   This allows a fair comparison: every agent sees the same batch sizes,
   isolating the effect of the arm-pulling strategy from the schedule.
"""

from math import exp
from random import randint

from statisticalrl_experiments.BatchMABs.massiveRuns  import runLargeBatchMulticoreExperiment as xp

import statisticalrl_environments.register as bW
from statisticalrl_learners.BatchMABs.Oracle import BatchOracle as Oracle
from statisticalrl_learners.BatchMABs.BCB   import BCB, BCBnaif
from statisticalrl_learners.BatchMABs.BABA  import BABA


# ---------------------------------------------------------------------------
# 1. Fixed-schedule experiments
# ---------------------------------------------------------------------------

def run_fixed_schedule_experiment(env_name, target_num_steps=1e4, nbReplicates=100,
                                   root_folder="results-batch/"):
    """Run all baselines + BABA on a classical fixed batch-size schedule."""

    env = bW.make(env_name)
    f = env.batchsize

    # if "constant" in env_name:
    #     f = lambda t: 10
    # elif "linear" in env_name:
    #     f = lambda t: t
    # elif "quadratic" in env_name:
    #     f = lambda t: t**2
    # elif "cubic" in env_name:
    #     f = lambda t: t**3
    # elif "double-exp" in env_name:
    #     f = lambda t: exp(2**t)
    # elif "exp" in env_name:
    #     f = lambda t: exp(t)
    # elif "abrupt" in env_name:
    #     f = lambda t: 100 if t<3 else t**3
    # else:
    #     raise ValueError("Unknown experiment name")

    timesteps = 0
    batchsequence=[]
    i = 1
    while (timesteps + f(i)) <= target_num_steps:
        timesteps += f(i)
        batchsequence.append(f(i))
        i += 1

    timeHorizon = i - 1
    B   = getattr(env, 'reward_max', 1.0)
    K   = env.nbArms
    kl  = 'bernoulli' if 'bern' in env_name else 'gaussian'

    # Compute BABA schedule separately — only used for phase/epoch info.
    # The environment still drives batch sizes; BABA cycles its phase grid
    # via modulo when the fixed schedule has more rounds than the BABA grid.
    from statisticalrl_learners.BatchMABs.baba_schedule import compute_baba_grid
    _, phase_labels, epoch_ids, epoch_I = compute_baba_grid(
        100_000, K, I1=2000, alpha=3)
    print(phase_labels)

    baba_kwargs = dict(
        nbArms       = K,
        bound        = B,
        phase_labels = phase_labels,
        epoch_ids    = epoch_ids,
        epoch_I      = epoch_I,
        kl_type      = kl,
    )

    agents = [
        (BABA,      baba_kwargs),
        (BCBnaif,   {"nbArms": K, "bound": B}),
        #(BatchIMED,     {"nbArms": K, "bound": B, "batchagnostic": False}),
        #(BatchIMED2,     {"nbArms": K, "bound": B, "batchagnostic": False}),
        #(BatchIMED,     {"nbArms": K, "bound": B, "batchagnostic": True}),
        #(BatchIMED2,     {"nbArms": K, "bound": B, "batchagnostic": True}),
        #(BBIMED,     {"nbArms": K, "bound": B}),
        #(IMEDnaive, {"nbArms": K, "bound": B}),
        #(IMED,      {"nbArms": K, "bound": B}),
        #(BSIMED, {"nbArms": K, "bound": B}),
    ]

    oracle = Oracle(env)

    print("-" * 60)
    print(f" Environment: {env_name}")
    print(f" Schedule: {batchsequence}")
    print(f" B={B}, timeHorizon={timeHorizon}, replicates={nbReplicates}")
    print("-" * 60)
    xp(env, agents, oracle, timeHorizon=timeHorizon,
       nbReplicates=nbReplicates, root_folder=root_folder)
    print("-" * 60 + "\n")


# ---------------------------------------------------------------------------
# 2. BABA-schedule experiments
# ---------------------------------------------------------------------------

def run_baba_schedule_experiment(dist_type, means=None, target_num_steps=1e5,
                                  nbReplicates=100, alpha=3,
                                  root_folder="results-baba/"):
    """
    Run ALL baselines plus BABA on the BABA static time grid.

    The environment drives the batch-size sequence; every agent (including the
    non-BABA baselines) receives the same sequence of batch sizes.  Only BABA
    uses the phase/epoch information to decide *which* arms to pull; the
    baselines use their own index rules on whatever batch size they receive.

    Parameters
    ----------
    dist_type         : 'trunc' or 'bern'
    means             : list of arm means (default: [0.1, 0.4, 0.7, 0.9])
    target_num_steps  : approximate total arm-pull budget
    nbReplicates      : number of independent replications
    alpha             : BABA epoch growth exponent (paper uses 3)
    root_folder       : where to write results
    """
    if means is None:
    #    means = [0.1, 0.4, 0.7, 0.9]
        means = [0.2, 0.6, 0.8, 0.8, 0.95, 0.9]

    # Build env + recover full schedule (phase_labels, epoch_ids, epoch_I)
    # so BABA can be initialised with the exact same grid the env uses.
    env, n_rounds, I1, phase_labels, epoch_ids, epoch_I = bW.make_baba_env(
        dist_type, means=means,
        target_num_steps=int(target_num_steps),
        alpha=alpha,
        I1=2000,   # matches Jin et al. (ICML 2021): I1=2000, alpha=3
    )

    K   = len(means)
    B   = getattr(env, 'reward_max', 1.0)
    kl  = 'bernoulli' if dist_type == 'bern' else 'gaussian'

    # BABA receives the full schedule; baselines only see the batch sizes
    baba_kwargs = dict(
        nbArms       = K,
        bound        = B,
        phase_labels = phase_labels,
        epoch_ids    = epoch_ids,
        epoch_I      = epoch_I,
        kl_type      = kl,
    )

    agents = [
        # (BCB,       {"nbArms": K, "bound": B}),
        (BABA, baba_kwargs),
        (BCBnaif, {"nbArms": K, "bound": B}),
        #(BatchIMED2,     {"nbArms": K, "bound": B, "batchagnostic": False}),
        #(BatchIMED2,     {"nbArms": K, "bound": B, "batchagnostic": True}),
        #(BBIMED, {"nbArms": K, "bound": B}),
        #(IMEDnaive, {"nbArms": K, "bound": B}),
        # (IMED,      {"nbArms": K, "bound": B}),
        # (BSIMED, {"nbArms": K, "bound": B}),
    ]

    oracle = Oracle(env)

    print("-" * 60)
    print(f" BABA schedule: dist={dist_type}, I1={I1}, α={alpha}")
    print(f" n_rounds={n_rounds} (total pulls≈{int(target_num_steps):.2e}), "
          f"replicates={nbReplicates}")
    print("-" * 60)
    xp(env, agents, oracle, timeHorizon=n_rounds,
       nbReplicates=nbReplicates, root_folder=root_folder)
    print("-" * 60 + "\n")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_TRUNC_FIXED = [
    # "trunc-constant-batch",
    # "trunc-linear-batch",
    # "trunc-quadratic-batch",
    # "trunc-cubic-batch",
    # "trunc-exp-batch",
    # "trunc-double-exp-batch",
]

_BERN_FIXED = [
    "bern-constant-batch",
    "bern-linear-batch",
    "bern-quadratic-batch",
    "bern-cubic-batch",
    "bern-exp-batch",
    "bern-exotic1-batch",
    "bern-exotic2-batch",
    #"bern-abrupt-batch",
    "bern-double-exp-batch",
]

if __name__ == "__main__":

    # ── Fixed-schedule baseline experiments ─────────────────────────────
    for env_name in _TRUNC_FIXED + _BERN_FIXED:
        run_fixed_schedule_experiment(env_name, target_num_steps=1e4,
                                      nbReplicates=64,
                                      root_folder="results-batch-rb/")
    #print("Start BABA experiments")
    # ── BABA-schedule experiments (all agents including BABA) ────────────
    # for dist in ["bern","trunc"]:
    #      run_baba_schedule_experiment(dist, target_num_steps=2e4,
    #                                   nbReplicates=64,
    #                                   root_folder="results-baba2/")