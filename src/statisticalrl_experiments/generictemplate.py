from src.statisticalrl_experiments.fullExperiment import runLargeMulticoreExperiment

#######################
# Import registered environments
import statisticalrl_environments as bW
#######################



#######################
# Instantiate one environment
env = bW.make('river-swim-6')


# Import some learners
import statisticalrl_learners.Generic.Qlearning as ql
import statisticalrl_learners.MDPs_discrete.UCRL3 as ucrl3
import statisticalrl_learners.MDPs_discrete.IMED_RL as imedrl
import statisticalrl_learners.MDPs_discrete.Optimal.OptimalControl  as opt
#######################
# Instantiate a few learners to be compared:
agents = []
nS = env.observation_space.n
nA = env.action_space.n
delta=0.05
#agents.append( [random.Random, {"env": env.env}])
agents.append( [ql.Qlearning, {"nS":nS, "nA":nA}])
agents.append( [ucrl3.UCRL3_lazy, {"nS":nS, "nA":nA, "delta":delta}])
agents.append(([imedrl.IMEDRL, {"nbr_states":nS, "nbr_actions":nA}]))

#############################
# Compute oracle policy:
oracle = opt.build_opti(env.name, env, env.observation_space.n, env.action_space.n)

#######################
# Run a full experiment
#######################
import os
from src.statisticalrl_experiments.utils import get_project_root_dir
ROOT= get_project_root_dir()+"/main/templates/"+"generictemplate"+"_results/"
os.makedirs(ROOT,exist_ok=True)

runLargeMulticoreExperiment(env, agents, oracle, timeHorizon=5000, nbReplicates=64,root_folder=ROOT)

#######################
# Plotting Regret directly from dump files of past runs:
#files = plR.search_dump_cumRegretfiles("RiverSwim-6-v0", root_folder=ROOT)
#plR.plot_results_from_dump(files, 500)
