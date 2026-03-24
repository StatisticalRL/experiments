
import statisticalrl_environments as srl
from statisticalrl_environments.register import make
env = make('river-swim-6')

from statisticalrl_environments.fulldemo import print_registered_environments,random_environment,all_environments

# List register environments:
print_registered_environments()
# random_environment()
# all_environments()

from statisticalrl_experiments.fullExperiment import runLargeMulticoreExperiment as xp

#######################
# Import registered environments
import statisticalrl_environments.register as bW

#######################

# Instantiate one environment
#env = bW.make('river-swim-6')
env = bW.make('random-12')
nS = env.observation_space.n
nA = env.action_space.n

#######################
# Import some learners
from statisticalrl_learners.Generic.Random import Random as rd
from statisticalrl_learners.MDPs_discrete.UCRL3 import UCRL3_lazy as ucrl3
from statisticalrl_learners.MDPs_discrete.IMED_RL import IMEDRL as imedrl
from statisticalrl_learners.MDPs_discrete.PSRL import PSRL as psrl
import statisticalrl_learners.MDPs_discrete.Optimal.OptimalControl as opt

#######################
# List a few learners to be compared:
agents = []
agents.append([rd, {"env": env}])
agents.append([psrl, {"nS": nS, "nA": nA,"delta": 0.05}])
agents.append([ucrl3, {"nS": nS, "nA": nA, "delta": 0.05}])
agents.append(([imedrl, {"nbr_states": nS, "nbr_actions": nA}]))

#############################
# Compute oracle policy:
oracle = opt.build_opti(env.name, env, env.observation_space.n, env.action_space.n)

#######################
# Run a full experiment
# This function produces all results including plots, logs, etc in the folder "root_folder"
#######################
xp(env, agents, oracle, timeHorizon=1000, nbReplicates=16, root_folder="results/")

#######################
# Plotting Regret directly from dump files of past runs (here until time horizon tplot=500):
#######################
# from statisticalrl_experiments.plotResults import search_dump_cumRegretfiles, plot_results_from_dump
# files = search_dump_cumRegretfiles("RiverSwim-S6-v0", root_folder="results/")
# if files:
#    plot_results_from_dump(files, tplot=500)