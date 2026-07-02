import statisticalrl_experiments.BatchMABs.oneRun as oR
import statisticalrl_experiments.parallelRuns as pR
import statisticalrl_experiments.analyzeRuns as aR
import statisticalrl_experiments.plotResults as plR
from statisticalrl_experiments.utils import clear_auxiliaryfiles
import time
#from src.statisticalrl_experiments.utils import get_project_root_dir
#ROOT= get_project_root_dir()+"/src/results"
import os
ROOT="results/"

def runLargeBatchMulticoreExperiment(env, agents, oracle, timeHorizon=1000,  opttimeHorizon= 10000, nbReplicates=100, root_folder=ROOT):
    os.mkdir(root_folder)
    envFullName = env.name

    # opti_learner=opt.build_opti(envFullName, env.env, env.observation_space.n, env.action_space.n)
    learners = [x[0](**x[1]) for x in agents]

    print(">" * 20 + "\n Deploying " + str(nbReplicates) + " replicate experiments with time horizon " + str(
        timeHorizon) + " on multicore CPU")
    dump_cumRewardsAlgos = []
    dump_cumGapsAlgos = []
    names = []
    meanelapsedtimes = []

    for learner in learners:
        names.append(learner.name())
        dump_cumRewards, meanelapsedtime = pR.multicoreRuns(envFullName, learner, nbReplicates, timeHorizon,
                                                            oR.oneXpNoRenderWithDump, root_folder=root_folder)
        dump_cumRewardsAlgos.append(dump_cumRewards)
        meanelapsedtimes.append(meanelapsedtime)

    ## Cumulative reward of optimal policy:
    # if (opttimeHorizon):
    #     opttimeHorizon = opttimeHorizon
    # else:
    #     opttimeHorizon = min(max((10000, timeHorizon)), 10 ** 8)
    dump_cumRewardsopt = oR.oneRunOptWithDump(env, oracle, timeHorizon, root_folder=root_folder)
    dump_cumRewardsAlgos.append(dump_cumRewardsopt)

    ## Report statistics and compute regret:
    # print('************** ANALYSIS **************')
    timestamp = str(time.time())
    logfilename = root_folder + "logfile_" + env.name + "_" + timestamp + ".txt"
    logfile = open(logfilename, 'w')
    logfile.write("Environment " + env.name + "\n")
    logfile.write("Optimal policy is: " + str(oracle.policy) + "\n")
    logfile.write("Learners " + str([learner.name() for learner in learners]) + "\n")
    logfile.write("Time horizon is " + str(timeHorizon) + ", nb of replicates is " + str(nbReplicates) + "\n")
    [logfile.write(str(names[i]) + " average runtime is " + str(meanelapsedtimes[i]) + "\n") for i in range(len(names))]
    mean, median, quantile1, quantile2, times = aR.computeCumulativeRegrets(names, dump_cumRewardsAlgos, timeHorizon,
                                                                            envFullName, root_folder=root_folder)
    title = f"{env.name}"
    plR.plotCumulativeRegrets(names, envFullName, title, mean, median, quantile1, quantile2, times, timeHorizon,
                              logfile=logfile, timestamp=timestamp, root_folder=root_folder)
    # print("*********************************************")
    clear_auxiliaryfiles(env, root_folder)
    print("\n[INFO] A log-file has been generated in ", logfilename)