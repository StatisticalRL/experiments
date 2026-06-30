import statisticalrl_experiments.BatchMABs.oneRun as BoR
import statisticalrl_experiments.parallelRuns as pR
import statisticalrl_experiments.analyzeRuns as aR
import statisticalrl_experiments.plotResults as plR
from statisticalrl_experiments.utils import clear_auxiliaryfiles
import time
#from src.statisticalrl_experiments.utils import get_project_root_dir
#ROOT= get_project_root_dir()+"/src/results"
import os
ROOT="results/"

def runLargeBatchMulticoreExperiment(env, agents, oracle, timeHorizon=1000, nbReplicates=100, root_folder=ROOT):
    ''' Better adapted for bandits (run oracle directly) '''
    try:
        os.mkdir(root_folder)
    except:
        ()

    envFullName= env.name

    #opti_learner=opt.build_opti(envFullName, env.env, env.observation_space.n, env.action_space.n)
    learners = [x[0](**x[1]) for x in agents]

    print(">" * 20 + "\n Deploying "+str(nbReplicates)+" replicate experiments with time horizon "+str(timeHorizon) +" on multicore CPU")
    dump_cumRewardsAlgos = []
    dump_cumGapsAlgos = []
    names = []
    meanelapsedtimes = []

    for learner in learners:
        names.append(learner.name())
        dump_cumRewardsAndGaps, meanelapsedtime = pR.multicoreRuns(envFullName, learner, nbReplicates, timeHorizon,BoR.oneXpNoRenderWithDump, root_folder=root_folder)
        dump_cumRewards,dump_cumGaps=list(zip(*dump_cumRewardsAndGaps))
        dump_cumRewardsAlgos.append(list(dump_cumRewards))
        dump_cumGapsAlgos.append(list(dump_cumGaps))
        meanelapsedtimes.append(meanelapsedtime)

    ## Add cumulative reward of optimal policy:
    dump_cumRewardsAngGapopt = BoR.oneXpNoRenderWithDump(env, oracle, timeHorizon,root_folder=root_folder)
    dump_cumRewardsopt,dump_cumGapsopt=dump_cumRewardsAngGapopt
    dump_cumRewardsAlgos.append(dump_cumRewardsopt)

    print("<" * 20)
    ## Report statistics and compute regret:
    print(">" * 20 + "\n Analyzing results and computing statistics")
    timestamp = str(time.time())
    logfilename=root_folder+"logfile_"+env.name+"_"+timestamp+".txt"
    logfile = open(logfilename,'w')
    logfile.write("Environment "+env.name +"\n")
    logfile.write("Optimal policy is: " + str(oracle.policy)+"\n")
    logfile.write("Learners "+str([learner.name() for learner in learners]) +"\n")
    logfile.write("Time horizon is "+ str(timeHorizon) + ", nb of replicates is "+ str(nbReplicates) +"\n")
    [logfile.write(str(names[i])+ " average runtime is "+ str(meanelapsedtimes[i])  +"\n") for i in range(len(names))]
    mean,median, quantile1,quantile2,times = aR.computeCumulativeRegrets(names, dump_cumRewardsAlgos, timeHorizon, envFullName, root_folder=root_folder)
    mean2, median2, quantile12, quantile22, times2 = aR.computeCumulativeGaps(names, dump_cumGapsAlgos, timeHorizon,
                                                                         envFullName, root_folder=root_folder)

    print("<" * 20)
    print(">" * 20 + "\n Plotting results")
    title = f"{env.name}"
    plR.plotCumulativeRegrets(names, envFullName, title, mean, median, quantile1, quantile2, times, timeHorizon, logfile=logfile, timestamp=timestamp, root_folder=root_folder)
    plR.plotCumulativeRegrets(names, envFullName, title, mean2, median2, quantile12, quantile22, times2, timeHorizon, logfile=logfile, timestamp=timestamp+"-1", root_folder=root_folder)
    print("<" * 20)
    print(">" * 20 + "\n Cleaning auxiliary files")
    clear_auxiliaryfiles(env, root_folder)
    print("<" * 20)
    print("[INFO] A log-file has been generated in ",logfilename)