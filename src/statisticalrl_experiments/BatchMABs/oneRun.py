

import time
from statisticalrl_experiments.utils import dump


def oneXpNoRender(env, learner, timeHorizon, root_folder):
    observation, info = env.reset()
    learner.reset()
    B = info["nextbatchsize"]
    cumreward = 0
    cummean = 0
    cumgap = 0
    cumrewards = []
    cummeans = []
    cumgaps = []
    for t in range(timeHorizon):
        batchaction = learner.batchplay(B)  # Get action
        batchobservation, batchreward, done, truncated, info = env.step(batchaction)  # Get response
        learner.batchupdate(batchaction, batchobservation)  # Update learners
        # print({"batchobservation": batchobservation, "batchreward": batchreward, "pseudo-gap":max(env.means) * B-sum(batchreward), "info": info})
        cumreward += sum(batchreward)
        cummean += info["mean"]
        cumgap += max(env.means) * len(batchaction) - sum(batchreward)
        cumrewards.append(cumreward)
        cummeans.append(cummean)
        cumgaps.append(cumgap)
        B = info["nextbatchsize"]

        if done:
            print("Episode finished after {} timesteps".format(t + 1))

    return cummeans#,cumgaps



def oneXpNoRenderWithDump(env,learner,timeHorizon,root_folder):
    cummeans=oneXpNoRender(env,learner,timeHorizon,root_folder)

    tag = env.name + "_" + learner.name() + "_" + str(timeHorizon) + "_" + str(time.time())
    #filenameR = dump(cumrewards, "cumRewards", tag, root_folder)
    filenameM = dump(cummeans, "cumMeans", tag, root_folder)
    #filenameG = dump(cumgaps, "cumGaps", tag, root_folder)
    return filenameM#,filenameG#filenameR, filenameM, filenameG



def oneRunOptWithDump(env, opti_learner, timeHorizon, root_folder):
 ## Cumlative reward of optimal policy:
    opttimeHorizon = timeHorizon#min(max((1000000, timeHorizon)),10**8)
    cummeanreward_opti = oneXpNoRender(env, opti_learner, opttimeHorizon, root_folder=root_folder)
    #gain =  cummeanreward_opti[-1] / len(cummeanreward_opti)
    #print("Average gain is ", gain)
    #opti_cumgain = [[t * gain for t in range(timeHorizon)]]
    opti_cumgain = [cummeanreward_opti]

    tag = env.name + "_" + opti_learner.name() + "_" + str(timeHorizon) + "_" + str(time.time())
    filename= dump(opti_cumgain,"cumMeans",tag,root_folder)
    return filename