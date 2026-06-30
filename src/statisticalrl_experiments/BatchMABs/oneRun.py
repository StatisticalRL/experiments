

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
        batchobservation, batchreward, done, truncated, info = env.step(batchaction) # Get response
        learner.batchupdate(batchaction, batchobservation)  # Update learners
        B = info["nextbatchsize"]
        #print({"batchobservation": batchobservation, "batchreward": batchreward, "pseudo-gap":max(env.means) * B-sum(batchreward), "info": info})
        cumreward += sum(batchreward)
        cummean += info["mean"]
        cumgap += max(env.means) * B-sum(batchreward)
        cumrewards.append(cumreward)
        cummeans.append(cummean)
        cumgaps.append(cumgap)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))

    return cummeans,cumgaps



def oneXpNoRenderWithDump(env,learner,timeHorizon,root_folder):
    cummeans, cumgaps=oneXpNoRender(env,learner,timeHorizon,root_folder)

    tag = env.name + "_" + learner.name() + "_" + str(timeHorizon) + "_" + str(time.time())
    #filenameR = dump(cumrewards, "cumRewards", tag, root_folder)
    filenameM = dump(cummeans, "cumMeans", tag, root_folder)
    filenameG = dump(cumgaps, "cumGaps", tag, root_folder)
    return filenameM,filenameG#filenameR, filenameM, filenameG