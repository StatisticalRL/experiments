
import time
from statisticalrl_experiments.utils import dump

def oneXpNoRender(env,learner,timeHorizon,root_folder):
    observation, info = env.reset()
    learner.reset(observation)
    cummeanreward = 0.
    cummeanrewards = []
    print("[Info] New initialization of ", learner.name(), ' for environment ',env.name)
    #print("Initial state:" + str(observation))
    for t in range(timeHorizon):
        state = observation
        action = learner.play(state)  # Get action
        observation, reward, done,truncated, info = env.step(action)
        learner.update(state, action, reward, observation)  # Update learners
        #print("info:",info, "reward:", reward)
        cummeanreward += info["mean"]
        cummeanrewards.append(cummeanreward)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            observation, info=env.reset()
            #break
    return cummeanrewards



def oneXpNoRenderWithDump(env,learner,timeHorizon,root_folder):
    cummeanrewards=oneXpNoRender(env,learner,timeHorizon,root_folder)

    tag = env.name + "_" + learner.name() + "_" + str(timeHorizon) +"_" + str(time.time())
    filename = dump(cummeanrewards,"cumMeans",tag,root_folder)
    return filename



def oneRunOptWithDump(env, opti_learner, timeHorizon, root_folder):
 ## Cumlative reward of optimal policy:
    opttimeHorizon = min(max((1000000, timeHorizon)),10**8)
    cummeanreward_opti = oneXpNoRender(env, opti_learner, opttimeHorizon, root_folder=root_folder)
    gain =  cummeanreward_opti[-1] / len(cummeanreward_opti)
    #print("Average gain is ", gain)
 # TODO: remove one [] and update computeCumulativeRegrets accordingly.
    opti_cumgain = [[t * gain for t in range(timeHorizon)]]

    tag = env.name + "_" + opti_learner.name() + "_" + str(timeHorizon) + "_" + str(time.time())
    filename= dump(opti_cumgain,"cumMeans",tag,root_folder)
    return filename