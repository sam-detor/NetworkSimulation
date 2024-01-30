import set_random_seed

import math 
import numpy as np 

from net_resource import Resource
from pair import Pair

class Optimizer: 
    def __init__(self):
        pass

    ########################################
    # Optimizer Iteration 
    ########################################
    def optimizerIteration(self, network):
        alpha = 0.5
        beta = 0.5
        bigBeta = 0.5 #for when user limit exceeded

        delta_n = [0 for i in range(len(network._pairs))]

        #network.updateResourceTput()
        for dest in network._destinations:
            # check if the destination limits should be changed to simulate external traffic 
            if np.random.rand() < network._externalTrafficProbability:
                # change the optimal tput to within 20% of the current optimal tput 
                dest._optimalTput = dest._optimalTput * np.random.uniform(0.7, 1.3) 
                print("Dest", dest._hashVal, "optimal tput changed to", dest._optimalTput)
        
            # if we do not have previous information about this resource 
            # set halfway between user limit and zero 
            # TODO: fix this logic 
            if dest._prevSlots== 0:
                slots = min(round((dest._tpUserLimit/2) /dest.tputPerSlot),  round(((2 * dest._optimalTput)/dest.tputPerSlot) - 1))
                newSetpoint = slots
            
            # we have previous information to optimize on 
            else: 
                dest._currentTput = dest.getActualTput()
                deltaTput = dest._currentTput - dest._prevTput
                deltaSetpoint = dest._currentSlots - dest._prevSlots

                if(deltaSetpoint != 0):
                    gradient = (deltaTput)/(deltaSetpoint)
                else:
                    gradient = 1 #hack for zero gradient, idk what to do here for sure
            
                decision = ""
                userLimitExceeded = False
                #adjust setpoint
                if gradient >= 0 and dest._currentTput < dest.getLimit():
                    # increase setpoint up to user specified limit 
                    newSetpoint = max(min(math.floor(dest._tpUserLimit/dest.tputPerSlot), round(((2 * dest._optimalTput)/dest.tputPerSlot) - 1), dest._currentSlots + max(round((alpha * dest._totUserWeights)), 1)), 1)
                    if newSetpoint == math.floor(dest._tpUserLimit/dest.tputPerSlot):
                        decision = "can't increase above setpoint estimation"
                    elif newSetpoint == round(((2 * dest._optimalTput)/dest.tputPerSlot) - 1):
                        decision = "can't increase or will go neg"
                    else:
                        decision = "increase"
                elif dest._currentTput > dest.getLimit():
                    newSetpoint = math.floor(dest._currentSlots * bigBeta)
                    decision = "decrease: user limit"
                    userLimitExceeded = True
                else:
                    newSetpoint = math.floor(dest._currentSlots * beta)
                    decision = "decrease: opt"


                for pair in network._destinations[dest]:
                    pair._tput = pair._currentSlots * dest._currentTput/dest._prevSlots
                
                network.printDestState(dest, gradient, decision, newSetpoint)
                deltaSlots = newSetpoint - dest._currentSlots
                dest._prevSlots = dest._currentSlots
                dest._prevTput = dest._currentTput
                addedSlots = 0
                if deltaSlots > 0:
                    #idk if we give everyone weighted 1 or just give that amount??
                    for pair in network._destinations[dest]:
                        if pair._active: 
                            newSlots = max(1,round(deltaSlots * (pair.getWeight()/dest._totUserWeights)))
                            pair._currentSlots += newSlots
                            #pair._tput += pair._currentSlots * dest._prevTput/dest._prevSlots
                            addedSlots += newSlots
                    dest._currentSlots += addedSlots
                else:
                    if(userLimitExceeded):
                        dest._currentSlots = 0
                        for pair in network._destinations[dest]:
                            if pair._active: 
                                #targetSlots = max(1, math.floor(beta * dest._prevSlots/dest._prevTput * (pair.getWeight()/dest._totUserWeights)))
                                pair._currentSlots = max(math.floor(pair._currentSlots * beta), 1)
                                #pair._currentTput = 
                                dest._currentSlots += pair._currentSlots

                    else:
                        network._destinations[dest].sort(key=lambda pair: pair._tput/(pair.getWeight()/dest._totUserWeights))
                        slotNotGiven = False
                        maxIndex = len(network._destinations[dest]) - 1
                        while(not slotNotGiven and maxIndex >= 0):
                            if(network._destinations[dest][maxIndex]._currentSlots > 1 and network._destinations[dest][maxIndex]._active):
                                network._destinations[dest][maxIndex]._currentSlots -= 1
                                #network._destinations[dest][maxIndex]._tput -= dest._prevTput/dest._prevSlots
                                dest._currentSlots -= 1
                                slotNotGiven = True
                            
                            maxIndex -= 1
                        
                        if (not slotNotGiven):
                            print("Couldn't decrease")
            
            # can't get the actual currentTput until we calculate the actual setpoint. TODO: think about the logic of this more 
            if dest._prevSlots == 0: 
                dest._currentTput = dest.getActualTput()
                dest._prevSlots= dest._currentSlots # this forces the gradient to be negative -- do we want this?
                dest._prevTput = dest._currentTput


        network.saveState()
        #network.printState()
        return network