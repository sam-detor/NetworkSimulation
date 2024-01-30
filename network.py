import random 
import math 
import numpy as np
from collections import defaultdict

from net_resource import Resource 
from pair import Pair


########################################
# Network Functions 
########################################
class Network:
    
    def __init__(self, resources, pairs, destinations, numPairsPerResource=0):
        self._pairs = pairs
        self._resources = resources
        self._destinations = destinations
        self._numPairsPerResource = numPairsPerResource
        self._x_values = list()
        self._resource_data =  [[] for _ in range(len(resources))]
        self._pair_data = [[] for _ in range(len(pairs))]
        self._destination_data = [[] for _ in range(len(destinations))]
        self._destination_slots = [[] for _ in range(len(destinations))]
        self._resource_limits =  [[] for _ in range(len(resources))]
        self._pair_limits = [[] for _ in range(len(pairs))]
        self._destination_limits = [[] for _ in range(len(destinations))]


    def __init__(self, num_resources, num_pairs, externalTrafficProbability, numPairsPerResource=0):

        # If network has external traffic, the destinations will have fluctuating optimal capacities 
        self._externalTrafficProbability = externalTrafficProbability
        self._numPairsPerResource = numPairsPerResource

        # Initialize resources
        self._resources = []
        i = 0
        for _ in range(num_resources):
            tp_limit = random.randint(100,150)  # Adjust the range as needed
            tp_optimal = random.randint(50,150)
            resource = Resource(tp_limit, i, tp_optimal)
            self._resources.append(resource)
            i += 1
        
        self._resources.sort(key=lambda resource: resource.getLimit())
        self._destinations = dict()

        # Initialize pairs
        self._pairs = []
        for i in range(num_pairs):
            tp_limit =  100 # Adjust the range as needed, change this!!!! to be based on resource limits U_r / numPairsweighted 

            # if we are picking the resource randomly
            if self._numPairsPerResource == 0: 
                # picking one resource (destination) randomly 
                resource_index = random.choice(range(len(self._resources)))
                resource = [self._resources[resource_index]]
            else: 
                # picking the resource based on the number of pairs per resource 
                resource_index = i // self._numPairsPerResource  # Use integer division (//) instead of modulus (%)
                resource = [self._resources[resource_index]]
                
            # all pairs are equally weighted here 
            weight = 1
            if np.random.rand() < 0.25: 
                start = 0 
                end = 0.6
                pair = Pair(i, tp_limit, resource, weight, start, end) 
            elif np.random.rand() < 0.25:
                start = 0.25
                end = 1
                pair = Pair(i, tp_limit, resource, weight, start, end)
            else: 
                pair = Pair(i, tp_limit, resource, weight) # all pairs are equally weighted here 


            self._pairs.append(pair)
            
            #update destinations map
            if pair.getDestination() in self._destinations:
                self._destinations[pair.getDestination()].append(pair)
            else:
                self._destinations[pair.getDestination()] = [pair]

        self._x_values = list()
        self._resource_data =  [[] for _ in range(len(self._resources))]
        self._pair_data = defaultdict(list) 
        self._destination_data = [[] for _ in range(len(self._destinations))]
        self._destination_slots = [[] for _ in range(len(self._destinations))]
        self._resource_limits =  [[] for _ in range(len(self._resources))]
        self._pair_limits = [[] for _ in range(len(self._pairs))]
        self._destination_limits = [[] for _ in range(len(self._destinations))]
        self._destination_optimal = [[] for _ in range(len(self._destinations))]
    
    def init_tput(self):
        for dest in self._destinations:
            #tput = random.randint(1,2 * dest._optimalTput)
            slots = min(round((dest._tpUserLimit / 2) /dest.tputPerSlot),  round(((2 * dest._optimalTput)/dest.tputPerSlot) - 1)) #start half way inbetween user limit and zero
            dest._currentSlots = slots
            dest._totUserWeights = 0 

                
    def getDestinations(self): 
        return self._destinations

    def getPairs(self): 
        return self._pairs

    def updateResourceTput(self):
        for dest in self._destinations:
            CalculatedSetpoint = 0
            CalculatedWeights = 0
            for pair in self._destinations[dest]:
                if pair._active: 
                    CalculatedSetpoint += pair.getTput()
                    CalculatedWeights += pair.getWeight()
            print("Dest", dest._hashVal)
            print("\t actual setpoint:", dest._currentSlots * dest.tputPerSlot)
            print("\t calculated Setpoint:", CalculatedSetpoint)


    def saveState(self):
        self._x_values.append(len(self._x_values))
        i = 0
        for resource in self._resources:
            self._resource_data[i].append(resource._currentTput)
            self._resource_limits[i].append(resource.getLimit())
            i += 1

        i = 0
        for pair in self._pairs:
            self._pair_data[pair.getId()].append(pair._currentSlots)
            # self._pair_data[i].append(pair.getTput())
            self._pair_limits[i].append(pair.getLimit())
            i += 1

        i = 0
        for dest in self._destinations:
            self._destination_slots[i].append(dest._prevSlots) # or current?
            self._destination_data[i].append(dest._currentTput)
            self._destination_limits[i].append(dest.getLimit())
            self._destination_optimal[i].append(dest._optimalTput)
            i += 1
    
    def clearState(self):
        self._x_values = list()
        self._resource_data =  [[] for _ in range(len(self._resources))]
        self._pair_data = defaultdict(list)
        self._destination_data = [[] for _ in range(len(self._destinations))]
        self._destination_data = [[] for _ in range(len(self._destinations))]
        self._resource_limits =  [[] for _ in range(len(self._resources))]
        self._pair_limits = [[] for _ in range(len(self._pairs))]
        self._destination_limits = [[] for _ in range(len(self._destinations))]
        
    def printDestState(self,dest, gradient, decision, newSetpoint):
        print("Dest " + str(dest._hashVal) + "State")
        print("\t current tput: ", dest._currentTput)
        print("\t current slots: ", dest._currentSlots)
        print("\t prev tput : ", dest._prevTput)
        print("\t prev slots: ", dest._prevSlots)
        print("\t gradient: ", gradient)
        print("\t opt tput: ", dest._optimalTput)
        print("\t limit:", dest.getLimit())
        print("\t decision: ", decision)
        print("\t new slots: ", newSetpoint)
        print("\t tot user weights: ", dest._totUserWeights)
        
    def printState(self):
        i = 0
        print("Resources\n")
        for resource in self._resources:
            print("Resource ", i, ": Throuput: ", resource._currentTput, " Setpoint: ", resource._currentSlots, "Optimal: ", resource._optimalTput, " Limit: ", resource.getLimit(), "\n")
            i+=1
        
        i = 0
        print("Pairs\n")
        for pair in self._pairs:
            print("Pair ", i, ": Throuput: ", pair.getTput(), " Limit: ", pair.getLimit(), "\n")
            i+=1
    
    def printNetworkState(self):
        print("Pairs:")
        for i in range(len(self._pairs)):
            print("Pair ", i)
            print("Dest: Resource ", self._pairs[i]._destination._hashVal)

    ########################################
    # Optimizer Iteration 
    ########################################
    def optimizerIteration(self):
        alpha = 0.5
        beta = 0.5
        bigBeta = 0.5 #for when user limit exceeded

        #self.updateResourceTput()
        for dest in self._destinations:
            # check if the destination limits should be changed to simulate external traffic 
            if np.random.rand() < self._externalTrafficProbability:
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


                for pair in self._destinations[dest]:
                    pair._tput = pair._currentSlots * dest._currentTput/dest._prevSlots
                
                self.printDestState(dest, gradient, decision, newSetpoint)
                deltaSlots = newSetpoint - dest._currentSlots
                dest._prevSlots = dest._currentSlots
                dest._prevTput = dest._currentTput
                addedSlots = 0
                if deltaSlots > 0:
                    #idk if we give everyone weighted 1 or just give that amount??
                    for pair in self._destinations[dest]:
                        if pair._active: 
                            newSlots = max(1,round(deltaSlots * (pair.getWeight()/dest._totUserWeights)))
                            pair._currentSlots += newSlots
                            #pair._tput += pair._currentSlots * dest._prevTput/dest._prevSlots
                            addedSlots += newSlots
                    dest._currentSlots += addedSlots
                else:
                    if(userLimitExceeded):
                        dest._currentSlots = 0
                        for pair in self._destinations[dest]:
                            if pair._active: 
                                #targetSlots = max(1, math.floor(beta * dest._prevSlots/dest._prevTput * (pair.getWeight()/dest._totUserWeights)))
                                pair._currentSlots = max(math.floor(pair._currentSlots * beta), 1)
                                #pair._currentTput = 
                                dest._currentSlots += pair._currentSlots

                    else:
                        self._destinations[dest].sort(key=lambda pair: pair._tput/(pair.getWeight()/dest._totUserWeights))
                        slotNotGiven = False
                        maxIndex = len(self._destinations[dest]) - 1
                        while(not slotNotGiven and maxIndex >= 0):
                            if(self._destinations[dest][maxIndex]._currentSlots > 1 and self._destinations[dest][maxIndex]._active):
                                self._destinations[dest][maxIndex]._currentSlots -= 1
                                #self._destinations[dest][maxIndex]._tput -= dest._prevTput/dest._prevSlots
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


        self.saveState()
        #self.printState()

    def simulate(self, numIterations):
        self.clearState()
        self.init_tput()
        self.saveState()
        for i in range(numIterations):
            percentComplete = (i/numIterations) 
            for pair in self._pairs:
                if pair._active: 
                    if percentComplete > pair._end: 
                        pair._active = False
                        pair.updateTput(0)
                        pair.getDestination()._currentSlots -= pair._currentSlots
                        pair._currentSlots = 0
                        pair.getDestination()._totUserWeights -= pair.getWeight()
                else:
                    if pair._start <= percentComplete < pair._end: # within the pair's active range
                        pair._active = True
                        pair.getDestination()._totUserWeights += pair.getWeight()

            self.optimizerIteration()
        
        return self._x_values, self._pair_data, self._pair_limits, self._destination_data, \
                        self._destination_limits, self._destination_optimal, self._destination_slots, \
                        self._resource_data, self._resource_limits