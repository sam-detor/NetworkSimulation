import random
import string
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
import numpy as np
import os
from datetime import datetime
from collections import defaultdict


########################################
# Pair Functions 
########################################
class Pair:
    def __init__(self, idNum, tpLimit, resources, weight, start=0, end=1):
        self._id = idNum
        self._tpLimit = tpLimit
        self._resources = resources # first element in resource array is the destination
        self._destination = resources[0]
        self._tput = 0
        self._proposedDecisions = [1]
        self._proposedTputs = list()
        self._currentSlots = 0
        self._prevSlots = 0
        self._weight = weight
        self._start = start
        self._end = end # percentage of time pair is active during simulation 
        self._active = False 


    def updateLimit(self,new_limit):
        self._tpLimit = new_limit
    
    def updateTput(self,tput):
       self._tput = tput

    def getLimit(self):
        return self._tpLimit
    
    def getTput(self):
       return self._tput
    
    def getWeight(self):
       return self._weight

    def getId(self): 
        return self._id 
    
    def getDestination(self):
        return self._destination


########################################
# Resource Functions 
########################################
class Resource:
    tputPerSlot = 1
    def __init__(self, tpLimit, idNum, optTput):
        self._tpUserLimit = tpLimit
        self._optimalTput = optTput
        self._currentTput = 0
        self._prevTput = 0
        #self._currentTputSetpoint = 0
        self._currentSlots = 0
        self._prevSlots =0
        #self._prevTputSetpoint = 0
        self._totUserWeights = 0
        self._hashVal = idNum
        self._numPairs = 0

    def __hash__(self) -> int:
        return hash(id)
    def __eq__(self,other):
        if isinstance(other, Resource):
            return self._hashVal == other._hashVal
        
    def updateLimit(self,new_limit):
        self._tpUserLimit = new_limit
    
    def getLimit(self):
        return self._tpUserLimit
    
    def quadFunc(self, setpoint):
        return (-1/self._optimalTput) * (setpoint - self._optimalTput) * (setpoint - self._optimalTput) + self._optimalTput
    
    def rightEdgeFunc(self, setpoint):
        return (1.25 * self._optimalTput) - (0.25 * setpoint)
    
    '''def getActualTput(self):
        actualTput = self._currentTputSetpoint
        # if greater than optimal tput, the actual tput will begin to decrease (neg gradient)
        if self._currentTputSetpoint > self._optimalTput: 
            actualTput = self.rightEdgeFunc(self._currentTputSetpoint)

        return actualTput '''
    
    
    def getActualTput(self):
        #if(self._currentTputSetpoint - (1.5 * self._optimalTput) > 0 and self.rightEdgeFunc(self._currentTputSetpoint) < (self._optimalTput * 0.8)):
            #return self.rightEdgeFunc(self._currentTputSetpoint)
        #elif (-self._currentTputSetpoint + (.5 * self._optimalTput) > 0 and self.leftEdgeFunc(self._currentTputSetpoint) < (self._optimalTput * 0.8)):
            #return self.leftEdgeFunc(self._currentTputSetpoint)
        #else:
        return self.quadFunc(self._currentSlots * self.tputPerSlot)



########################################
# Network Functions 
########################################
class Network:
    
    def __init__(self, resources, pairs, destinations, pairIndictorFunction, resourceIndicatorFunction, numPairsPerResource=0):
        self._pairs = pairs
        self._resources = resources
        self._destinations = destinations
        self._numPairsPerResource = numPairsPerResource
        self._pairIndictorFunction = pairIndictorFunction
        self._resourceIndicatorFunction = resourceIndicatorFunction
        self._x_values = list()
        self._resource_data =  [[] for _ in range(len(resources))]
        self._pair_data = [[] for _ in range(len(pairs))]
        self._destination_data = [[] for _ in range(len(destinations))]
        self._destination_slots = [[] for _ in range(len(destinations))]
        self._resource_limits =  [[] for _ in range(len(resources))]
        self._pair_limits = [[] for _ in range(len(pairs))]
        self._destination_limits = [[] for _ in range(len(destinations))]


    def __init__(self, num_resources, num_pairs, externalTrafficProbability, pairIndictorFunction, resourceIndicatorFunction, numPairsPerResource=0):

        # If network has external traffic, the destinations will have fluctuating optimal capacities 
        self._externalTrafficProbability = externalTrafficProbability
        self._numPairsPerResource = numPairsPerResource

        # Initialize resources
        self._resources = []
        pairResourceAssigments = []
        i = 0
        for _ in range(num_resources):
            tp_limit = random.randint(100,150)  # Adjust the range as needed
            tp_optimal = random.randint(50,150)
            resource = Resource(tp_limit, i, tp_optimal)
            self._resources.append(resource)
            i += 1
            pair_indices = []
            
            if self._numPairsPerResource == 0: 
                numResourcesRand = random.randint(1,num_resources)
                pair_indices = random.choices(range(num_pairs), k=numResourcesRand)
                
            else:
                pair_indices = random.choices(range(num_pairs), k=numPairsPerResource - 1)

            pairResourceAssigments.append(pair_indices)

            
        
        #self._resources.sort(key=lambda resource: resource.getLimit())
        self._destinations = dict()
        self._pairsOnResources = dict()

        # Initialize pairs
        self._pairs = []
        for i in range(num_pairs):
            tp_limit =  100 # Adjust the range as needed, change this!!!! to be based on resource limits U_r / numPairsweighted 

            # picking multiple resources 
            #resource_indicies = random.sample(range(len(self._resources)), random.randint(1, len(self._resources))) 
            #resources = [self._resources[i] for i in resource_indicies]

            # # if we are picking the resource randomly
            # if self._numPairsPerResource == 0: 
            #     # picking one resource (destination) randomly 
            #     numResourcesRand = random.randint(1,len(self._resources))
            #     resource_indices = random.choices(range(len(self._resources)), k=numResourcesRand)
            #     resources = list()
            #     for j in resource_indices:
            #         resources.append(self._resources[j])
            # else: 
            #     # picking the resource based on the number of pairs per resource
            #     randNumResources = random.randint(1,len(self._resources)/2)
            #     resource_indices = random.choices(range(len(self._resources)), k=randNumResources)
            #     resources = list()
            #     for j in resource_indices:
            #         if self._resources[j]._numPairs < numPairsPerResource:
            #             resources.append(self._resources[j])
            #             self._resources[j]._numPairs += 1
            #     if len(resources) == 0:
            #         resources.append(self._resources[random.randint(0,len(self._resources) - 1)])

            resources = [self._resources[i % num_resources]]
            for j in range(len(pairResourceAssigments)):
                for k in range(len(pairResourceAssigments[j])):
                    if pairResourceAssigments[j][k] == i:
                        resources.append(self._resources[j])
            
            resources = list(set(resources))
            # this is variable weight pairs 
            #pair = Pair(tp_limit, resources, random.random() * 10) #first resource is dest
                
            # all pairs are equally weighted here 
            weight = 1
            if np.random.rand() < 0.25: 
                start = 0 
                end = 0.6
                pair = Pair(i, tp_limit, resources, weight, start, end) 
            elif np.random.rand() < 0.25:
                start = 0.25
                end = 1
                pair = Pair(i, tp_limit, resources, weight, start, end)
            else: 
                pair = Pair(i, tp_limit, resources, weight) # all pairs are equally weighted here 


            self._pairs.append(pair)
            
            #update destinations map
            if pair.getDestination() in self._destinations:
                self._destinations[pair.getDestination()].append(pair)
            else:
                self._destinations[pair.getDestination()] = [pair]

            #update resources map:
            for resource in pair._resources:
                if resource in self._pairsOnResources:
                    self._pairsOnResources[resource].append(pair)
                else:
                    self._pairsOnResources[resource] = [pair]
        

        self._resources.sort(key=lambda resource: resource._hashVal)

        self._x_values = list()
        self._pair_data = defaultdict(list) 
        self._resource_data = [[] for _ in range(len(self._resources))]
        self._resource_slots = [[] for _ in range(len(self._resources))]
        self._resource_limits =  [[] for _ in range(len(self._resources))]
        self._pair_limits = [[] for _ in range(len(self._pairs))]
        self._resource_optimal = [[] for _ in range(len(self._resources))]

        
        self._pairIndictorFunction = pairIndictorFunction
        self._resourceIndicatorFunction = resourceIndicatorFunction

        for resource in self._pairsOnResources:
            print("Resource", resource._hashVal, ":")
            for pair in self._pairsOnResources[resource]:
                print("\t pair ", pair.getId())
        print(pairResourceAssigments)
    
    def init_tput(self):
        for dest in self._destinations:
            #tput = random.randint(1,2 * dest._optimalTput)
            slots = min(round((dest._tpUserLimit / 2) /dest.tputPerSlot),  round(((2 * dest._optimalTput)/dest.tputPerSlot) - 1)) #start half way inbetween user limit and zero
            dest._currentSlots = slots
            dest._totUserWeights = 0 
            # TODO: clean this part up 
            #for pair in self._destinations[dest]:
                #dest._totUserWeights += pair.getWeight()
            #for pair in self._destinations[dest]:
                #pair.updateTput(tput * pair.getWeight()/dest._totUserWeights)
            #for resource in pair._resources:
                #resource._tput += tput
            #pair._destination._currentTputSetpoint += tput
            #pair._destination._currentTput += pair._destination.getActualTput()
                
    def getDestinations(self): 
        return self._destinations

    def getPairs(self): 
        return self._pairs

    def updateResourceTput(self):
        for dest in self._destinations:
            CalculatedSetpoint = 0
            CalculatedWeights = 0
            #dest._currentTputSetpoint = 0
            #dest.__totUserWeights = 0
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
            self._resource_slots[i].append(resource._prevSlots)
            self._resource_optimal[i].append(resource._optimalTput)
            i += 1

        i = 0
        for pair in self._pairs:
            self._pair_data[pair.getId()].append(pair._currentSlots)
            # self._pair_data[i].append(pair.getTput())
            self._pair_limits[i].append(pair.getLimit())
            i += 1

        # i = 0
        # for dest in self._destinations:
        #     self._destination_slots[i].append(dest._prevSlots) # or current?
        #     self._destination_data[i].append(dest._currentTput)
        #     self._destination_limits[i].append(dest.getLimit())
        #     self._destination_optimal[i].append(dest._optimalTput)
        #     i += 1
    
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
    
    def updatePairTput(self):
        for resource in self._pairsOnResources:
            resource._currentTput = resource.getActualTput()
            for pair in self._pairsOnResources[resource]:
                #print("Pair:", pair.getId(), " Resource: ", resource._hashVal)
                pair._proposedTputs.append(pair._currentSlots * resource._currentTput/resource._currentSlots)
        for pair in self._pairs:
            pair._tput = min(pair._proposedTputs)
            pair._proposedTputs = list()

           



    ########################################
    # Optimizer Iteration 
    ########################################
    def optimizerIteration(self):
        alpha = 0.5
        beta = 0.5
        bigBeta = 0.5 #for when user limit exceeded
        noinfo = True

        if self._resources[0]._prevSlots != 0:
            self.updatePairTput()
            noinfo = False

        #self.updateResourceTput()
        for resource in self._pairsOnResources:
            # check if the destination limits should be changed to simulate external traffic 
            if np.random.rand() < self._externalTrafficProbability:
                # change the optimal tput to within 20% of the current optimal tput 
                resource._optimalTput = resource._optimalTput * np.random.uniform(0.7, 1.3) 
                print("Resource", resource._hashVal, "optimal tput changed to", resource._optimalTput)
        
            # if we do not have previous information about this resource 
            # set halfway between user limit and zero 
            # TODO: fix this logic 
            if resource._prevSlots == 0:
                slots = min(round((resource._tpUserLimit/2) /resource.tputPerSlot),  round(((2 * resource._optimalTput)/resource.tputPerSlot) - 1))
                newSetpoint = slots
                for pair in self._pairsOnResources[resource]:
                    pair._proposedDecisions.append(newSetpoint * pair.getWeight()/resource._totUserWeights)
            
            # we have previous information to optimize on 
            else: 
                resource._currentTput = resource.getActualTput()
                deltaTput = resource._currentTput - resource._prevTput
                deltaSetpoint = resource._currentSlots - resource._prevSlots

                if(deltaSetpoint != 0):
                    gradient = (deltaTput)/(deltaSetpoint)
                else:
                    gradient = 1 #hack for zero gradient, idk what to do here for sure
            
                decision = ""
                userLimitExceeded = False
                #adjust setpoint
                if gradient > 0 and resource._currentTput < resource.getLimit():
                    # increase setpoint up to user specified limit 
                    newSetpoint = max(min(math.floor(resource._tpUserLimit/resource.tputPerSlot), round(((2 * resource._optimalTput)/resource.tputPerSlot) - 1), resource._currentSlots + max(round((alpha * resource._totUserWeights)), 1)), 1)
                    if newSetpoint == math.floor(resource._tpUserLimit/resource.tputPerSlot):
                        decision = "can't increase above setpoint estimation"
                    elif newSetpoint == round(((2 * resource._optimalTput)/resource.tputPerSlot) - 1):
                        decision = "can't increase or will go neg"
                    else:
                        decision = "increase"
                elif resource._currentTput > resource.getLimit():
                    newSetpoint = math.floor(resource._currentSlots * bigBeta)
                    decision = "decrease: user limit"
                    userLimitExceeded = True
                else:
                    newSetpoint = math.floor(resource._currentSlots * beta)
                    decision = "decrease: opt"


                #for pair in self._pairsOnResources[resource]:
                    #pair._tput = pair._currentSlots * dest._currentTput/dest._prevSlots
                
                self.printDestState(resource, gradient, decision, newSetpoint)
                deltaSlots = newSetpoint - resource._currentSlots
                resource._prevSlots = resource._currentSlots
                resource._prevTput = resource._currentTput
                addedSlots = 0
                if deltaSlots > 0:
                    #idk if we give everyone weighted 1 or just give that amount??
                    for pair in self._pairsOnResources[resource]:
                        if pair._active: 
                            print("resource", resource._hashVal, "pair", pair.getId(), "proposedIncrease")
                            proposedNewSlots = max(1,round(deltaSlots * (pair.getWeight()/resource._totUserWeights)))
                            pair._proposedDecisions.append(pair._currentSlots + proposedNewSlots)
                            #pair._tput += pair._currentSlots * dest._prevTput/dest._prevSlots
                            #addedSlots += newSlots
                    #dest._currentSlots += addedSlots
                else:
                    if(userLimitExceeded):
                        #dest._currentSlots = 0
                        for pair in self._destinations[resource]:
                            if pair._active: 
                                #targetSlots = max(1, math.floor(beta * dest._prevSlots/dest._prevTput * (pair.getWeight()/dest._totUserWeights)))
                                pair._proposedDecisions.append(max(math.floor(pair._currentSlots * beta), 1))
                                #pair._currentTput = 
                                #dest._currentSlots += pair._currentSlots

                    else:
                        self._pairsOnResources[resource].sort(key=lambda pair: pair._tput/(pair.getWeight()/resource._totUserWeights))
                        slotNotGiven = False
                        maxIndex = len(self._pairsOnResources[resource]) - 1
                        while(not slotNotGiven and maxIndex >= 0):
                            if(self._pairsOnResources[resource][maxIndex]._currentSlots > 1 and self._pairsOnResources[resource][maxIndex]._active):
                                self._pairsOnResources[resource][maxIndex]._proposedDecisions.append(self._pairsOnResources[resource][maxIndex]._currentSlots - 1)
                                #self._destinations[dest][maxIndex]._tput -= dest._prevTput/dest._prevSlots
                                #dest._currentSlots -= 1
                                slotNotGiven = True
                            
                            maxIndex -= 1
                        
                        if (not slotNotGiven):
                            print("Couldn't decrease")


            # can't get the actual currentTput until we calculate the actual setpoint. TODO: think about the logic of this more 
            if resource._prevSlots == 0: 
                resource._currentTput = resource.getActualTput()
                resource._prevSlots= resource._currentSlots # this forces the gradient to be negative -- do we want this?
                resource._prevTput = resource._currentTput 


        if not noinfo:
            for pair in self._pairs:
                print("pair", pair.getId(), "decisions", pair._proposedDecisions, "current", pair._currentSlots)
            for pair in self._pairs:
                currentSlots = pair._currentSlots
                if len(pair._proposedDecisions) != 0:
                    proposedMin = min(pair._proposedDecisions)
                    proposedMax = max(pair._proposedDecisions)
                
                    if(proposedMin == currentSlots):
                        pair._currentSlots = proposedMax
                    else:
                        pair._currentSlots = proposedMin
            
                    pair._proposedDecisions = []
        else:
            for pair in self._pairs:
                pair._currentSlots = min(pair._proposedDecisions)
            
        for resource in self._pairsOnResources:
                resource.currentSlots = 0
                for pair in self._pairsOnResources[resource]:
                    resource.currentSlots += pair._currentSlots


        
        #choose pair decision
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
                        for resource in pair._resources:
                            resource._currentSlots -= pair._currentSlots
                            resource._totUserWeights -= pair.getWeight()
                            if(resource._currentSlots < 0):
                                print("resource", resource._hashVal, "pair", pair.getId(),"Something Is Wrong!!!!")
                        #pair.getDestination()._currentSlots -= pair._currentSlots
                        pair._currentSlots = 0
                        #pair.getDestination()._totUserWeights -= pair.getWeight()
                else:
                    if pair._start <= percentComplete < pair._end: # within the pair's active range
                        pair._active = True
                        for resource in pair._resources:
                           resource._totUserWeights += pair.getWeight()

            self.optimizerIteration()
        
        return self._x_values, self._pair_data, self._pair_limits, \
                self._resource_data, self._resource_limits, \
                self._resource_optimal, self._resource_slots, \
        



########################################
# Indicator Functions 
########################################

# currently unused
# TODO: add pair success rate indicator function 
def myPairIndicatorFunction(pair, maxC, minCVal, gradient):
    beta = 0.9
    alpha = 1
    if gradient < 0:
        return minCVal * pair.getWeight() * beta
    else:
        return (maxC * pair.getWeight()/pair._destination._totUserWeights) + (alpha * pair.getWeight()/pair._destination._totUserWeights)
    
# currently unused 
def myResourceIndicatorFunction(pair):
    # if exceeded resource limit, calculate the new proportional allocation for pair  
    if pair._resources[0]._currentTput > pair._resources[0].getLimit():
        return pair._resources[0].getLimit() * pair.getWeight()/pair._resources[0]._totUserWeights
    else:
        return float('inf')






def main():
    #  network parameters 
    numResources = 4 
    numPairs = 5
    numIterations = 100
    externalTrafficProbability = 0
    numPairsPerResource = 2 # 0 means random resource assignment, otherwise it is the number of pairs per resource

    # initialize network simulation 
    myNetwork = Network(numResources, numPairs, externalTrafficProbability, myPairIndicatorFunction,myResourceIndicatorFunction, numPairsPerResource)
    myNetwork.printNetworkState()
    data = myNetwork.simulate(numIterations)
    print(data)

    # save plots 
    timestamp_str = datetime.now().strftime("%m-%d_%H-%M-%S")
    folder_path = os.path.join('/Users/samdetor/networkSim/', f"folder_{timestamp_str}/")

    # create the new folder
    try:
        os.makedirs(folder_path)
        print(f"New folder '{folder_path}' created successfully.")
    except OSError as e:
        print(f"Error creating folder: {e}")




    ################################################
    # Plot 1: All Pairs 
    ################################################
        
    #plt.subplot(3, 1, 1)  # 3 rows, 1 column, plot 1
    for i in range(len(data[1])):
        myLabel1 = "Pair" + str(i)+ "Throuput"
        myLabel2 = "Pair" + str(i) + "Limit"
        plt.plot(data[0], data[1][i], label=myLabel1) # this is fuckign breaking 
        plt.plot(data[0], data[2][i], label=myLabel2)
    plt.title('Pairs')
    plt.legend()

    plt.savefig(folder_path + "Pairs.png")
    plt.clf()
    



    #############################################
    # Plot 2: Destinations (1 plot for each dest)
    #############################################
    
    #plt.subplot(3, 1, 3)  # 3 rows, 1 column, plot 3
    for i in range(len(data[3])):
        plt.clf()
        myLabel1 = "Resource" + str(i) + "Throuput"
        myLabel2 = "Resource" + str(i) + "Limit"
        myLabel3 = "Resource" + str(i) + "Opt"
        myLabel4 = "Resource" + str(i) + "Target"
        plt.plot(data[0], data[3][i], label=myLabel1)
        plt.plot(data[0], data[4][i], label=myLabel2)
        plt.plot(data[0], data[5][i], label=myLabel3)
        plt.plot(data[0], data[6][i], label=myLabel4)
        plt.title("Resource" + str(i))
        plt.legend()
        plt.savefig(folder_path + "Resource" + str(i) + ".png")

    plt.clf()

    # how can i print out the goal setpoint  


    ################################################
    # Plot 3: All Resources 
    ################################################
        
    #plt.subplot(3, 1, 2)  # 3 rows, 1 column, plot 2
    #print("Data 3:", data[7])
    #print("Data 3 len:", data[7])
    for i in range(len(data[3])):
        myLabel1 = "Resource" + str(i) + "Throuput"
        myLabel2 = "Resource" + str(i) + "Limit"
        plt.plot(data[0], data[3][i], label=myLabel1)
        plt.plot(data[0], data[4][i], label=myLabel2)
    plt.title('Resources')
    plt.legend()
    plt.savefig(folder_path + "Resources.png")


    ################################################
    # Plot 4: Pairs on Each Resource 
    ################################################
    resources = myNetwork._pairsOnResources
    pairData = data[1]

    fig_size = (10, 6)

    for resource, pairs in resources.items():
        plt.figure(figsize=fig_size)
        
        for pair in pairs:
            plt.plot(data[0], pairData[pair.getId()], label="Pair" + str(pair.getId()))

        plt.title("Pairs on Resource " + str(resource._hashVal))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(folder_path + "Resource" + str(resource._hashVal) + "_Pairs_Plot.png", bbox_inches='tight')


if __name__ == "__main__":
    main()    