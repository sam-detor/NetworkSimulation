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

            # picking multiple resources 
            #resource_indicies = random.sample(range(len(self._resources)), random.randint(1, len(self._resources))) 
            #resources = [self._resources[i] for i in resource_indicies]

            # if we are picking the resource randomly
            if self._numPairsPerResource == 0: 
                # picking one resource (destination) randomly 
                resource_index = random.choice(range(len(self._resources)))
                resource = [self._resources[resource_index]]
            else: 
                # picking the resource based on the number of pairs per resource 
                resource_index = i // self._numPairsPerResource  # Use integer division (//) instead of modulus (%)
                resource = [self._resources[resource_index]]

            # this is variable weight pairs 
            #pair = Pair(tp_limit, resources, random.random() * 10) #first resource is dest
                
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

        
        self._pairIndictorFunction = pairIndictorFunction
        self._resourceIndicatorFunction = resourceIndicatorFunction
    
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
        beta = 0.9
        bigBeta = 0.9 #for when user limit exceeded

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
                    newSetpoint = min(math.floor(dest._tpUserLimit/dest.tputPerSlot), round(((2 * dest._optimalTput)/dest.tputPerSlot) - 1), dest._currentSlots + min(round((alpha * dest._totUserWeights)), 1)) 
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



                   
                
                


                
            
                #dest._currentTputSetpoint = newSetpoint

            #dest._currentTputSetpoint = 0 
            # propagate new setpoint to pairs
            # slotsLeft = newSetpoint
            # self._destinations[dest].sort(key=lambda pair: pair.getWeight()/dest._totUserWeights) # sorted by most weighted
            # currentPair = 0

            # for pair in self._destinations[dest]:
            #     pair._prevSlots = pair._currentSlots
            #     pair._currentSlots = 0
            # while(slotsLeft > 0):
            #     self._destinations[dest][currentPair]._currentSlots += 1
            #     if currentPair >= len(self._destinations[dest]):
            #         currentPair = 0
            #     else:
            # #         currentPair += 1
            # dest._currentSlots = newSetpoint
            # slotsLeft = newSetpoint
            # for pair in self._destinations[dest]:
            #     if pair._active: 
            #         newSlots = max(1,round(newSetpoint * (pair.getWeight()/dest._totUserWeights)))
            #         newTput = newSlots * dest.tputPerSlot
            #         slotsLeft = slotsLeft - newSlots
            #         pair.updateTput(newTput)
            #         pair._prevSlots = pair._currentSlots
            #         pair.currentSlots = newSlots
            #         print("Pair ", pair.getId(), "has" , newSlots, "slots")
            #         #dest._currentSlots += 1
            # #print("Current dest slots: ", dest._currentSlots)
            # #print("Slots left: ", slotsLeft)
            # if slotsLeft > 0:
            #     totDestWeight = dest._totUserWeights
            #     self._destinations[dest].sort(key=lambda pair: pair.getWeight()/totDestWeight)
            #     currentPair = 0
            #     while(slotsLeft > 0):
            #         #print("current pair is", currentPair)
            #         if self._destinations[dest][currentPair]._active:
            #             self._destinations[dest][currentPair]._currentSlots += 1
            #             slotsLeft  = slotsLeft - 1
            #             #print("Pair ", self._destinations[dest][currentPair].getId(), "got one extra slot")
            #         if currentPair >= len(self._destinations[dest]) - 1:
            #            currentPair = 0
            #         else:
            #             currentPair += 1


            
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
                        pair.getDestination()._totUserWeights -= pair.getWeight()
                else:
                    if pair._start <= percentComplete < pair._end: # within the pair's active range
                        pair._active = True
                        pair.getDestination()._totUserWeights += pair.getWeight()

            self.optimizerIteration()
        
        return self._x_values, self._pair_data, self._pair_limits, self._destination_data, \
                        self._destination_limits, self._destination_optimal, self._destination_slots, \
                        self._resource_data, self._resource_limits
        



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
    numPairs = 12
    numIterations = 100
    externalTrafficProbability = 0
    numPairsPerResource = 3 # 0 means random resource assignment, otherwise it is the number of pairs per resource

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
        myLabel1 = "Dest" + str(i) + "Throuput"
        myLabel2 = "Dest" + str(i) + "Limit"
        myLabel3 = "Dest" + str(i) + "Opt"
        myLabel4 = "Dest" + str(i) + "Target"
        plt.plot(data[0], data[3][i], label=myLabel1)
        plt.plot(data[0], data[4][i], label=myLabel2)
        plt.plot(data[0], data[5][i], label=myLabel3)
        plt.plot(data[0], data[6][i], label=myLabel4)
        plt.title("Destination" + str(i))
        plt.legend()
        plt.savefig(folder_path + "Destination" + str(i) + ".png")

    plt.clf()

    # how can i print out the goal setpoint  


    ################################################
    # Plot 3: All Resources 
    ################################################
        
    #plt.subplot(3, 1, 2)  # 3 rows, 1 column, plot 2
    print("Data 3:", data[7])
    print("Data 3 len:", data[7])
    for i in range(len(data[7])):
        myLabel1 = "Resource" + str(i) + "Throuput"
        myLabel2 = "Resource" + str(i) + "Limit"
        plt.plot(data[0], data[7][i], label=myLabel1)
        plt.plot(data[0], data[8][i], label=myLabel2)
    plt.title('Resources')
    plt.legend()
    plt.savefig(folder_path + "Resources.png")


    ################################################
    # Plot 4: Pairs on Each Resource 
    ################################################
    destinations = myNetwork.getDestinations()
    pairData = data[1]

    fig_size = (10, 6)

    for dest, pairs in destinations.items():
        plt.figure(figsize=fig_size)
        
        for pair in pairs:
            plt.plot(data[0], pairData[pair.getId()], label="Pair" + str(pair.getId()))

        plt.title("Pairs on Destination " + str(dest._hashVal))
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(folder_path + "Destination" + str(dest._hashVal) + "_Pairs_Plot.png", bbox_inches='tight')


if __name__ == "__main__":
    main()    