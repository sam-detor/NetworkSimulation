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
    def __init__(self, idNum, tpLimit, resources, weight):
        self._id = idNum
        self._tpLimit = tpLimit
        self._resources = resources #first element in resource array is the destination
        self._destination = resources[0]
        self._tput = 0
        self._weight = weight

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
    def __init__(self, tpLimit, idNum, optTput):
        self._tpUserLimit = tpLimit
        self._optimalTput = optTput
        self._currentTput = 0
        self._prevTput = 0
        self._currentTputSetpoint = 0
        self._prevTputSetpoint = 0
        self._totUserWeights = 0
        self._hashVal = idNum

    def __hash__(self) -> int:
        return hash(id)
    def __eq__(self,other):
        if isinstance(other, Resource):
            return self._hashVal == other._hashVal
        
    def updateLimit(self,new_limit):
        self._tpUserLimit = new_limit
    
    def updateTputSetpoint(self,tput):
       self._tputSetpoint = tput
    
    def getLimit(self):
        return self._tpUserLimit
    
    def getTput(self):
       return self._tput
    
    def quadFunc(self, setpoint):
        return (-1/self._optimalTput) * (setpoint - self._optimalTput) * (setpoint - self._optimalTput) + self._optimalTput
    
    def rightEdgeFunc(self, setpoint):
        return (1.25 * self._optimalTput) - (0.25 * setpoint)
    
    def getActualTput(self):
        actualTput = self._currentTputSetpoint
        # if greater than optimal tput, the actual tput will begin to decrease (neg gradient)
        if self._currentTputSetpoint > self._optimalTput: 
            actualTput = self.rightEdgeFunc(self._currentTputSetpoint)

        return actualTput 
    
    
    """  def getActualTput(self):
        if(self._currentTputSetpoint - (1.5 * self._optimalTput) > 0 and self.rightEdgeFunc(self._currentTputSetpoint) < (self._optimalTput * 0.8)):
            return self.rightEdgeFunc(self._currentTputSetpoint)
        elif (-self._currentTputSetpoint + (.5 * self._optimalTput) > 0 and self.leftEdgeFunc(self._currentTputSetpoint) < (self._optimalTput * 0.8)):
            return self.leftEdgeFunc(self._currentTputSetpoint)
        else:
            return self.quadFunc(self._currentTputSetpoint)
 """


########################################
# Network Functions 
########################################
class Network:
    
    def __init__(self, resources, pairs, destinations, pairIndictorFunction, resourceIndicatorFunction):
        self._pairs = pairs
        self._resources = resources
        self._destinations = destinations
        self._pairIndictorFunction = pairIndictorFunction
        self._resourceIndicatorFunction = resourceIndicatorFunction
        self._x_values = list()
        self._resource_data =  [[] for _ in range(len(resources))]
        self._pair_data = [[] for _ in range(len(pairs))]
        self._destination_data = [[] for _ in range(len(destinations))]
        self._destination_tput_setpoint = [[] for _ in range(len(destinations))]
        self._resource_limits =  [[] for _ in range(len(resources))]
        self._pair_limits = [[] for _ in range(len(pairs))]
        self._destination_limits = [[] for _ in range(len(destinations))]

    def __init__(self, num_resources, num_pairs, externalTrafficProbability, pairIndictorFunction, resourceIndicatorFunction):

        # If network has external traffic, the destinations will have fluctuating optimal capacities 
        self._externalTrafficProbability = externalTrafficProbability

        # Initialize resources
        self._resources = []
        i = 0
        for _ in range(num_resources):
            tp_limit = random.randint(5,150)  # Adjust the range as needed
            tp_optimal = random.randint(5,150)
            resource = Resource(tp_limit, i, tp_optimal)
            self._resources.append(resource)
            i += 1
        
        self._resources.sort(key=lambda resource: resource.getLimit())
        self._destinations = dict()

        # Initialize pairs
        self._pairs = []
        for i in range(num_pairs):
            tp_limit =  100 # Adjust the range as needed, change this!!!! to be based on resource limits U_r / numPairsweighted 
            resource_indicies = random.sample(range(len(self._resources)), random.randint(1, len(self._resources))) # change this to just pick one
            resources = [self._resources[i] for i in resource_indicies]
            #pair = Pair(tp_limit, resources, random.random() * 10) #first resource is dest
        
            pair = Pair(i, tp_limit, resources, 1) # all pairs are equally weighted here 
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
        self._destination_tput_setpoint = [[] for _ in range(len(self._destinations))]
        self._resource_limits =  [[] for _ in range(len(self._resources))]
        self._pair_limits = [[] for _ in range(len(self._pairs))]
        self._destination_limits = [[] for _ in range(len(self._destinations))]
        self._destination_optimal = [[] for _ in range(len(self._destinations))]

        
        self._pairIndictorFunction = pairIndictorFunction
        self._resourceIndicatorFunction = resourceIndicatorFunction
    
    def init_tput(self):
        for dest in self._destinations:
            #tput = random.randint(1,2 * dest._optimalTput)
            tput = min(dest._tpUserLimit,  (5 * dest._optimalTput - 1))
            dest._currentTputSetpoint = tput
            dest._totUserWeights = 0 
            for pair in self._destinations[dest]:
                dest._totUserWeights += pair.getWeight()
            for pair in self._destinations[dest]:
                pair.updateTput(tput * pair.getWeight()/dest._totUserWeights)
            #for resource in pair._resources:
                #resource._tput += tput
            #pair._destination._currentTputSetpoint += tput
            #pair._destination._currentTput = pair._destination.getActualTput()
                
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
                CalculatedSetpoint += pair.getTput()
                CalculatedWeights += pair.getWeight()
            print("Dest", dest._hashVal)
            print("\t actual setpoint:", dest._currentTputSetpoint)
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
            self._pair_data[pair.getId()].append(pair.getTput())
            # self._pair_data[i].append(pair.getTput())
            self._pair_limits[i].append(pair.getLimit())
            i += 1

        i = 0
        for dest in self._destinations:
            self._destination_tput_setpoint[i].append(dest._prevTputSetpoint) # or current?
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
        print("\t current tput setpoint: ", dest._currentTputSetpoint)
        print("\t prev tput : ", dest._prevTput)
        print("\t prev tput setpoint: ", dest._prevTputSetpoint)
        print("\t gradient: ", gradient)
        print("\t opt tput: ", dest._optimalTput)
        print("\t limit:", dest.getLimit())
        print("\t decision: ", decision)
        print("\t new setpoint: ", newSetpoint)
        print("\t tot user weights: ", dest._totUserWeights)
        
    def printState(self):
        i = 0
        print("Resources\n")
        for resource in self._resources:
            print("Resource ", i, ": Throuput: ", resource._currentTput, " Setpoint: ", resource._currentTputSetpoint, "Optimal: ", resource._optimalTput, " Limit: ", resource.getLimit(), "\n")
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
        beta = 0.75
        bigBeta = 0.6 #for when user limit exceeded

        #self.updateResourceTput()
        for dest in self._destinations:

            # check if the destination limits should be changed to simulate external traffic 
            if np.random.rand() < self._externalTrafficProbability:
                # change the optimal tput to within 20% of the current optimal tput 
                dest._optimalTput = dest._optimalTput * np.random.uniform(0.7, 1.3) 
                print("Dest", dest._hashVal, "optimal tput changed to", dest._optimalTput)
        
            # if we do not have previous information about this resource 
            # set at the user specified limit 
            if dest._prevTputSetpoint == 0:
                tput = min(dest._tpUserLimit,  (5 * dest._optimalTput))
                dest._currentTputSetpoint = tput
                dest._currentTput = dest.getActualTput()
                dest._prevTputSetpoint = tput # this forces the gradient to be negative -- do we want this?
                dest._prevTput = dest._currentTput 
            
            # we have previous information to optimize on 
            else: 
                dest._currentTput = dest.getActualTput()
                deltaTput = dest._currentTput - dest._prevTput
                deltaSetpoint = dest._currentTputSetpoint - dest._prevTputSetpoint

                if(deltaSetpoint != 0):
                    gradient = (deltaTput)/(deltaSetpoint)
                else:
                    gradient = 1 #hack for zero gradient, idk what to do here for sure
            
                decision = ""
            
                #adjust setpoint
                if gradient >= 0 and dest._currentTputSetpoint < dest.getLimit():
                    # increase setpoint up to user specified limit 
                    newSetpoint = min(dest._tpUserLimit, (2 * dest._optimalTput - 1), dest._currentTputSetpoint + (alpha * dest._totUserWeights)) 
                    decision = "increase"
                elif dest._currentTput > dest.getLimit():
                    newSetpoint = dest._currentTputSetpoint * bigBeta 
                    decision = "decrease: user limit"
                else:
                    newSetpoint = dest._currentTputSetpoint * beta
                    decision = "decrease: opt"
                
                self.printDestState(dest, gradient, decision, newSetpoint)
                dest._prevTputSetpoint = dest._currentTputSetpoint
                dest._prevTput = dest._currentTput
            
                dest._currentTputSetpoint = newSetpoint

            # propogate new setpoint to pairs
            for pair in self._destinations[dest]:
                newTput = dest._currentTputSetpoint * (pair.getWeight()/dest._totUserWeights)
                pair.updateTput(newTput)


        self.saveState()
        #self.printState()

    def simulate(self, numIterations):
        self.clearState()
        self.init_tput()
        self.saveState()
        for i in range(numIterations):
            self.optimizerIteration()
        
        return self._x_values, self._pair_data, self._pair_limits, self._destination_data, \
                        self._destination_limits, self._destination_optimal, self._destination_tput_setpoint, \
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
    numPairs = 20 
    numIterations = 30 
    externalTrafficProbability = 0.2

    # initialize network simulation 
    myNetwork = Network(numResources, numPairs, externalTrafficProbability, myPairIndicatorFunction,myResourceIndicatorFunction)
    myNetwork.printNetworkState()
    data = myNetwork.simulate(numIterations)
    print(data)

    # save plots 
    timestamp_str = datetime.now().strftime("%m-%d_%H-%M-%S")
    folder_path = os.path.join('/Users/jennymao/Documents/repos/NetworkSimulation/', f"folder_{timestamp_str}/")

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
    


    


    
  



        





    