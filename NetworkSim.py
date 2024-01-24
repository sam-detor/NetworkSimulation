
import random
import string
import matplotlib.pyplot as plt
import math
from scipy.optimize import fsolve
import numpy as np
import os

class Pair:
    def __init__(self, tpLimit, resources, weight):
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

    def getDestination(self):
        return self._destination



class Resource:
    def __init__(self, tpLimit, id, optTput):
        self._tpUserLimit = tpLimit
        self._optimalTput = optTput
        self._currentTput = 0
        self._prevTput = 0
        self._currentTputSetpoint = 0
        self._prevTputSetpoint = 0
        self._totUserWeights = 0
        self._hashVal = id

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
        return self._optimalTput/ np.sqrt(setpoint - (1.5 * self._optimalTput))
    
    def getActualTput(self):
        return self.quadFunc(self._currentTputSetpoint)
    def leftEdgeFunc(self, setpoint):
        return self._optimalTput/ np.sqrt(-setpoint + (.5 * self._optimalTput))
    
    """  def getActualTput(self):
        if(self._currentTputSetpoint - (1.5 * self._optimalTput) > 0 and self.rightEdgeFunc(self._currentTputSetpoint) < (self._optimalTput * 0.8)):
            return self.rightEdgeFunc(self._currentTputSetpoint)
        elif (-self._currentTputSetpoint + (.5 * self._optimalTput) > 0 and self.leftEdgeFunc(self._currentTputSetpoint) < (self._optimalTput * 0.8)):
            return self.leftEdgeFunc(self._currentTputSetpoint)
        else:
            return self.quadFunc(self._currentTputSetpoint)
 """

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
        self._resource_limits =  [[] for _ in range(len(resources))]
        self._pair_limits = [[] for _ in range(len(pairs))]
        self._destination_limits = [[] for _ in range(len(destinations))]

    def __init__(self, num_resources, num_pairs, pairIndictorFunction, resourceIndicatorFunction):
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
        for _ in range(num_pairs):
            tp_limit =  random.random() * 100 # Adjust the range as needed
            resource_indicies = random.sample(range(len(self._resources)), random.randint(1, len(self._resources)))
            resources = [self._resources[i] for i in resource_indicies]
            pair = Pair(tp_limit, resources, random.random() * 100) #first resource is dest
            self._pairs.append(pair)
            
            #update destinations map
            if pair.getDestination() in self._destinations:
                self._destinations[pair.getDestination()].append(pair)
            else:
                self._destinations[pair.getDestination()] = [pair]

        self._x_values = list()
        self._resource_data =  [[] for _ in range(len(self._resources))]
        self._pair_data = [[] for _ in range(len(self._pairs))]
        self._destination_data = [[] for _ in range(len(self._destinations))]

        self._resource_limits =  [[] for _ in range(len(self._resources))]
        self._pair_limits = [[] for _ in range(len(self._pairs))]
        self._destination_limits = [[] for _ in range(len(self._destinations))]

        self._destination_optimal = [[] for _ in range(len(self._destinations))]

        
        self._pairIndictorFunction = pairIndictorFunction
        self._resourceIndicatorFunction = resourceIndicatorFunction
    
    def init_tput(self):
        for dest in self._destinations:
            tput = random.randint(1,2 * dest._optimalTput)
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
            self._pair_data[i].append(pair.getTput())
            self._pair_limits[i].append(pair.getLimit())
            i += 1

        i = 0
        for dest in self._destinations:
            self._destination_data[i].append(dest._currentTput)
            self._destination_limits[i].append(dest.getLimit())
            self._destination_optimal[i].append(dest._optimalTput)
            i += 1
    
    def clearState(self):
        self._x_values = list()
        self._resource_data =  [[] for _ in range(len(self._resources))]
        self._pair_data = [[] for _ in range(len(self._pairs))]
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

    def optimizerIteration(self):
        alpha = 2
        beta = 0.75
        bigBeta = 0.6 #for when user limit excedded

        #self.updateResourceTput()
        for dest in self._destinations:
            dest._currentTput = dest.getActualTput()

            deltaTput = dest._currentTput - dest._prevTput
            deltaSetpoint = dest._currentTputSetpoint - dest._prevTputSetpoint

            if(deltaSetpoint != 0):
                gradient = (dest._currentTput - dest._prevTput)/(dest._currentTputSetpoint - dest._prevTputSetpoint)
            else:
                gradient = 1 #hack for zero gradient, idk what to do here for sure
           
            decision = ""
            #adjust setpoint
            if gradient >= 0 and dest._currentTput < dest.getLimit():
                newSetpoint = dest._currentTputSetpoint + alpha
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
        
        return self._x_values, self._pair_data, self._pair_limits, self._destination_data, self._destination_limits, self._destination_optimal, self._resource_data, self._resource_limits
        




def myPairIndicatorFunction(pair, maxC, minCVal, gradient):
    beta = 0.9
    alpha = 1
    if gradient < 0:
        return minCVal * pair.getWeight() * beta
    else:
        return (maxC * pair.getWeight()/pair._destination._totUserWeights) + (alpha * pair.getWeight()/pair._destination._totUserWeights)
    
def myResourceIndicatorFunction(pair):
    #proposedDecreases = list()
    #for resource in pair._resources:
    if pair._resources[0]._currentTput > pair._resources[0].getLimit():
        #proposedDecreases.append(resource.getLimit() * pair.getWeight()/resource._totUserWeights)
        return pair._resources[0].getLimit() * pair.getWeight()/pair._resources[0]._totUserWeights
    else:
        return float('inf')
    #if len(proposedDecreases) == 0:
        #return float('inf')
    #return min(proposedDecreases) 


def main():
    myNetwork = Network(4,20,myPairIndicatorFunction,myResourceIndicatorFunction)
    myNetwork.printNetworkState()
    data = myNetwork.simulate(30)
    print(data)


    # Specify the path for the new folder
    folder_path = '/Users/samdetor/networkSim/' + str(random.randint(1,100)) + '/'

    # Create the new folder
    try:
        os.makedirs(folder_path)
        print(f"New folder '{folder_path}' created successfully.")
    except OSError as e:
        print(f"Error creating folder: {e}")

    # Plot 1
    #plt.subplot(3, 1, 1)  # 3 rows, 1 column, plot 1
    for i in range(len(data[1])):
        myLabel1 = "Pair" + str(i)+ "Throuput"
        myLabel2 = "Pair" + str(i) + "Limit"
        plt.plot(data[0], data[1][i], label=myLabel1)
        plt.plot(data[0], data[2][i], label=myLabel2)
    plt.title('Pairs')
    plt.legend()

    plt.savefig(folder_path + "Pairs.png")
    plt.clf()
    # Plot 3
    #plt.subplot(3, 1, 3)  # 3 rows, 1 column, plot 3
    for i in range(len(data[3])):
        plt.clf()
        myLabel1 = "Dest" + str(i) + "Throuput"
        myLabel2 = "Dest" + str(i) + "Limit"
        myLabel3 = "Dest" + str(i) + "Opt"
        plt.plot(data[0], data[3][i], label=myLabel1)
        plt.plot(data[0], data[4][i], label=myLabel2)
        plt.plot(data[0], data[5][i], label=myLabel3)
        plt.title("Destination" + str(i))
        plt.legend()
        plt.savefig(folder_path + "Destination" + str(i) + ".png")

    plt.clf()
    # Plot 2
    #plt.subplot(3, 1, 2)  # 3 rows, 1 column, plot 2
    print("Data 3:", data[6])
    print("Data 3 len:", data[6])
    for i in range(len(data[6])):
        myLabel1 = "Resource" + str(i) + "Throuput"
        myLabel2 = "Resource" + str(i) + "Limit"
        plt.plot(data[0], data[6][i], label=myLabel1)
        plt.plot(data[0], data[7][i], label=myLabel2)
    plt.title('Resources')
    plt.legend()
    plt.savefig(folder_path + "Resources.png")

    # Adjust layout for better spacing
    #plt.tight_layout()

    # Show the plots
   
    



if __name__ == "__main__":
    main()
    


    


    
  



        





    