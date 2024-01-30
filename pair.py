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
        self._proposedDecisions = list()
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