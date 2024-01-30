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
    
    def getActualTput(self):
        return self.quadFunc(self._currentSlots * self.tputPerSlot)