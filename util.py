import json

t_all = 2

class Glb:
    def __init__(self):
        self.regX = None
        self.regY = None
        self.inferX = None
        self.inferY = None
        self.prdX = None
        self.prdY = None

theGlb = Glb()

def getGlb():
    return theGlb

def writeParams(params):
    with open('data/arm.json', 'w') as f:
        json.dump(params, f, indent=4)
