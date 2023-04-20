import json

def readCONST():
    json_file = open('Reinforcement_Learning/CONSTANTS.json')
    CONST = json.load(json_file)
    return CONST