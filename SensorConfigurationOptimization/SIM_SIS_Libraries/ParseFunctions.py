import SIM_SIS_Libraries.SensorsClass as sc
import Config as cf

#----------------- First Testbed
'''
space = [(0.0, 0.0), (0.0, 8.0), (8.0, 8.0), (8.0, 0.0)]
rooms = {'bedroom': [(3.9, 0.0), (8.0, 4.4)],
         'livingroom': [(0.0, 1.9), (6.3, 6.7)],
         'diningroom': [(0.0, 3.0), (2.9, 8.0)],
         'kitchen': [(0.0, 3.0), (2.9, 8.0)],
         'bathroom': [(6.1, 3.2), (8.0, 6.7)],
		 'storage': [(2.8, 6.4), (8.0, 8.0)]} 


objects = ['0.5, 2.7', '3.5, 2.7', '6.7, 1.4', '4.2, 3.2', '1.7, 6.0', '6.0, 3.6', '7.4, 3.6', '1.0, 5.5', '6.8, 5.5', '0.5, 7.1', '2.2, 7.1', '7.1, 6.8']
'''


#----------------- Second Testbed
'''
space = [(0.0, 0.0), (0.0, 8.0), (5.3, 8.0), (5.3, 0.0)]
rooms = {'bedroom': [(0.0, 1.9), (3.0, 4.7)],
         'livingroom': [(1.6, 0.0), (5.3, 3.3)],
         'diningroom': [(3.1, 2.0), (5.3, 3.3)],
         'kitchen': [(3.0, 3.3), (5.3, 6.0)],
         'bathroom': [(0.0, 4.9), (2.4, 6.9)],
		 'entryway': [(2.4, 6.0), (5.3, 8.0)]}

objects = ['0.5, 2.7', '3.5, 2.7', '6.7, 1.4', '4.2, 3.2', '1.7, 6.0', '6.0, 3.6', '7.4, 3.6', '1.0, 5.5', '6.8, 5.5', '0.5, 7.1', '2.2, 7.1', '7.1, 6.8']
'''


def GetUsersParameters():
    '''Gets and parses a file of user defined parameters. The output is (list of [type, number])'''
    #TODO: Real implementation!
    types = [['motion sensors', 14], ['beacon sensors', 14]]
    distribution = {'bedroom': [3, 3],
         'livingroom': [3, 3],
         'diningroom': [2, 2],
         'kitchen': [4, 4],
         'bathroom': [2, 2]
    }
    
    return types, distribution

def ParseWorld(simworldname):
    '''Gets a sensor configuration space in XML and parses it. The output is (space dimensions, rooms dimensions)'''
    #TODO: Space and Rooms needs to be read from a file
        
    return cf.space, cf.rooms, cf.objects