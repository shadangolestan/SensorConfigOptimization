from ipywidgets import IntProgress
import sys
from scipy.stats import norm
from numpy import argmax
import SIM_SIS_Libraries.SensorsClass
import SIM_SIS_Libraries.SIM_SIS_Simulator as sim_sis
import SIM_SIS_Libraries.ParseFunctions as pf
import itertools
import numpy as np
import pandas as pd
import copy
from datetime import datetime
import pytz
import ast
import os
import random
import plotly
import CASAS.al as al
import pickle

class Chromosome:
    def __init__(self, 
                 grid = None, 
                 mode = None, 
                 space = None, 
                 initSensorNum = 10, 
                 epsilon = 0.5, 
                 new = True,
                 sensorTypesNum = 1,
                 greedy = True,
                 counter = 1,
                 chromosome_pointer = 0):
        
        self.sensitivity = {'3': 'pressure',
                            '4': 'accelerometer',
                            '5': 'electricity'}
        
        self.sensorTypesNum = sensorTypesNum
        self.radius = 1
        self.mode = mode
        self.epsilon = epsilon
        self.initSensorNum = initSensorNum
        self.space = space
        self.placeHolders = []
        self.fitness = -1
        self.SensorPlaceHolderSetup()
        self.chromosome_pointer = chromosome_pointer
        
        if greedy:
            if new:
                self.GreedySensorConfigurationSetup(counter)
            else:
                self.grid = grid
            
        else:
            if new:
                self.SensorConfigurationSetup()
            else:
                self.grid = grid
            
    '''
    def __init__(self, *args, new = False, sensorTypesNum = 1, greedy = False, counter = 1):
        if greedy:
            print(greedy)
            self.sensorTypesNum = sensorTypesNum
            self.radius = 1
            self.epsilon = args[3]
            self.initSensorNum = args[2]
            self.mode = args[0]
            self.space = args[1]
            self.placeHolders = []
            self.fitness = -1
            self.SensorPlaceHolderSetup()
            self.GreedySensorConfigurationSetup(counter)

        else:
            self.sensorTypesNum = sensorTypesNum
            self.radius = 1
            if new:
                print(greedy, ' new')
                self.epsilon = args[3]
                self.initSensorNum = args[2]
                self.mode = args[0]
                self.space = args[1]
                self.placeHolders = []
                self.fitness = -1
                self.SensorPlaceHolderSetup()
                self.SensorConfigurationSetup()

            elif new == False:
                print(greedy, ' old')
                self.epsilon = args[3]
                self.placeHolders = []
                self.fitness = -1
                self.grid = args[0]
                self.mode = args[1]
                self.space = args[2]
                self.SensorPlaceHolderSetup()
                
                print('Grid:', self.grid)
                print('mode:', self.mode)
                print('space:', self.space)
                print('epsilon:', self.epsilon)
                print('sensorTypesNum:', self.sensorTypesNum)
                print('radius:', self.radius)            
    '''
        
    def frange(self, start, stop, step):
        steps = []
        while start < stop:
            steps.append(start)
            start +=step
            
        return steps
    
    def GoToNeighbour(self):
        nonzeroind = list(np.nonzero(self.grid)[0])
        zeroind = list(np.where(np.array(self.grid) == 0)[0])
        
        sensor = random.choice(nonzeroind)
        emptyPlace = random.choice(zeroind)
        
        # TODO: for mutiple sensor types:
        self.grid[sensor] = 0
        self.grid[emptyPlace] = 1
        

    def SensorPlaceHolderSetup(self):   
        Xs = self.frange(self.epsilon, self.space[0], self.epsilon)
        Ys = self.frange(self.epsilon, self.space[1], self.epsilon)
        
        import matplotlib.pyplot as plt
        
        for x in Xs:
          for y in Ys:
            if (x < 2 and y < 2):
                continue
            else:
                self.placeHolders.append([x, y])


    def GreedySensorConfigurationSetup(self, counter):
        # Xs = self.frange(self.epsilon, self.space[0], self.epsilon)
        # Ys = self.frange(self.epsilon, self.space[1], self.epsilon)
        self.grid = np.zeros(len(self.placeHolders)).tolist()

        cell = counter
        self.grid[cell] = 1
            
    def SensorConfigurationSetup(self):
        # Xs = self.frange(self.epsilon, self.space[0], self.epsilon)
        # Ys = self.frange(self.epsilon, self.space[1], self.epsilon)
        self.grid = np.zeros(len(self.placeHolders)).tolist()
        
        i = 0
        while i < self.initSensorNum:
            cell = random.randrange(len(self.grid))
            if self.grid[cell] == 0:
                self.grid[cell] = 1
                i += 1
                
    def makeBoundaries(self, minValue, maxValue):
        m = self.epsilon * np.ceil(float(1/self.epsilon) * minValue)
        M = self.epsilon * np.floor(float(1/self.epsilon) * maxValue)
        return m, M
            
    def GetSensorConfiguration(self):
        from collections import Counter
        sensorLocations, sensorTypes = self.GetSensorLocations()
        _, rooms, _ = pf.ParseWorld(simworldname = '')
        
        summaryDict = Counter(sensorTypes)

        # TODO: DIFFERENT SENSOR TYPE DEFINITIONS SHOULD BE ADDED HERE:
        configurationSummary = []
        IS_handled = False
        
        for key in summaryDict.keys():
            if (key == 1):
                configurationSummary.append(['motion sensors', summaryDict[key]])

            elif (key == 2):
                configurationSummary.append(['beacon sensors', summaryDict[key]])
                
            elif (key >= 3):
                if IS_handled == False:
                    ISCount = sum([v for k,v in summaryDict.items() if k >= 3])
                    configurationSummary.append(['IS', ISCount])           
                    IS_handled = True
                
        configurationDetails = []
        
        for index, loc in enumerate(sensorLocations):
            room = ""
            for r in rooms:
                if (loc[0] >= rooms[r][0][0] and loc[0] <= rooms[r][1][0] and loc[1] >= rooms[r][0][1] and loc[1] <= rooms[r][1][1]):
                    room = r
                    break

            if (sensorTypes[index] == 1):
                configurationDetails.append(tuple([loc, room, 'motion sensors']))

            elif (sensorTypes[index] == 2):
                configurationDetails.append(tuple([loc, room, 'beacon sensors']))
                
            elif (sensorTypes[index] == 3):
                configurationDetails.append(tuple([loc, room, 'IS', self.sensitivity['3']]))
                
            elif (sensorTypes[index] == 4):
                configurationDetails.append(tuple([loc, room, 'IS', self.sensitivity['4']]))
                
            elif (sensorTypes[index] == 5):
                configurationDetails.append(tuple([loc, room, 'IS', self.sensitivity['5']]))

            else:
                configurationDetails.append(tuple([loc, room, 'motion sensors']))
        
        sensor_config = [[configurationSummary, [tuple(configurationDetails)]], self.radius]
        
        # print(sensor_config)
        return sensor_config


    def GetSensorLocations(self):
        sensorLocations = []
        sensorTypes = []
        for index, sensorIndicator in enumerate(self.grid):
            if (sensorIndicator > 0):
                sensorLocations.append(self.placeHolders[index])
                sensorTypes.append(sensorIndicator)

        return sensorLocations, sensorTypes
    
class GreedyAndLocalSearch:
    chromosomes = []
    def __init__(self, testbed, initializationMethod, path, epsilon, initSensorNum, maxSensorNum, radius, ROS, learning_rate, sensorTypesNum):
        self.results = []
        self.best_configuration_history = []
        self.data_path = path + testbed
        self.base_path = path
        self.mode = initializationMethod
        self.epsilon = epsilon
        self.initSensorNum = initSensorNum
        self.maxSensorNum = maxSensorNum
        self.radius = radius
        self.learning_rate = learning_rate
        self.sensorTypesNum = sensorTypesNum
        
        self.sensor_distribution, self.types, self.space, self.rooms, self.agentTraces = self.ModelsInitializations(ROS)

        # TODO: THIS FORMULA NEEDS TO BE GENERALIZED:
        self.population = int(int(((self.space[0] - 2*epsilon) / epsilon) + 2) * int(((self.space[1] - 2*epsilon) / epsilon) + 1))

        W = []
        H = []
        grid = []
        start = self.epsilon

        while start < self.space[0]:
            W.append(start)
            start += self.epsilon

        start = self.epsilon
        while start < self.space[1]:
            H.append(start)
            start += self.epsilon

        for w in W:
            for h in H:
                if w < 2 and h < 2:
                    continue
                else:
                    grid.append([w, h])

        self.population = len(grid)
        # self.population = int(int((self.space[1] / epsilon) - epsilon) * int((self.space[0]) / epsilon) - epsilon)

        for i in range(self.population):
            self.chromosomes.append(Chromosome(grid = None,
                                               mode = self.mode,
                                               space = self.space,
                                               initSensorNum = self.initSensorNum,
                                               epsilon = self.epsilon,
                                               new = True,
                                               sensorTypesNum = self.sensorTypesNum,
                                               greedy = True,
                                               counter = i,
                                               chromosome_pointer = i))

    def sortChromosomes(self, chroms):
        chroms.sort(key=lambda x: x.fitness, reverse=True)
   
    def RunGreedyAlgorithm(self):
        picked_sensors = 1
        self.RunFitnessFunction(self.chromosomes, True, False, False, False, 1)        
        self.sortChromosomes(self.chromosomes)
        
        self.results.append([(c.fitness, sum(c.grid)) for c in self.chromosomes])
        self.best_configuration_history.append(self.chromosomes[0])
        
        grid = self.chromosomes[0].grid
        self.current_configs = [Chromosome(grid = grid,
                                       mode = self.chromosomes[0].mode,
                                       space = self.chromosomes[0].space,
                                       initSensorNum = self.initSensorNum,
                                       epsilon = self.epsilon,
                                       new = False,
                                       sensorTypesNum = self.sensorTypesNum,
                                       greedy = True,
                                       counter = 1)]
        
        
        while picked_sensors < self.maxSensorNum:   
            self.test_configs = []

            for i in range(1, len(self.chromosomes)):
                new_grid = []
                new_grid = [int(x + y) for x, y in zip(grid, self.chromosomes[i].grid)]

                self.test_configs.append(Chromosome(grid = new_grid,
                                        mode = self.chromosomes[0].mode,
                                        space = self.chromosomes[0].space,
                                        initSensorNum = self.initSensorNum,
                                        epsilon = self.epsilon,
                                        new = False,
                                        sensorTypesNum = self.sensorTypesNum,
                                        greedy = True,
                                        counter = 1,
                                        chromosome_pointer = self.chromosomes[i].chromosome_pointer))

            
            self.RunFitnessFunction(self.test_configs, True, False, False, False, 1)            
            self.sortChromosomes(self.test_configs)
            
            self.results.append([(c.fitness, sum(c.grid)) for c in self.test_configs])
            self.best_configuration_history.append(self.test_configs[0])
            grid = self.test_configs[0].grid
            self.chromosomes = list(filter(lambda x: x.chromosome_pointer != self.test_configs[0].chromosome_pointer, self.chromosomes)) 
            picked_sensors = picked_sensors + 1

                             
        self.GreedyOutput = [Chromosome(grid = copy.deepcopy(grid),
                                        mode = self.chromosomes[0].mode,
                                        space = self.chromosomes[0].space,
                                        initSensorNum = self.initSensorNum,
                                        epsilon = self.epsilon,
                                        new = False,
                                        sensorTypesNum = self.sensorTypesNum,
                                        greedy = True,
                                        counter = 1)]
        
        print('Greedy Performance is (F1-score, sensors placed): ', self.results[-1][0])
        
        
    def GetNextGeneration(self, epoch):
        import copy 
        
        self.newConfigs = []
        self.newConfigs = copy.deepcopy(self.GreedyOutput)
        # self.newConfigs.append(copy.copy(self.chromosomes[0]))
        
        for config in self.newConfigs:
            config.GoToNeighbour()
        
        # self.RunFitnessFunction(self.GreedyOutput, True, False, False, False, 1)
        self.RunFitnessFunction(self.newConfigs, True, False, False, False, 1)
        
        self.results.append([(c.fitness, sum(c.grid)) for c in self.newConfigs])
        
        if self.newConfigs[0].fitness > self.GreedyOutput[0].fitness:
            self.GreedyOutput = copy.deepcopy(self.newConfigs)
            self.best_configuration_history.append(self.newConfigs[0])
            
        else:
            prob = random.uniform(0, 1)
            if prob <= self.learning_rate:
                self.GreedyOutput = copy.deepcopy(self.newConfigs)
                self.best_configuration_history.append(self.newConfigs[0])
                
            else:
                self.best_configuration_history.append(self.GreedyOutput)
                
        self.learning_rate = self.learning_rate / np.sqrt(epoch + 1)

    def ModelsInitializations(self, ROS):
        #----- Space and agent models -----: 
        simworldname = self.base_path + '/Configuration Files/simulationWorld2.xml'
        agentTraces = []
        agent_trace_path_ROS = 'Agent Trace Files ROS/'
        agent_trace_path = 'Agent Trace Files/'
        import os
        if ROS:
            directory = os.fsencode(self.data_path + agent_trace_path_ROS)
        else:
            directory = os.fsencode(self.data_path +agent_trace_path)
            
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".csv"): 
                if ROS:
                    agentTraces.append(self.data_path + agent_trace_path_ROS + filename)
                else:
                    agentTraces.append(self.data_path + agent_trace_path + filename)

        # Parsing the space model: 
        space, rooms, _ = pf.ParseWorld(simworldname)
        sim_sis.AddRandomnessToDatasets(self.epsilon, self.data_path, rooms)
        space = [space[-1][0], space[1][1]]

        # User parameters 
        types, sensor_distribution = pf.GetUsersParameters()

        roomsList = []
        for room in sensor_distribution:
            roomsList.append(room)
              
        return sensor_distribution, types, space, rooms, agentTraces

    def RunFitnessFunction(self, chroms, simulateMotionSensors, simulateEstimotes, simulateIS, Plotting, iteration):
        for index, chromosome in enumerate(chroms):
            files = []
            
            all_sensors = set([])

            for agentTrace in self.agentTraces:
                filecoding = ' '#"_" + str(iteration) + "_c" + str(index + 1) + '(' + self.mode + ')'
                df_ = sim_sis.RunSimulator(self.space, 
                                           self.rooms, 
                                           agentTrace, 
                                           chromosome.GetSensorConfiguration(), 
                                           simulateMotionSensors, 
                                           simulateEstimotes, 
                                           simulateIS, 
                                           Plotting , 
                                           self.base_path)

                dataFile, sensors = self.PreProcessor(df_)
                all_sensors.update(sensors)
                #self.D = dataFile
                files.append(dataFile)    

            all_sensors = list(all_sensors)
            
            fitness = al.leave_one_out(files, all_sensors)
            CONSTANTS['iterations'] = CONSTANTS['iterations'] - 1
            
            try:
                if len(fitness[0]) > 1:
                    fitness = fitness[0]
            except:
                pass
            
                    
            chromosome.fitness = fitness[0] * 100
            
            if chromosome.fitness < 0:
                chromosome.fitness = 0
                
                

    def frange(self, start, stop, step):
        steps = []
        while start < stop:
            steps.append(start)
            start +=step
            
        return steps

    def MakeSensorCombinations(self, start, end, epsilon, sensorType, room):
        a1, b1 = self.makeBoundaries(epsilon, start[0], end[0])
        a2, b2 = self.makeBoundaries(epsilon, start[1], end[1])    
        Xs = self.frange(a1, b1, epsilon)
        Ys = self.frange(a2, b2, epsilon)
        
        points = list(itertools.product(list(itertools.product(Xs, Ys)), [room], [sensorType[0]])) 
        C = itertools.combinations(points, distribution[room][types.index(sensorType)])

        return C
    
    
    def PreProcessor(self, df):
        df['motion sensors'] = df['motion sensors'].apply(lambda s: list(map(int, s)))
        try:
            df['beacon sensors'] = df['beacon sensors'].apply(lambda s: list(map(int, s)))
        except:
            pass
        try:
            df['IS'] = df['IS'].apply(lambda s: list(map(int, s)))
        except:
            pass

        pre_activity = ''
        save_index = 0

        for index, row in df.iterrows():
            save_index = index
            Activity = row['activity']

            if Activity != pre_activity:
                if pre_activity != '':
                    df.at[index - 1, 'motion sensors'] += [0]

                else:
                    df.at[index, 'motion sensors'] += [1]

                pre_activity = Activity
            else:
                df.at[index - 1, 'motion sensors'] += [1]

        df.at[save_index, 'motion sensors'] += [0]

        sensors = set([])

        previous_M = None
        previous_B = None
        previous_I = None

        output_file = []
        
        for index, row in df.iterrows():
          T = row['time']
          M = row['motion sensors']
          try:
            B = row['beacon sensors']
          except:
            pass

          try:
            I = row['IS']
          except:
            pass

          Activity = row['activity']
          Activity = Activity.replace(' ', '_')
          MotionSensor_Names = []
          sensorNames = []
          MotionSensor_Message = []
          BeaconSensor_Names = []
          BeaconSensor_Message = []
          ISSensor_Names = []
          ISSensor_Message = []


          # time = convertTime(T)
          time = "2020-06-16 " + T + ".00"



          # Motion Sensor
          #print(len(M))
          for i in range(len(M)):
                sensorNames.append(self.Name(i, 'M'))
                if M[i] == 1:
                      if (previous_M != None):
                        if (previous_M[i] == 0):
                          MotionSensor_Names.append(self.Name(i,'M'))
                          MotionSensor_Message.append('ON')

                      else:
                        MotionSensor_Names.append(self.Name(i,'M'))
                        MotionSensor_Message.append('ON')

                if previous_M != None:
                      if M[i] == 0 and previous_M[i] == 1:
                        MotionSensor_Names.append(self.Name(i,'M'))
                        MotionSensor_Message.append('OFF')

          previous_M = M



          try:
              for i in range(len(I)):
                sensorNames.append(self.Name(i, 'IS'))
                if I[i] == 1:
                      if (previous_I != None):
                        if (previous_I[i] == 0):
                          ISSensor_Names.append(self.Name(i,'IS'))
                          ISSensor_Message.append('ON')

                      else:
                        ISSensor_Names.append(self.Name(i,'IS'))
                        ISSensor_Message.append('ON')

                if previous_I != None:
                      if I[i] == 0 and previous_I[i] == 1:
                        ISSensor_Names.append(self.Name(i,'IS'))
                        ISSensor_Message.append('OFF')

              previous_I = I

          except:
              pass 

          for m in range(len(MotionSensor_Names)):
            output_file.append(time +' '+ MotionSensor_Names[m] + ' ' + MotionSensor_Names[m] + ' ' + MotionSensor_Message[m] + ' ' + Activity)

          for i_s in range(len(ISSensor_Names)):
            output_file.append(time +' '+ ISSensor_Names[i_s] + ' ' + ISSensor_Names[i_s] + ' ' + ISSensor_Message[i_s] + ' ' + Activity)

          for s in sensorNames:
              sensors.add(s)
                
        return output_file, list(sensors)
    

    def frange(self, start, stop, step):
        steps = []
        while start < stop:
            steps.append(start)
            start +=step

        return steps
    
    def BestAnswer(self):
        bestAnswerIndex = 0
        for index, c in enumerate(self.chromosomes):
            if c.fitness > self.chromosomes[bestAnswerIndex].fitness:
                bestAnswerIndex = index

        return self.chromosomes[bestAnswerIndex]

    def AverageAnswer(self):
        return np.sum([c.fitness for c in self.chromosomes]) / len(self.chromosomes)

    def Name(self, number, typeSensor):
        if number < 10:
          return typeSensor + str(0) + str(number)
        else:
          return typeSensor + str(number)

    #converts epoch time to human readable
    def convertTime(self, posix_timestamp):
        tz = pytz.timezone('MST')
        dt = datetime.fromtimestamp(posix_timestamp, tz)
        time = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return time

    
def MakeDataBoundaries(height = 10.5, width = 6.6, MaxLSSensors = 15):
    from collections import defaultdict, OrderedDict
    d = dict()

    for idx in range(MaxLSSensors):
            d['x' + str(idx)] = (0.5, width - 0.5)
            d['y' + str(idx)] = (0.5, height - 0.5)

    return d
    
def run(testbed = 'Testbed1/',
        run_on_colab = False, 
        iteration = 1000, 
        population = 1,
        epsilon = 1, # The distance between two nodes in the space grid:
        LSmaxSensorNum = 15,  # max location sensitive sensor numbers
        ISmaxSensorNum = 10,  # max location sensitive sensor numbers
        radius = 1, # radius of the motion sensors
        print_epochs = True,
        height = 8.0,
        width = 8.0,
        ROS = False,
        multi_objective = False,
        initial_state = 'fixed',
        input_sensor_types = {'model_motion_sensor': True, 
                              'model_beacon_sensor': False,
                              'model_pressure_sensor': False,
                              'model_accelerometer': False,
                              'model_electricity_sensor': False},
        learning_rate = 1
      ):

    global multi_objective_flag
    global CONSTANTS
    global sensor_types
    global LSsensorTypesNum
    global ISsensorTypesNum

    multi_objective_flag = multi_objective
    CONSTANTS = {
        'iterations': iteration,
        'initial_samples': 10,
        'epsilon': epsilon,
        'radius': radius,
        'height': height,
        'width': width,
        'max_LS_sensors': LSmaxSensorNum,
        'max_IS_sensors': ISmaxSensorNum
    }
    
    sensor_types = input_sensor_types
    LSsensorTypesNum = sum(1 for condition in list(input_sensor_types.values())[0:2] if condition)
    ISsensorTypesNum = sum(1 for condition in list(input_sensor_types.values())[2:5] if condition)

    base_path = '../SensorConfigurationOptimization/'
    sys.path.append('..')

    finalResults = []
    w = CONSTANTS['width'] - 0.5
    h = CONSTANTS['height'] - 0.5

    results = []
    best_configuration_history = []
    
    print('----- Running SimulatedAnnealing for epsilon = ' + str(epsilon))
    # f = IntProgress(min=0, max=iteration) # instantiate the bar
    # display(f) # display the bar
    

    GLS = GreedyAndLocalSearch(testbed, 
                               'expert', 
                               base_path, 
                               epsilon, 
                               CONSTANTS['max_LS_sensors'], 
                               CONSTANTS['max_LS_sensors'], 
                               radius, 
                               ROS,
                               learning_rate,
                               LSsensorTypesNum + ISsensorTypesNum)
    
    print('Running Greedy Algorithm...', end='')
    GLS.RunGreedyAlgorithm()
    print('[Done!]')
    
    GLS.RunFitnessFunction(GLS.GreedyOutput, True, False, False, False, 1)
    print('number of sensors:', sum(GLS.GreedyOutput[0].grid))
    print('remaining queries:', CONSTANTS['iterations'])
    
    return GLS.results, GLS.best_configuration_history