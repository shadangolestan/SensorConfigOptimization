# GA ALGORITHM

from ipywidgets import IntProgress
from IPython.display import display
from SIM_SIS_Libraries import SensorsClass
from SIM_SIS_Libraries import SIM_SIS_Simulator
from SIM_SIS_Libraries import ParseFunctions as pf
import itertools
import numpy as np
import pandas as pd
import copy
from datetime import datetime
import pytz
import ast
import os
import random

class Chromosome:
    def __init__(self, testbed, *args):
        
        self.radius = 1
        self.testbed = testbed

        if len(args) < 5:
            self.epsilon = args[3]
            self.initSensorNum = args[2]
            self.mode = args[0]
            self.space = args[1]
            self.placeHolders = []
            self.fitness = -1
            self.SensorPlaceHolderSetup()
            self.SensorConfigurationSetup()
              
        elif len(args) == 5:
            self.epsilon = args[3]
            self.placeHolders = []
            self.fitness = -1
            self.grid = args[0]
            self.mode = args[1]
            self.space = args[2]
            self.SensorPlaceHolderSetup()
       
    def frange(self, start, stop, step):
        steps = []
        while start < stop:
            steps.append(start)
            start +=step
            
        return steps

    def SensorPlaceHolderSetup(self):  

        # print(self.space)

        Xs = self.frange(self.epsilon, self.space[0], self.epsilon)
        Ys = self.frange(self.epsilon, self.space[1], self.epsilon)
        

        import matplotlib.pyplot as plt
        
        for x in Xs:
          for y in Ys:
            if self.testbed == 'Testbed2/':
              if not (x <= 2 and y <= 2):
                self.placeHolders.append([x, y])
                
            else:
                self.placeHolders.append([x, y])

            

    def SensorConfigurationSetup(self):
        Xs = self.frange(self.epsilon, self.space[0], self.epsilon)
        Ys = self.frange(self.epsilon, self.space[1], self.epsilon)
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

        summaryDict = Counter(sensorTypes)

        # TODO: DIFFERENT SENSOR TYPE DEFINITIONS SHOULD BE ADDED HERE:
        configurationSummary = []
        for key in summaryDict:
            if (key == 1):
                configurationSummary.append(['motion sensors', summaryDict[key]])

            elif (key == 2):
                configurationSummary.append(['beacon sensors', summaryDict[key]])

        configurationDetails = []

        for index, loc in enumerate(sensorLocations):
            if (sensorTypes[index] == 1):
                configurationDetails.append(tuple([loc, 'kitchen', 'motion sensors']))

            elif (sensorTypes[index] == 2):
                configurationDetails.append(tuple([loc, 'kitchen', 'beacon sensors']))

        return [[configurationSummary, [tuple(configurationDetails)]], self.radius]


    def GetSensorLocations(self):
        sensorLocations = []
        sensorTypes = []
        for index, sensorIndicator in enumerate(self.grid):
            if (sensorIndicator > 0):
                sensorLocations.append(self.placeHolders[index])
                sensorTypes.append(sensorIndicator)

        return sensorLocations, sensorTypes


class GA:
    chromosomes = []
    def __init__(self, testbed, population, initializationMethod, path, epsilon, initSensorNum, maxSensorNum, radius, mutation_rate, crossover, survival_rate, reproduction_rate, ROS):
        self.population = population
        self.mode = initializationMethod
        self.data_path = path + testbed
        self.base_path = path
        self.epsilon = epsilon
        self.initSensorNum = initSensorNum
        self.maxSensorNum = maxSensorNum
        self.radius = radius
        self.mutation_rate = mutation_rate
        self.crossover = crossover
        self.survival_rate = survival_rate
        self.reproduction_rate = reproduction_rate
        
        self.sensor_distribution, self.types, self.space, self.rooms, self.agentTraces = self.ModelsInitializations(ROS)
        
        

        for i in range(population):
            self.chromosomes.append(Chromosome(testbed, self.mode, self.space, self.initSensorNum, self.epsilon))

    def Mutation(self, chromosome):
        for i in range(len(chromosome.grid)):
          if (random.random() < self.mutation_rate):
              if (chromosome.grid[i] == 0):
                  chromosome.grid[i] = 1
              else:
                  chromosome.grid[i] = 0

        return chromosome            

    def GetNextGeneration(self):
        import copy
        self.newGeneration = []

        last_one = True
        if int(np.ceil( (1 - self.survival_rate) *  (self.population / 2))) % 2 == 0:
            last_one = False

        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)

        for i in range(int(np.floor((1 - self.survival_rate) *  (self.population / 2)))):
            valid_child = False
            while not valid_child:
                coin1 = random.randrange(0, len(self.chromosomes) * self.reproduction_rate)
                coin2 = random.randrange(0, len(self.chromosomes) * self.reproduction_rate)

                p1 = copy.deepcopy(self.chromosomes[coin1])
                p2 = copy.deepcopy(self.chromosomes[coin2])

                p1.grid, p2.grid = self.Crossover(p1.grid, p2.grid)

                child1 = Chromosome(p1.testbed, p1.grid, p1.mode, p1.space, self.epsilon, None)
                child2 = Chromosome(p1.testbed, p2.grid, p2.mode, p2.space, self.epsilon, None)

                if sum(child1.grid) <= self.maxSensorNum or sum(child2.grid) <= self.maxSensorNum:
                    valid_child = True

            self.newGeneration.append(self.Mutation(child1))
            self.newGeneration.append(self.Mutation(child2))

        if last_one == True:
            self.newGeneration.append(self.chromosomes[int(np.floor(self.population / 2))])

        self.chromosomes = self.chromosomes[0: int(self.survival_rate * len(self.chromosomes))]

        for ng in self.newGeneration:
            self.chromosomes.append(ng)

    def Selection(self):
        self.chromosomes.sort(key=lambda x: x.fitness, reverse=True)
        self.chromosomes = self.chromosomes[0:self.population]

    def Crossover(self, l, q):
        l = list(l)
        q = list(q)
              
        f1 = random.randint(0, len(l)-1)
        f2 = random.randint(0, len(l)-1)
        while f1 == f2:
            f1 = random.randint(0, len(l)-1)
            f2 = random.randint(0, len(l)-1)

        if f1 > f2:
            tmp = f1
            f1 = f2
            f2 = tmp

        
        # interchanging the genes
        for i in range(f1, f2):
            l[i], q[i] = q[i], l[i]
        
        return l, q

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
            directory = os.fsencode(self.data_path + agent_trace_path)
            
        # Parsing the space model: 
        space, rooms, _ = pf.ParseWorld(simworldname)
        SIM_SIS_Simulator.AddRandomnessToDatasets(self.epsilon, self.data_path, rooms)

        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".csv"): 
                if ROS:
                    agentTraces.append(self.data_path + agent_trace_path_ROS + filename)
                else:
                    agentTraces.append(self.data_path + agent_trace_path + filename)

        space = [space[-1][0], space[1][1]]


        # User parameters 
        types, sensor_distribution = pf.GetUsersParameters()

        roomsList = []
        for room in sensor_distribution:
            roomsList.append(room)
              
        return sensor_distribution, types, space, rooms, agentTraces

    def calculate_confusion_matrix(self, config, simulateMotionSensors, simulateEstimotes, SimulateIS, Plotting, iteration):       
        
        files = []
        all_sensors = set([])

        for agentTrace in self.agentTraces:
            filecoding = ' '#"_" + str(iteration) + "_c" + str(index + 1) + '(' + self.mode + ')'
            df_ = SIM_SIS_Simulator.RunSimulator(self.space, 
                                                 self.rooms, 
                                                 agentTrace, 
                                                 config, 
                                                 simulateMotionSensors, 
                                                 simulateEstimotes, 
                                                 Plotting,
                                                 SimulateIS, 
                                                 self.base_path)


            dataFile, sensors = self.PreProcessor(df_)
            all_sensors.update(sensors)
            #self.D = dataFile
            files.append(dataFile)

        import CASAS.al as al
        import imp
        imp.reload(al)
        all_sensors = list(all_sensors)

        return al.get_confusion_matrix(files, all_sensors)

    
    def RunFitnessFunction(self, simulateMotionSensors, simulateEstimotes, SimulateIS, Plotting, iteration):       
        for index, chromosome in enumerate(self.chromosomes):
            if (chromosome.fitness == -1):
                files = []

                all_sensors = set([])

                for agentTrace in self.agentTraces:
                    filecoding = ' '#"_" + str(iteration) + "_c" + str(index + 1) + '(' + self.mode + ')'

                    df_ = SIM_SIS_Simulator.RunSimulator(self.space, 
                                                 self.rooms, 
                                                 agentTrace, 
                                                 chromosome.GetSensorConfiguration(), 
                                                 simulateMotionSensors, 
                                                 simulateEstimotes, 
                                                 Plotting,
                                                 SimulateIS, 
                                                 self.base_path)

                    dataFile, sensors = self.PreProcessor(df_)
                    all_sensors.update(sensors)
                    #self.D = dataFile
                    files.append(dataFile)
                    

                import CASAS.al as al
                import imp
                imp.reload(al)
                all_sensors = list(all_sensors)

                chromosome.fitness = (al.leave_one_out(files, all_sensors)[0] - (sum(chromosome.grid) / 100)) * 100 # - (len(chromosome.placeHolders)**(np.sum(chromosome.grid)/len(chromosome.placeHolders))) / len(chromosome.placeHolders)) * 100
                if chromosome.fitness < 0:
                    chromosome.fitness = 0

    def frange(self, start, stop, step):
        steps = []
        while start <= stop:
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
            # df['motion sensors'] = df['motion sensors'].apply(ast.literal_eval)
            df['motion sensors'] = df['motion sensors'].apply(lambda s: list(map(int, s)))
            #df['motion sensors'] = df['motion sensors'].apply(lambda s: s + [1])
            # df['beacon sensors'] = df['beacon sensors'].apply(ast.literal_eval)
            try:
              df['beacon sensors'] = df['beacon sensors'].apply(lambda s: list(map(int, s)))
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
            output_file = []

            for index, row in df.iterrows():
              T = row['time']
              M = row['motion sensors']
              try:
                B = row['beacon sensors']
              except:
                pass

              Activity = row['activity']
              Activity = Activity.replace(' ', '_')
              MotionSensor_Names = []
              sensorNames = []
              MotionSensor_Message = []
              BeaconSensor_Names = []
              BeaconSensor_Message = []
      

              # time = convertTime(T)
              time = "2020-06-16 " + T + ".00"

              # Motion Sensor
              for i in range(len(M)):
                sensorNames.append(Name(i, 'M'))
                if M[i] == 1:
                  if (previous_M != None):
                    if (previous_M[i] == 0):
                      MotionSensor_Names.append(Name(i,'M'))
                      MotionSensor_Message.append('ON')

                  else:
                    MotionSensor_Names.append(Name(i,'M'))
                    MotionSensor_Message.append('ON')

                if previous_M != None:
                  if M[i] == 0 and previous_M[i] == 1:
                    MotionSensor_Names.append(Name(i,'M'))
                    MotionSensor_Message.append('OFF')

              previous_M = M
              # Beacon Sensor

              try:
                for i in range(len(B)):
                  sensorNames.append(Name(i, 'B'))
                  if B[i] == 1:
                    BeaconSensor_Names.append(Name(i,'B'))
                    BeaconSensor_Message.append('ON')
                  if previous_B != None:
                    if B[i] == 0 and previous_B[i] == 1: 
                      BeaconSensor_Names.append(Name(i,'B'))
                      BeaconSensor_Message.append('OFF')
                previous_B = B

              except:
                pass

              for m in range(len(MotionSensor_Names)):
                output_file.append(time +' '+ MotionSensor_Names[m] + ' ' + MotionSensor_Names[m] + ' ' + MotionSensor_Message[m] + ' ' + Activity)
        
              for s in sensorNames:
                  sensors.add(s)

            return output_file, list(sensors)

    #returns the name of the sensor
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

    def BestAnswer(self):
        bestAnswerIndex = 0
        for index, c in enumerate(self.chromosomes):
            if c.fitness > self.chromosomes[bestAnswerIndex].fitness:
                bestAnswerIndex = index

        return self.chromosomes[bestAnswerIndex]

    def AverageAnswer(self):
        return np.sum([c.fitness for c in self.chromosomes]) / len(self.chromosomes)

    
def PreProcessor(df):
        # df['motion sensors'] = df['motion sensors'].apply(ast.literal_eval)
        df['motion sensors'] = df['motion sensors'].apply(lambda s: list(map(int, s)))
        # df['beacon sensors'] = df['beacon sensors'].apply(ast.literal_eval)
        try:
          df['beacon sensors'] = df['beacon sensors'].apply(lambda s: list(map(int, s)))
        except:
          pass

        sensors = set([])

        previous_M = None
        previous_B = None
        output_file = []

        for index, row in df.iterrows():
          T = row['time']
          M = row['motion sensors']
          try:
            B = row['beacon sensors']
          except:
            pass

          Activity = row['activity']
          Activity = Activity.replace(' ', '_')
          MotionSensor_Names = []
          sensorNames = []
          MotionSensor_Message = []
          BeaconSensor_Names = []
          BeaconSensor_Message = []
          

          # time = convertTime(T)
          time = "2020-06-16 " + T + ".00"

          # Motion Sensor
          for i in range(len(M)):
            sensorNames.append(Name(i, 'M'))
            if M[i] == 1:
              if (previous_M != None):
                if (previous_M[i] == 0):
                  MotionSensor_Names.append(Name(i,'M'))
                  MotionSensor_Message.append('ON')

              else:
                MotionSensor_Names.append(Name(i,'M'))
                MotionSensor_Message.append('ON')

            if previous_M != None:
              if M[i] == 0 and previous_M[i] == 1:
                MotionSensor_Names.append(Name(i,'M'))
                MotionSensor_Message.append('OFF')

          previous_M = M
          # Beacon Sensor

          try:
            for i in range(len(B)):
              sensorNames.append(Name(i, 'B'))
              if B[i] == 1:
                BeaconSensor_Names.append(Name(i,'B'))
                BeaconSensor_Message.append('ON')
              if previous_B != None:
                if B[i] == 0 and previous_B[i] == 1: 
                  BeaconSensor_Names.append(Name(i,'B'))
                  BeaconSensor_Message.append('OFF')
            previous_B = B

          except:
            pass

          for m in range(len(MotionSensor_Names)):
            output_file.append(time +' '+ MotionSensor_Names[m] + ' ' + MotionSensor_Names[m] + ' ' + MotionSensor_Message[m] + ' ' + Activity)
            
          for s in sensorNames:
              sensors.add(s)

        return output_file, list(sensors)

#returns the name of the sensor
def Name(number, typeSensor):
    if number < 10:
      return typeSensor + str(0) + str(number)
    else:
      return typeSensor + str(number)

#converts epoch time to human readable
def convertTime(posix_timestamp):
    tz = pytz.timezone('MST')
    dt = datetime.fromtimestamp(posix_timestamp, tz)
    time = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    return time
  

def get_confusion_matrix(config,
                         run_on_google_colab = False, 
                         iteration = 100, 
                         population = 10,
                         epsilon = 1, # The distance between two nodes in the space grid:
                         initSensorNum = 14, # initial sensor numbers
                         maxSensorNum = 25,  # max sensor numbers
                         radius = 1, # radius of the motion sensors
                         mutation_rate = 0.005, # Mutation rate for each item in a chromosome's data (each sensor placeholder)
                         crossover = 2, # number of folds in the crossover process
                         survival_rate = 0.1,
                         reproduction_rate = 0.2,
                         print_epochs = True,
                         ROS = False
                        ):
       
        global runningOnGoogleColab
        runningOnGoogleColab = run_on_google_colab
    

        import sys

        base_path = '../SensorConfigurationOptimization/'
        sys.path.append('..')
           
        ga = GA(population, 
                'expert', 
                base_path, 
                epsilon, 
                initSensorNum, 
                maxSensorNum, 
                radius, 
                mutation_rate, 
                crossover, 
                survival_rate, 
                reproduction_rate,
                ROS)
        

        return ga.calculate_confusion_matrix(config.GetSensorConfiguration(), True, False, False, False, 1)



def run(testbed,
        input_sensor_types = {'model_motion_sensor': True, 
                              'model_beacon_sensor': False,
                              'model_pressure_sensor': False,
                              'model_accelerometer': False,
                              'model_electricity_sensor': False},
        run_on_google_colab = False, 
        iteration = 100, 
        population = 10,
        epsilon = 1, # The distance between two nodes in the space grid:
        initSensorNum = 14, # initial sensor numbers
        maxSensorNum = 25,  # max sensor numbers
        radius = 1, # radius of the motion sensors
        mutation_rate = 0.005, # Mutation rate for each item in a chromosome's data (each sensor placeholder)
        crossover = 2, # number of folds in the crossover process
        survival_rate = 0.1, # top % of the sorted chromosomes of each generation that goes to the next generation
        reproduction_rate = 0.2, # top % of the sorted chromosomes of each generation that contributes in breeding children
        print_epochs = True,
        ROS = False
      ):
       
        global runningOnGoogleColab
        runningOnGoogleColab = run_on_google_colab
    

        import sys


        base_path = '../SensorConfigurationOptimization/'
        sys.path.append('..')

        

        # random_seed = 0.1234
        # random.seed(random_seed)

        results = []
        best_configuration_history = []

        print('----- Running GA for epsilon = ' + str(epsilon))
        f = IntProgress(min=0, max=iteration) # instantiate the bar
        display(f) # display the bar

        ga = GA(testbed,
                population, 
                'expert', 
                base_path, 
                epsilon, 
                initSensorNum, 
                maxSensorNum, 
                radius, 
                mutation_rate, 
                crossover, 
                survival_rate, 
                reproduction_rate,
                ROS)

        ga.RunFitnessFunction(True, False, False, False, 1)

        for epoch in range(iteration):
            f.value += 1
            ga.GetNextGeneration()
            ga.RunFitnessFunction(True, False, False, False, 1)
            ga.Selection()

            if (print_epochs == True):
                print("(epoch %d) ----- The best answer: {%f} with (%d) number of sensors (average fitness is: %f)" 
                      %(epoch + 1, 
                        ga.chromosomes[0].fitness + (sum(ga.chromosomes[0].grid) / 100) * 100,  
                        np.sum(ga.chromosomes[0].grid),
                        ga.AverageAnswer()))

            results.append([(c.fitness + (sum(c.grid) / 100) * 100, sum(c.grid)) for c in ga.chromosomes])
            best_configuration_history.append(ga.chromosomes[0])
            
        return results, best_configuration_history
        