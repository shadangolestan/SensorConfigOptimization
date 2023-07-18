from ipywidgets import IntProgress
import sys
from scipy.stats import norm
from numpy import argmax
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
                 sensors = None, 
                 mode = None, 
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
        self.chosen_sensors = []
        self.fitness = -1
        self.chromosome_pointer = chromosome_pointer
        
        motion_places = ['M00' + str(i) if i < 10 else 'M0' + str(i) for i in range(1,32)]
        # door_places = ['D00' + str(i) for i in range(1,4)]
        # t_places = ['T00' + str(i)  for i in range(1,5)]

        self.sensors_option = motion_places # + door_places + t_places

        if greedy:
            if new:
                self.GreedySensorConfigurationSetup(counter)
            else:                
                self.chosen_sensors = sensors
            
        else:
            if new:
                self.SensorConfigurationSetup()
            else:
                self.chosen_sensors = sensors
                
    def GoToNeighbour(self):
        nonzeroind = list(np.nonzero(self.grid)[0])
        zeroind = list(np.where(np.array(self.grid) == 0)[0])
        
        sensor = random.choice(nonzeroind)
        emptyPlace = random.choice(zeroind)
        
        # TODO: for mutiple sensor types:
        self.grid[sensor] = 0
        self.grid[emptyPlace] = 1
        
    def GreedySensorConfigurationSetup(self, counter):
        self.chosen_sensors = [self.sensors_option[counter]]
        
    
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

        motion_places = ['M00' + str(i) if i < 10 else 'M0' + str(i) for i in range(1,32)]
        # door_places = ['D00' + str(i) for i in range(1,4)]
        # t_places = ['T00' + str(i)  for i in range(1,5)]

        self.sensors = motion_places # + door_places + t_places

        self.population = len(self.sensors)

        input_file = 'RealWorldDataset/aruba/data'
        D = self.convert_data(input_file)
        self.data = self.separate_by_day(D)
        self.activities = self.get_unique_activities(self.data)


        for i in range(self.population):
            self.chromosomes.append(Chromosome(sensors = None,
                                               mode = self.mode,
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
        
        self.results.append([(c.fitness, len(c.chosen_sensors)) for c in self.chromosomes])
        self.best_configuration_history.append(self.chromosomes[0])
        
        chosen_sensors = self.chromosomes[0].chosen_sensors
        self.current_configs = [Chromosome(sensors = chosen_sensors,
                                       mode = self.chromosomes[0].mode,
                                       initSensorNum = self.initSensorNum,
                                       epsilon = self.epsilon,
                                       new = False,
                                       sensorTypesNum = self.sensorTypesNum,
                                       greedy = True,
                                       counter = 1)]
        
        
        while picked_sensors < self.maxSensorNum:   
            print('-- picked ' + str(picked_sensors) + ' sensors so far | performance:', self.results[-1][0][0])
            self.test_configs = []

            for i in range(1, len(self.chromosomes)):
                new_chosen_sensors = chosen_sensors + self.chromosomes[i].chosen_sensors

                self.test_configs.append(Chromosome(sensors = new_chosen_sensors,
                                        mode = self.chromosomes[0].mode,
                                        initSensorNum = self.initSensorNum,
                                        epsilon = self.epsilon,
                                        new = False,
                                        sensorTypesNum = self.sensorTypesNum,
                                        greedy = True,
                                        counter = 1,
                                        chromosome_pointer = self.chromosomes[i].chromosome_pointer))

            
            self.RunFitnessFunction(self.test_configs, True, False, False, False, 1)            
            self.sortChromosomes(self.test_configs)
            
            self.results.append([(c.fitness, len(c.chosen_sensors)) for c in self.test_configs])
            self.best_configuration_history.append(self.test_configs[0])
            chosen_sensors = self.test_configs[0].chosen_sensors
            self.chromosomes = list(filter(lambda x: x.chromosome_pointer != self.test_configs[0].chromosome_pointer, self.chromosomes)) 
            
            picked_sensors = picked_sensors + 1
            

                             
        self.GreedyOutput = [Chromosome(sensors = copy.deepcopy(chosen_sensors),
                                        mode = self.chromosomes[0].mode,
                                        initSensorNum = self.initSensorNum,
                                        epsilon = self.epsilon,
                                        new = False,
                                        sensorTypesNum = self.sensorTypesNum,
                                        greedy = True,
                                        counter = 1)]
        
        print('Greedy Performance is (F1-score, sensors placed): ', self.results[-1][0])
        
    def separate_by_day(self, converted_data):
        day_data = {}
        
        day_number = 1
        previous_hour = 0


        for timestamp, sensor_name, activity in converted_data:
            if '.' in timestamp:
                time_format = '%Y-%m-%d %H:%M:%S.%f'
            else:
                time_format = '%Y-%m-%d %H:%M:%S'
            
            time = datetime.strptime(timestamp, time_format)

            if previous_hour > time.hour:
                day_number = day_number + 1
            
            if day_number not in day_data:
                day_data[day_number] = []
            
            day_data[day_number].append(timestamp + ' ' + sensor_name + ' ' + activity)
            previous_hour = time.hour
        
        first_five = {k: day_data[k] for k in list(day_data)[:10]}

        return first_five
    
    def split_train_test_data(self, data, train_percentage):
        keys = list(data.keys())
        split_index = int(len(keys) * train_percentage)
        train_keys = keys[:split_index]
        test_keys = keys[split_index:]

        train_data = [item for key in train_keys for item in data[key]]
        test_data = [item for key in test_keys for item in data[key]]
        
        return train_data, test_data

    def convert_data(self, file_path):
        with open(file_path, 'r') as file:
                data = file.read()

        data = data.replace('\t', ' ')
        converted_data = []
        # activity_map = {}
        
        current_activity = ''
        for line in data.split('\n'):
            line = line.strip()
            if line:
                parts = line.split(' ')
                date = parts[0]
                timestamp = date + ' ' + parts[1]
                sensor_name = parts[2] + ' ' + parts[2] + ' ' + parts[3]
                activity = ' '.join(parts[4:])
                
                # if sensor_name.startswith('M'):
                if activity.endswith('begin'):
                    activity_name = activity[:-6]
                    activity_name = activity_name.replace(' ', '')
                    current_activity = activity_name
                    # activity_map[sensor_name] = activity_name
                    converted_data.append((timestamp, sensor_name, activity_name))

                elif activity.endswith('end'):
                    converted_data.append((timestamp, sensor_name, current_activity))
                    current_activity = 'Walking'

                else:
                    converted_data.append((timestamp, sensor_name, current_activity))

        with open('RealWorldDataset/aruba/data_2.txt', 'w') as file:
            for item in converted_data:
                file.write(str(item) + '\n')

        return converted_data

    def filter_data_by_sensors(self, data, sensor_names):
        filtered_data = {}
    
        for day, day_data in data.items():
            filtered_day_data = []

            preactivity = ''
            for entry in day_data:
                d = entry.split(' ')

                if entry.split(' ')[-3] in sensor_names:
                    filtered_day_data.append(entry)

                
                if d[5] != preactivity:
                    if preactivity == '':
                        null_data = d
                        null_data[2] = 'M000'
                        null_data[3] = 'M000'
                        null_data[4] = 'ON'
                        preactivity = d[5]

                    else:
                        null_data = d
                        null_data[2] = 'M000'
                        null_data[3] = 'M000'
                        null_data[4] = 'OFF'
                        null_data[5] = preactivity
                        preactivity = ''

                    filtered_day_data.append(' '.join(null_data))
                # print(filtered_day_data)
            
            if filtered_day_data:
                filtered_data[day] = filtered_day_data
        
        return filtered_data

    def get_unique_activities(self, data):
        unique_activities = set()
        
        for day in data.keys():
            for row in data[day]:
                activity = row.split(' ')[-1]
                unique_activities.add(activity)
            
        return list(unique_activities)

    def black_box_function(self, train_data, test_data, sensors):   
        f1_score = (al.one_occupant_model(train_data, test_data, sensors)) * 100
        try:
            return f1_score

        except:
            return 0

    def RunFitnessFunction(self, chroms, simulateMotionSensors, simulateEstimotes, simulateIS, Plotting, iteration):
        for index, chromosome in enumerate(chroms):
            sensors = chromosome.chosen_sensors + ['M000']
            data = self.filter_data_by_sensors(self.data, sensors)
            train_data, test_data = self.split_train_test_data(data, 0.7)
            
            f1_score = (al.one_occupant_model(train_data, test_data, sensors)) * 100
            try:
                chromosome.fitness = f1_score

            except:
                chromosome.fitness = 0
                

    def BestAnswer(self):
        bestAnswerIndex = 0
        for index, c in enumerate(self.chromosomes):
            if c.fitness > self.chromosomes[bestAnswerIndex].fitness:
                bestAnswerIndex = index

        return self.chromosomes[bestAnswerIndex]

    def AverageAnswer(self):
        return np.sum([c.fitness for c in self.chromosomes]) / len(self.chromosomes)

    
def run(testbed = 'Testbed1/',
        run_on_colab = False, 
        iteration = 1000, 
        population = 1,
        epsilon = 1, # The distance between two nodes in the space grid:
        LSSensorNum = 10,  # max location sensitive sensor numbers
        ISSensorNum = 10,  # max location sensitive sensor numbers
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

    global CONSTANTS
    global sensor_types
    global LSsensorTypesNum
    global ISsensorTypesNum

    CONSTANTS = {
        'iterations': iteration,
        'initial_samples': 10,
        'epsilon': epsilon,
        'radius': radius,
        'height': height,
        'width': width,
        'LS_sensors': LSSensorNum,
        'IS_sensors': LSSensorNum
    }
    
    sensor_types = input_sensor_types
    LSsensorTypesNum = sum(1 for condition in list(input_sensor_types.values())[0:2] if condition)
    ISsensorTypesNum = sum(1 for condition in list(input_sensor_types.values())[2:5] if condition)

    base_path = '../SensorConfigurationOptimization/'
    sys.path.append('..')
    
    GLS = GreedyAndLocalSearch(testbed, 
                               'expert', 
                               base_path, 
                               epsilon, 
                               CONSTANTS['LS_sensors'], 
                               CONSTANTS['IS_sensors'], 
                               radius, 
                               ROS,
                               learning_rate,
                               LSsensorTypesNum + ISsensorTypesNum)
    
    print('Running Greedy Algorithm... \n', end='')
    GLS.RunGreedyAlgorithm()
    print('[Done!]')
    
    GLS.RunFitnessFunction(GLS.GreedyOutput, True, False, False, False, 1)
    print('number of sensors:', len(GLS.GreedyOutput[0].chosen_sensors))
    print('sensors placed:', GLS.GreedyOutput[0].chosen_sensors)
    print('remaining queries:', CONSTANTS['iterations'])
    
    return GLS.results, GLS.best_configuration_history