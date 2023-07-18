# GA ALGORITHM

from ipywidgets import IntProgress
from IPython.display import display
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
        motion_places = ['M00' + str(i) if i < 10 else 'M0' + str(i) for i in range(1,32)]
        # door_places = ['D00' + str(i) for i in range(1,4)]
        # t_places = ['T00' + str(i)  for i in range(1,5)]

        self.sensors_option = motion_places # + door_places + t_places

        self.radius = 1
        self.testbed = testbed

        if len(args) < 5:
            self.epsilon = args[3]
            self.initSensorNum = args[2]
            self.mode = args[0]
            self.space = args[1]
            self.placeHolders = []
            self.fitness = -1
            self.SensorConfigurationSetup()
              
        elif len(args) == 5:
            self.epsilon = args[3]
            self.placeHolders = []
            self.fitness = -1
            self.grid = args[0]
            self.mode = args[1]
            self.space = args[2]

        

    def SensorConfigurationSetup(self):
        self.grid = np.zeros(len(self.sensors_option)).tolist()
 
        i = 0
        while i < self.initSensorNum:
            cell = random.randrange(len(self.grid))
            if self.grid[cell] == 0:
                self.grid[cell] = 1
                i += 1

    def get_sensors_names(self):
       chosen_sensors = ['M000']
       for i in range(len(self.grid)):
          if self.grid[i] == 1:
             chosen_sensors.append(self.sensors_option[i])

       return chosen_sensors
             

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

        for i in range(population):
            self.chromosomes.append(Chromosome(testbed, self.mode, None, self.initSensorNum, self.epsilon))

        motion_places = ['M00' + str(i) if i < 10 else 'M0' + str(i) for i in range(1,32)]
        # door_places = ['D00' + str(i) for i in range(1,4)]
        # t_places = ['T00' + str(i)  for i in range(1,5)]

        self.sensors = motion_places # + door_places + t_places

        input_file = 'RealWorldDataset/aruba/data'
        D = self.convert_data(input_file)
        self.data = self.separate_by_day(D)
        self.activities = self.get_unique_activities(self.data)

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

    def RunFitnessFunction(self, simulateMotionSensors, simulateEstimotes, SimulateIS, Plotting, iteration):       
        for index, chromosome in enumerate(self.chromosomes):
            import CASAS.al as al
            sensors = chromosome.get_sensors_names()

            data = self.filter_data_by_sensors(self.data, sensors)
            train_data, test_data = self.split_train_test_data(data, 0.7)
            f1_score = (al.one_occupant_model(train_data, test_data, sensors))

            try:
                chromosome.fitness = (f1_score - (sum(chromosome.grid) / 100)) * 100

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
        