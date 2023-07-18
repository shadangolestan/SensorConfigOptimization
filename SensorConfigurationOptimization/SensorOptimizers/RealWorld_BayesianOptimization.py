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
from openbox import Optimizer
import CASAS.al as al
import pickle
from openbox import sp
import Config as cf
from datetime import datetime


class BayesianOptimization:
    def __init__(self,
                 testbed = 'Testbed1/', 
                 surrogate_model = 'prf',
                 # acq_optimizer_type = 'random_scipy',
                 acq_optimizer_type = 'local_random',
                 acquisition_function = 'ei',
                 task_id = 'SPO',
                 iteration = 1000, 
                 epsilon = 1, # The distance between two nodes in the space grid:
                 error = 0.25,
                 LSmaxSensorNum = 15,  # max location sensitive sensor numbers
                 ISmaxSensorNum = 10,  # max location sensitive sensor numbers
                 radius = 1, # radius of the motion sensors
                 print_epochs = True,
                 height = 8.0,
                 width = 8.0,
                 ROS = False,
                 initial_state = 'fixed',
                 input_sensor_types = {'model_motion_sensor': True, 
                                       'model_beacon_sensor': False,
                                       'model_pressure_sensor': False,
                                       'model_accelerometer': False,
                                       'model_electricity_sensor': False}):
        
        input_file = 'RealWorldDataset/aruba/data'
        D = self.convert_data(input_file)
        self.data = self.separate_by_day(D)

        self.activities = self.get_unique_activities(self.data)

        self.surrogate_model = surrogate_model
        self.testbed = testbed
        self.acq_optimizer_type = acq_optimizer_type
        self.acquisition_function = acquisition_function
        self.task_id = task_id
        self.initial_state = initial_state

        self.is_sensor_types = []
        iss = list(input_sensor_types.values())[2:5]
        for t in range(len(iss)):
            if iss[t] == True:
                self.is_sensor_types.append(3 + t)

        self.sensor_types = input_sensor_types
        self.LSsensorTypesNum = sum(1 for condition in list(input_sensor_types.values())[0:2] if condition)
        self.ISsensorTypesNum = sum(1 for condition in list(input_sensor_types.values())[2:5] if condition)

        base_path = '../SensorConfigurationOptimization/'
        sys.path.append('..')

        # finalResults = []
        # w = self.CONSTANTS['width'] - 0.5
        # h = self.CONSTANTS['height'] - 0.5

        self.CONSTANTS = {
            'iterations': iteration,
            'initial_samples': 10,
            'epsilon': epsilon,
            'radius': radius,
            'height': height,
            'width': width,
            'max_LS_sensors': LSmaxSensorNum,
            'max_IS_sensors': ISmaxSensorNum,
            'error': error
        }

    def black_box_function(self, train_data, test_data, sensors):   
        f1_score = (al.one_occupant_model(train_data, test_data, sensors)) * 100


        try:
            return f1_score

        except:
            return 0
    
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
        
    def get_unique_activities(data):
        unique_activities = set()
        
        for day in data.keys():
            for row in data[day]:
                activity = row.split(' ')[-1]
                unique_activities.add(activity)
            
        return list(unique_activities)

    def split_train_test_data(self, data, train_percentage):
        keys = list(data.keys())
        split_index = int(len(keys) * train_percentage)
        train_keys = keys[:split_index]
        test_keys = keys[split_index:]

        train_data = [item for key in train_keys for item in data[key]]
        test_data = [item for key in test_keys for item in data[key]]
        
        return train_data, test_data

    def objective_function(self, config):
        sensors = ['M000']
        for sensor in config.keys():
            sensors.append(config[sensor])

        data = self.filter_data_by_sensors(self.data, sensors)
        
        train_data, test_data = self.split_train_test_data(data, 0.7)
        # print(len(train_data))
        # print(len(test_data))
        return 100 - self.black_box_function(train_data, test_data, sensors)
 
    def dictionary_to_matrix(self, dictionary):
        # Find the dimensions of the matrix
        max_row = int(max(eval(col)[0] for col in dictionary.keys()) / cf.pivots_granularity)
        max_col = int(max(eval(col)[1] for col in dictionary.keys()) / cf.pivots_granularity)

        # Initialize the matrix with zeros
        matrix = [[100] * (max_col + 1) for _ in range(max_row + 1)]

        # Fill in the values from the dictionary into the matrix
        for key, value in dictionary.items():
            col, row = eval(key)
            matrix[int(row / cf.pivots_granularity)][int(col / cf.pivots_granularity)] = value

        return matrix

    def create_pivots_matrix(self):
        self.greedy_map = dict()
        motion_places = ['M00' + str(i) if i < 10 else 'M0' + str(i) for i in range(1,32)]
        # door_places = ['D00' + str(i) for i in range(1,4)]
        # t_places = ['T00' + str(i) for i in range(1,5)]
        sensor_places = motion_places # + door_places + t_places

        for sp in sensor_places:

            data = self.filter_data_by_sensors(self.data, ['M000', sp])
            train_data, test_data = self.split_train_test_data(data, 0.7)
            self.greedy_map[sp] = 0
            self.greedy_map[sp] = self.black_box_function(train_data, test_data, ['M000', sp])

        return self.greedy_map

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
        
        first_five = {k: day_data[k] for k in list(day_data)[:30]}

        return first_five

    def get_unique_activities(self, data):
        unique_activities = set()
        
        for day in data.keys():
            for row in data[day]:
                activity = row.split(' ')[-1]
                unique_activities.add(activity)
            
        return list(unique_activities)

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

    def BuildConfigurationSearchSpace(self, initial_state, map_points_count = None):
        list_of_variables = []
        if (self.LSsensorTypesNum > 0):
            motion_places = ['M00' + str(i) if i < 10 else 'M0' + str(i) for i in range(1,32)]
            # door_places = ['D00' + str(i) for i in range(1,4)]
            # t_places = ['T00' + str(i)  for i in range(1,5)]

            ls = motion_places # + door_places + t_places

            for i in range(1, self.CONSTANTS['max_LS_sensors'] + 1):                
                list_of_variables.append(sp.Categorical("ls" + str(i), ls, default_value= random.choice(ls)))

        return list_of_variables

    def run(self, RLBO = False):
        # Define Search Space
        self.space = sp.Space()
        self.RLBO = RLBO

        if cf.acquisition_function == 'dg':
            cf.info_matrix = self.create_pivots_matrix()
        
        else:
            cf.info_matrix = []

        list_of_variables = self.BuildConfigurationSearchSpace(self.initial_state, map_points_count = None)

        self.space.add_variables(list_of_variables)
        history_list = []
        
        
        opt = Optimizer(
            self.objective_function,
            self.space,
            max_runs = self.CONSTANTS['iterations'] - len(cf.info_matrix),
            acq_optimizer_type = self.acq_optimizer_type,
            acq_type = self.acquisition_function,
            surrogate_type = self.surrogate_model,
            time_limit_per_trial=31000,
            task_id = self.task_id,
            epsilon = self.CONSTANTS['epsilon'],
            error = self.CONSTANTS['error']
        )
        if RLBO == True:
            history, s, a, r = opt.run_RLBO(RLBO = self.RLBO)
            return history, s, a, r
        else:
            history = opt.run()
            return history