import math
import os
import pathlib
from abc import abstractmethod

import numpy as np
import torch
from sklearn.svm import SVR
from torch import Tensor
from xgboost import XGBRegressor
import pandas as pd
from copy import deepcopy

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
import pickle
from datetime import datetime
import CASAS.al as al
import Config as cf
import SIM_SIS_Libraries.SensorsClass
import SIM_SIS_Libraries.SIM_SIS_Simulator as sim_sis
import SIM_SIS_Libraries.ParseFunctions as pf
import Config as cf


class Data:
    def __init__(self, sensorPositions, sensorTypes, space, epsilon):
        self.radius = 1
        self.placeHolders = sensorPositions

        self.sensorTypes = sensorTypes
        self.epsilon = epsilon
        self.space = space
        # self.SensorPlaceHolderSetup()
        self.sensitivity = {'3': 'pressure',
                            '4': 'accelerometer',
                            '5': 'electricity'}
        
    def frange(self, start, stop, step):
        steps = []
        while start <= stop:
            steps.append(start)
            start +=step
            
        return steps

    def GetSensorConfiguration(self):
        from collections import Counter
        sensorLocations = self.GetSensorLocations()
        _, rooms, _ = pf.ParseWorld(simworldname = '')
        summaryDict = Counter(self.sensorTypes)

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

            if (self.sensorTypes[index] == 1):
                configurationDetails.append(tuple([loc, room, 'motion sensors']))

            elif (self.sensorTypes[index] == 2):
                configurationDetails.append(tuple([loc, room, 'beacon sensors']))
                
            elif (self.sensorTypes[index] == 3):
                configurationDetails.append(tuple([loc, room, 'IS', self.sensitivity['3']]))
                
            elif (self.sensorTypes[index] == 4):
                configurationDetails.append(tuple([loc, room, 'IS', self.sensitivity['4']]))
                
            elif (self.sensorTypes[index] == 5):
                configurationDetails.append(tuple([loc, room, 'IS', self.sensitivity['5']]))

            else:
                configurationDetails.append(tuple([loc, room, 'motion sensors']))
        
        sensor_config = [[configurationSummary, [tuple(configurationDetails)]], self.radius]
        return sensor_config

    def GetSensorLocations(self):
        sensorLocations = []
        for index, sensorIndicator in enumerate(self.placeHolders):
            sensorLocations.append(self.placeHolders[index])

        return sensorLocations


class TestFunction:
    """
    The abstract class for all benchmark functions acting as objective functions for BO.
    Note that we assume all problems will be minimization problem, so convert maximisation problems as appropriate.
    """

    # this should be changed if we are tackling a mixed, or continuous problem, for e.g.
    problem_type = "categorical"

    def __init__(self, normalize=True, **kwargs):
        self.normalize = normalize
        self.n_vertices = None
        self.config = None
        self.dim = None
        self.continuous_dims = None
        self.categorical_dims = None
        self.int_constrained_dims = None

    def _check_int_constrained_dims(self):
        if self.int_constrained_dims is None:
            return
        assert self.continuous_dims is not None, (
            "int_constrained_dims must be a subset of the continuous_dims, " "but continuous_dims is not supplied!"
        )
        int_dims_np = np.asarray(self.int_constrained_dims)
        cont_dims_np = np.asarray(self.continuous_dims)
        assert np.all(np.in1d(int_dims_np, cont_dims_np)), (
            "all continuous dimensions with integer "
            "constraint must be themselves contained in the "
            "continuous_dimensions!"
        )

    @abstractmethod
    def compute(self, x, normalize=None):
        raise NotImplementedError()

    def sample_normalize(self, size=None):
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for _ in range(size):
            x = np.array([np.random.choice(self.config[_]) for _ in range(self.dim)])
            y.append(self.compute(x, normalize=False,))
        y = np.array(y)
        return np.mean(y), np.std(y)

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)


class _MaxSAT(TestFunction):
    def __init__(self, data_filename, random_seed=None, normalize=False, **kwargs):
        super(_MaxSAT, self).__init__(normalize, **kwargs)
        base_path = os.path.dirname(os.path.realpath(__file__))
        f = open(os.path.join(base_path, "data/", data_filename), "rt")
        line_str = f.readline()
        while line_str[:2] != "p ":
            line_str = f.readline()
        self.n_variables = int(line_str.split(" ")[2])
        self.n_clauses = int(line_str.split(" ")[3])
        self.n_vertices = np.array([2] * self.n_variables)
        self.config = self.n_vertices
        clauses = [(float(clause_str.split(" ")[0]), clause_str.split(" ")[1:-1]) for clause_str in f.readlines()]
        f.close()
        weights = np.array([elm[0] for elm in clauses]).astype(np.float32)
        weight_mean = np.mean(weights)
        weight_std = np.std(weights)
        self.weights = (weights - weight_mean) / weight_std
        self.clauses = [
            ([abs(int(elm)) - 1 for elm in clause], [int(elm) > 0 for elm in clause]) for _, clause in clauses
        ]

    def compute(self, x, normalize=None):
        if not isinstance(x, torch.Tensor):
            try:
                x = torch.tensor(x.astype(int))
            except:
                raise Exception("Unable to convert x to a pytorch tensor!")
        return self.evaluate(x)

    def evaluate(
        self, x,
    ):
        assert x.numel() == self.n_variables
        if x.dim() == 2:
            x = x.squeeze(0)
        x_np = (x.cpu() if x.is_cuda else x).numpy().astype(bool)
        satisfied = np.array([(x_np[clause[0]] == clause[1]).any() for clause in self.clauses])
        return np.sum(self.weights * satisfied) * x.float().new_ones(1, 1)


class MaxSAT60(_MaxSAT):
    def __init__(self, n_binary, **tkwargs):
        super().__init__(data_filename="frb-frb10-6-4.wcnf")
        self.n_binary = n_binary
        self.binary_inds = list(range(n_binary))
        self.n_continuous = 0
        self.continuous_inds = []
        self.n_categorical = 0
        self.categorical_inds = []
        self.dim = n_binary
        self.bounds = torch.stack((torch.zeros(n_binary), torch.ones(n_binary))).to(**tkwargs)



class BOVariables:
    def __init__(self, base_path, testbed, epsilon, initSensorNum, maxLSSensorNum, maxISSensorNum, radius, sampleSize, ROS):
        self.epsilon = epsilon
        self.Data_path = base_path + testbed
        self.initSensorNum = initSensorNum
        self.maxLSSensorNum = maxLSSensorNum
        self.maxISSensorNum = maxISSensorNum
        self.radius = radius
        self.sensor_distribution, self.types, self.space, self.rooms, self.objects, self.agentTraces = self.ModelsInitializations(ROS)
        self.CreateGrid()

    def CreateGrid(self):
        x = self.space[0]
        y = self.space[1]

        W = []
        start = self.epsilon

        while start < x:
            W.append(start)
            start += self.epsilon

        H = []
        start = self.epsilon

        while start < y:
            H.append(start)
            start += self.epsilon

        self.grid = []

        for w in W:
            for h in H:
                self.grid.append([w, h])

    def ModelsInitializations(self, ROS):
        #----- Space and agent models -----: 
        simworldname = ''
        agentTraces = []
        
        if ROS:
            directory = os.fsencode(self.Data_path + 'Agent Trace Files ROS/')
        else:
            directory = os.fsencode(self.Data_path + 'Agent Trace Files/')
            
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".csv"): 
                if ROS:
                    agentTraces.append(self.Data_path + 'Agent Trace Files ROS/' + filename)
                else:
                    agentTraces.append(self.Data_path + 'Agent Trace Files/' + filename)

        # Parsing the space model: 
        space, rooms, objects = pf.ParseWorld(simworldname)
        sim_sis.AddRandomnessToDatasets(self.epsilon, self.Data_path, rooms)

        space = [space[-1][0], space[1][1]]

        # User parameters 
        types, sensor_distribution = pf.GetUsersParameters()

        roomsList = []
        for room in sensor_distribution:
            roomsList.append(room)
              
        return sensor_distribution, types, space, rooms, objects, agentTraces














class SIM(TestFunction):
    problem_type = "categorical"
    # Taken and adapted from the the MVRSM codebase
    def __init__(self, n_features: int, feature_costs: Tensor, lamda=1e-6, normalize=False, **tkwargs):
        # super(Ackley53, self).__init__(normalize)

        # input_file = 'RealWorldDataset/aruba/data'
        # D = self.convert_data(input_file)
        # self.data = self.separate_by_day(D)

        self.CONSTANTS = {
            'iterations': cf.iteration,
            'initial_samples': 10,
            'epsilon': cf.epsilon,
            'radius': cf.radius,
            'max_LS_sensors': cf.maxSensorNum,
            'max_IS_sensors': cf.maxSensorNum,
            'error': cf.error
        }

        base_path = '../bodi/'
        sys.path.append('..')

        self.BOV =  BOVariables(base_path, 
                                cf.testbed,
                                self.CONSTANTS['epsilon'], 
                                self.CONSTANTS['initial_samples'],
                                self.CONSTANTS['max_LS_sensors'], 
                                self.CONSTANTS['max_IS_sensors'], 
                                self.CONSTANTS['radius'],
                                self.CONSTANTS['initial_samples'],
                                ROS = True)

        if (cf.maxSensorNum > 0):
            ls = []
            for sensor_placeholder in self.BOV.grid:
                ls.append(str(sensor_placeholder))


        self.n_categorical = 0        
        self.n_continuous = 0
        self.continuous_inds = []
        self.n_binary = len(ls)

        self.binary_inds = list(range(self.n_binary))
        self.categorical_inds = [i for i in range(len(ls))]
        self.dim = self.n_binary + self.n_continuous + self.n_categorical
        
        
        # specifies the range for the continuous variables
        # self.lb, self.ub = np.array([-1, -1, -1]), np.array([+1, +1, +1])
        self.feature_idxs = torch.arange(50)
        self.bounds = torch.stack((torch.zeros(self.n_binary), torch.ones(self.n_binary))).to(**tkwargs)

        self.paceholders = self.grid_maker(self.BOV.space[0], self.BOV.space[1])
        self.feature_costs = feature_costs


    def is_valid(self, sensor_placeholder):
        # This is for checking locations where placing sensors are not allowed. 

        if cf.testbed == 'Testbed2' and sensor_placeholder[0] <= 2 and sensor_placeholder[1] <= 2:
            return False
        else:
            return True


    def convert_data(self, file_path):
        with open(file_path, 'r') as file:
                data = file.read()

        data = data.replace('\t', ' ')
        converted_data = []
        finalized_data = []
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
                    current_activity = 'None'

                else:
                    converted_data.append((timestamp, sensor_name, current_activity))

        for row in converted_data:
            if not 'None' in row:
                finalized_data.append(row)

        return finalized_data


    def convertTime(self, posix_timestamp):
        tz = pytz.timezone('MST')
        dt = datetime.fromtimestamp(posix_timestamp, tz)
        time = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return time

    def Name(self, number, typeSensor):
        if number < 10:
          return typeSensor + str(0) + str(number)
        else:
          return typeSensor + str(number)


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
          try:
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

          except:
            pass

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

    def black_box_function(self, sample, simulateMotionSensors = True, simulateEstimotes = False, simulateIS = False, Plotting = False):       
        files = []
        all_sensors = set([])

        for agentTrace in self.BOV.agentTraces:
            df_ = sim_sis.RunSimulator(self.BOV.space, 
                                       self.BOV.rooms, 
                                       agentTrace,
                                       sample.GetSensorConfiguration(), 
                                       simulateMotionSensors, 
                                       simulateEstimotes,
                                       simulateIS,
                                       Plotting, 
                                       self.BOV.Data_path)

            dataFile, sensors = self.PreProcessor(df_)
            all_sensors.update(sensors)
            files.append(dataFile)


        all_sensors = list(all_sensors)
        f1_score = (al.leave_one_out(files, all_sensors)[0]) * 100

        try:
            return f1_score[0]

        except:
            return f1_score

    def split_train_test_data(self, data, train_percentage):
        keys = list(data.keys())
        split_index = int(len(keys) * train_percentage)
        train_keys = keys[:split_index]
        test_keys = keys[split_index:]

        train_data = [item for key in train_keys for item in data[key]]
        test_data = [item for key in test_keys for item in data[key]]
        
        return train_data, test_data

    def grid_maker(self, X, Y):
        x = X
        y = Y

        W = []
        start = cf.epsilon

        while start < x:
            W.append(start)
            start += cf.epsilon

        H = []
        start = cf.epsilon

        while start < y:
            H.append(start)
            start += cf.epsilon

        grid = []

        for w in W:
            for h in H:
                if self.is_valid([w, h]):
                    grid.append([w, h])

        return grid

    def _SIM(self, config):
        sensorPositions = []
        sensorTypes = []
        sensor_xy = []

        for i, c in enumerate(config[0]):
            if c == 1:
                sensorPositions.append(self.paceholders[i])
                sensorTypes.append(1)
    

        print('number of sensors placed: ', len(sensorPositions))
        data = Data(sensorPositions, sensorTypes, self.BOV.space, self.CONSTANTS['epsilon'])

        res = 100 - self.black_box_function(data, 
                                    simulateMotionSensors = True,
                                    simulateEstimotes = False,
                                    simulateIS = False)

        feature_cost = self.feature_costs[inds_selected].sum().item()
        return [res, feature_cost]

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

        return day_data

    def compute(self, X, normalize=None):
        if type(X) == torch.Tensor:
            X = X.numpy()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # To make sure there is no cheating, round the discrete variables before calling the function
        # X[:, self.binary_inds] = np.round(X[:, self.binary_inds])
        # X[:, self.continuous_inds] = -1 + 2 * X[:, self.continuous_inds]
        result = self._SIM(X)
        return result







































class aruba(TestFunction):
    problem_type = "categorical"

    # Taken and adapted from the the MVRSM codebase
    def __init__(self, n_features: int, feature_costs: Tensor, lamda=1e-6, normalize=False, **tkwargs):
        # super(Ackley53, self).__init__(normalize)

        input_file = 'RealWorldDataset/aruba/data'
        D = self.convert_data(input_file)
        self.data = self.separate_by_day(D)

        self.n_categorical = 0
        
        self.n_continuous = 0
        self.continuous_inds = []

        self.motion_places = ['M00' + str(i) if i < 10 else 'M0' + str(i) for i in range(1,32)]
        # door_places = ['D00' + str(i) for i in range(1,4)]
        # t_places = ['T00' + str(i)  for i in range(1,5)]

        ls = self.motion_places # + door_places + t_places
        self.n_binary = len(ls)
        self.binary_inds = list(range(self.n_binary))
        self.categorical_inds = [i for i in range(len(ls))]
        self.dim = self.n_binary + self.n_continuous + self.n_categorical
        
        
        # specifies the range for the continuous variables
        # self.lb, self.ub = np.array([-1, -1, -1]), np.array([+1, +1, +1])
        self.feature_idxs = torch.arange(50)
        self.bounds = torch.stack((torch.zeros(self.n_binary), torch.ones(self.n_binary))).to(**tkwargs)
        self.feature_costs = feature_costs

    def convert_data(self, file_path):
        with open(file_path, 'r') as file:
                data = file.read()

        data = data.replace('\t', ' ')
        converted_data = []
        finalized_data = []
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
                    current_activity = 'None'

                else:
                    converted_data.append((timestamp, sensor_name, current_activity))

        for row in converted_data:
            if not 'None' in row:
                finalized_data.append(row)

        return finalized_data

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

    def black_box_function(self, train_data, test_data, sensors):   
        f1_score = (al.one_occupant_model(train_data, test_data, sensors)) * 100
        try:
            return f1_score

        except:
            return 0

    def split_train_test_data(self, data, train_percentage):
        keys = list(data.keys())
        split_index = int(len(keys) * train_percentage)
        train_keys = keys[:split_index]
        test_keys = keys[split_index:]

        train_data = [item for key in train_keys for item in data[key]]
        test_data = [item for key in test_keys for item in data[key]]
        
        return train_data, test_data

    def _aruba(self, config):
        sensors = ['M000']
        inds_selected = []
        for index, sensor_index in enumerate(config[0]):
            if sensor_index == 1:
                sensors.append(self.motion_places[int(sensor_index)])
                inds_selected.append(index)

        data = self.filter_data_by_sensors(self.data, sensors)
        
        train_data, test_data = self.split_train_test_data(data, 0.7)
        # print(len(train_data))
        # print(len(test_data))
        res = 100 - self.black_box_function(train_data, test_data, sensors)
        feature_cost = self.feature_costs[inds_selected].sum().item()
        return [res, feature_cost]

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

        return day_data

    def compute(self, X, normalize=None):
        if type(X) == torch.Tensor:
            X = X.numpy()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # To make sure there is no cheating, round the discrete variables before calling the function
        # X[:, self.binary_inds] = np.round(X[:, self.binary_inds])
        # X[:, self.continuous_inds] = -1 + 2 * X[:, self.continuous_inds]
        result = self._aruba(X)
        return result










class Ackley53(TestFunction):
    problem_type = "mixed"

    # Taken and adapted from the the MVRSM codebase
    def __init__(self, lamda=1e-6, normalize=False, **tkwargs):
        super(Ackley53, self).__init__(normalize)
        self.n_binary = 50
        self.binary_inds = list(range(self.n_binary))
        self.n_continuous = 3
        self.continuous_inds = [50, 51, 52]
        self.n_categorical = 0
        self.categorical_inds = []
        self.dim = self.n_binary + self.n_continuous
        self.bounds = torch.stack((torch.zeros(self.dim), torch.ones(self.dim))).to(**tkwargs)
        self.n_vertices = 2 * np.ones(len(self.binary_inds), dtype=int)
        self.config = self.n_vertices
        self.lamda = lamda
        # specifies the range for the continuous variables
        # self.lb, self.ub = np.array([-1, -1, -1]), np.array([+1, +1, +1])
        self.feature_idxs = torch.arange(50)

    @staticmethod
    def _ackley(X):
        a = 20
        b = 0.2
        c = 2 * np.pi
        sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(np.square(X), axis=1) / 53))
        cos_term = -1 * np.exp(np.sum(np.cos(c * np.copy(X)) / 53, axis=1))
        result = a + np.exp(1) + sum_sq_term + cos_term
        return result

    def compute(self, X, normalize=None):
        if type(X) == torch.Tensor:
            X = X.numpy()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # To make sure there is no cheating, round the discrete variables before calling the function
        X[:, self.binary_inds] = np.round(X[:, self.binary_inds])
        X[:, self.continuous_inds] = -1 + 2 * X[:, self.continuous_inds]
        result = self._ackley(X)
        return -1*(result + self.lamda * np.random.rand(*result.shape))[0]


class LABS(object):
    def __init__(self, n_binary, **tkwargs):
        self.n_binary = n_binary
        self.binary_inds = list(range(n_binary))
        self.n_continuous = 0
        self.continuous_inds = []
        self.n_categorical = 0
        self.categorical_inds = []
        self.dim = n_binary
        self.bounds = torch.stack((torch.zeros(n_binary), torch.ones(n_binary))).to(**tkwargs)

    def __call__(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return torch.tensor([self._evaluate_single(xx) for xx in x]).to(x)

    def _evaluate_single(self, x_eval):
        x = deepcopy(x_eval)
        assert x.dim() == 1
        if x.dim() == 2:
            x = x.squeeze(0)
        assert x.shape[0] == self.n_binary
        x = x.cpu().numpy()
        N = x.shape[0]
        x[x == 0] = -1.
        # print(f'x transformed {x}')
        E = 0  # energy
        for k in range(1, N):
            C_k = 0
            for j in range(0, N - k):
                C_k += (x[j] * x[j + k])
            E += C_k ** 2
        if E == 0:
            print("found zero")
        return (N**2)/ (2 * E)

def load_uci_data(seed, n_features):
    try:
        path = str(pathlib.Path(__file__).parent.resolve()) + "/data/slice_localization_data.csv"
        df = pd.read_csv(path, sep=",")
    except:
        raise ValueError(
            "Failed to load `slice_localization_data.csv`. The slice dataset can be downloaded "
            "from: https://archive.ics.uci.edu/ml/datasets/Relative+location+of+CT+slices+on+axial+axis"
        )
    data = df.to_numpy()

    # Get the input data
    X = data[:, :-1]
    X -= X.min(axis=0)
    X = X[:, X.max(axis=0) > 1e-6]  # Throw away constant dimensions
    X = X / (X.max(axis=0) - X.min(axis=0))
    X = 2 * X - 1
    assert X.min() == -1 and X.max() == 1

    # Standardize targets
    y = data[:, -1]
    y = (y - y.mean()) / y.std()

    # Only keep 10,000 data points and n_features features
    shuffled_indices = np.random.RandomState(0).permutation(X.shape[0])[:10_000]  # Use seed 0
    X, y = X[shuffled_indices], y[shuffled_indices]

    # Use Xgboost to figure out feature importances and keep only the most important features
    xgb = XGBRegressor(max_depth=8).fit(X, y)
    inds = (-xgb.feature_importances_).argsort()
    X = X[:, inds[:n_features]]

    # Train/Test split on a subset of the data
    train_n = int(math.floor(0.50 * X.shape[0]))
    train_x, train_y = X[:train_n], y[:train_n]
    test_x, test_y = X[train_n:], y[train_n:]

    return train_x, train_y, test_x, test_y


class SVM:
    def __init__(self, n_features: int, feature_costs: Tensor, **tkwargs):
        self.train_x, self.train_y, self.test_x, self.test_y = load_uci_data(seed=0, n_features=n_features)
        self.n_binary = n_features
        self.binary_inds = list(range(n_features))
        self.n_continuous = 3
        self.continuous_inds = list(range(n_features, n_features + 3))
        self.n_categorical = 0
        self.categorical_inds = []
        self.dim = n_features + 3
        self.bounds = torch.stack((torch.zeros(self.dim), torch.ones(self.dim))).to(**tkwargs)
        assert feature_costs.shape == (n_features,) and feature_costs.min() >= 0
        self.feature_costs = feature_costs

    def __call__(self, x: Tensor):
        assert x.shape == (self.dim,)
        assert (x >= self.bounds[0]).all() and (x <= self.bounds[1]).all()
        assert ((x[self.binary_inds] == 0) | (x[self.binary_inds] == 1)).all()  # Features must be 0 or 1
        inds_selected = np.where(x[self.binary_inds].cpu().numpy() == 1)[0]
        if len(inds_selected) == 0:  # Silly corner case with no features
            rmse, feature_cost = 1.0, 0.0
        else:
            epsilon = 0.01 * 10 ** (2 * x[-3])  # Default = 0.1
            C = 0.01 * 10 ** (4 * x[-2])  # Default = 1.0
            gamma = (1 / self.n_binary) * 0.1 * 10 ** (2 * x[-1])  # Default = 1.0 / self.n_features
            model = SVR(C=C.item(), epsilon=epsilon.item(), gamma=gamma.item())
            model.fit(self.train_x[:, inds_selected], self.train_y)  #
            pred = model.predict(self.test_x[:, inds_selected])
            rmse = math.sqrt(((pred - self.test_y) ** 2).mean(axis=0).item())
            feature_cost = self.feature_costs[inds_selected].sum().item()
        return [rmse, feature_cost]
