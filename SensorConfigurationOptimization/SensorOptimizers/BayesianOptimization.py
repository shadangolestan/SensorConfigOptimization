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
from openbox import Optimizer
import CASAS.al as al
import pickle
from openbox import sp

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
        # simworldname = self.Data_path + '/Configuration Files/simulationWorld2.xml'
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

        testbed
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

        self.BOV =  BOVariables(base_path, 
                                self.testbed,
                                self.CONSTANTS['epsilon'], 
                                self.CONSTANTS['initial_samples'],
                                self.CONSTANTS['max_LS_sensors'], 
                                self.CONSTANTS['max_IS_sensors'], 
                                self.CONSTANTS['radius'],
                                self.CONSTANTS['initial_samples'],
                                ROS = True)

        self.CONSTANTS['width'] = self.BOV.space[0]
        self.CONSTANTS['height'] = self.BOV.space[1]


        self.dataBoundaries = self.MakeDataBoundaries(
                                            height = self.CONSTANTS['height'], 
                                            width = self.CONSTANTS['width'], 
                                            MaxLSSensors = self.CONSTANTS['max_LS_sensors']
                                           )
    
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

    
    #returns the name of the sensor
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
    
    
    #converts epoch time to human readable
    def convertTime(self, posix_timestamp):
        tz = pytz.timezone('MST')
        dt = datetime.fromtimestamp(posix_timestamp, tz)
        time = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return time
    
    def sigma_neighbours(self, sensorPositions):        
            num_sensors = int(len(sensorPositions))

            Xs = []
            Ys = []
            for sensor in sensorPositions:
                Xs.append(sensor[0])
                Ys.append(sensor[1])

            def clamp(num, min_value, max_value):
                return max(min(num, max_value), min_value)

            import random
            import numpy as np
            neighbours = 200
            Ns = []
            for i in range(neighbours):  
                # TODO: change it to work with different sensor types:
                N_sensorType = 1
                
                N = []
                for s in range(num_sensors):
                    N_x = clamp(Xs[s] + (self.error/self.space[0] * random.randint(-1,1)), 0, 1)
                    N_y = clamp(Ys[s] + (self.error/self.space[1] * random.randint(-1,1)), 0, 1)
                    N.append([N_x, N_y])

                Ns.append(N)  
                
            return Ns

    
    def function_to_be_optimized(self, config):
        sensorPositions = []
        sensorTypes = []
        sensor_xy = []

        if (self.LSsensorTypesNum > 0):
            for i in range(1, self.CONSTANTS['max_LS_sensors'] + 1):
                sensor_xy = ast.literal_eval(config['ls' + str(i)])
                # sensorPositions.append([sensor_xy[0]*self.CONSTANTS['epsilon'], sensor_xy[1]*self.CONSTANTS['epsilon']])
                sensorPositions.append(sensor_xy)
                sensorTypes.append(config['ls_t' + str(i)])

        if (self.ISsensorTypesNum > 0):
            for i in range(1, self.CONSTANTS['max_IS_sensors'] + 1):
                sensor_xy = ast.literal_eval(config['is' + str(i)])
                sensorPositions.append(sensor_xy)
                sensorTypes.append(config['is_t' + str(i)])

        data = Data(sensorPositions, sensorTypes, self.BOV.space, self.CONSTANTS['epsilon'])

        return 100 - self.black_box_function(data, 
                                             simulateMotionSensors = self.sensor_types['model_motion_sensor'],
                                             simulateEstimotes = self.sensor_types['model_beacon_sensor'],
                                             simulateIS = (self.sensor_types['model_pressure_sensor'] and
                                                           self.sensor_types['model_accelerometer'] and
                                                           self.sensor_types['model_electricity_sensor']))

    
    def MakeDataBoundaries(self, height = 10.5, width = 6.6, MaxLSSensors = 15):
        from collections import defaultdict, OrderedDict

        d = dict()

        for idx in range(MaxLSSensors):
                d['x' + str(idx)] = (0.5, width - 0.5)
                d['y' + str(idx)] = (0.5, height - 0.5)

        return d
        
    def single_function_evaluation(self, config_x, config_y):
        sensorPositions = []
        sensorTypes = []
        sensor_xy = []
        excluded = []


        for i in range(len(config_x)):
            sensor_xy.append(config_x[i] * self.CONSTANTS['epsilon'])
            sensor_xy.append(config_y[i] * self.CONSTANTS['epsilon'])
            sensorTypes.append(1)
            sensorPositions.append(sensor_xy)
            sensor_xy = []

        

        data = Data(sensorPositions, sensorTypes, self.BOV.space, self.CONSTANTS['epsilon'])


        # print(sensorTypes)

        return 100 - self.black_box_function(data, 
                                             simulateMotionSensors = self.sensor_types['model_motion_sensor'],
                                             simulateEstimotes = self.sensor_types['model_beacon_sensor'],
                                             simulateIS = (self.sensor_types['model_pressure_sensor'] and
                                                           self.sensor_types['model_accelerometer'] and
                                                           self.sensor_types['model_electricity_sensor']))


    def is_valid(self, sensor_placeholder):
        # This is for checking locations where placing sensors are not allowed. 
        # TODO: the restricted area needs to be read from a config file.

        if sensor_placeholder[0] <= 2 and sensor_placeholder[1] <= 2:
            return False
        else:
            return True


    def BuildConfigurationSearchSpace(self, initial_state):
        list_of_variables = []
        if (self.LSsensorTypesNum > 0):
            ls = []
            for sensor_placeholder in self.BOV.grid:
                if self.is_valid(sensor_placeholder):
                    ls.append(str(sensor_placeholder))

            for i in range(1, self.CONSTANTS['max_LS_sensors'] + 1):                
                list_of_variables.append(sp.Categorical("ls" + str(i), ls, default_value= random.choice(ls)))
                if self.LSsensorTypesNum > 1:
                        list_of_variables.append(sp.Int("ls_t" + str(i), 1, self.LSsensorTypesNum, default_value=random.randint(1, self.LSsensorTypesNum)))

                else:
                    list_of_variables.append(sp.Constant("ls_t" + str(i), 1))


        if (self.ISsensorTypesNum > 0):
            for i in range(1, self.CONSTANTS['max_IS_sensors'] + 1):
                list_of_variables.append(sp.Categorical('is' + str(i), choices = self.BOV.objects, default_value = random.choice(self.BOV.objects)))
                if self.ISsensorTypesNum > 1:
                    list_of_variables.append(sp.Categorical("is_t" + str(i), choices = self.is_sensor_types, default_value = random.choice(self.is_sensor_types)))
                else:
                    list_of_variables.append(sp.Constant("is_t" + str(i), self.is_sensor_types[0]))

        return list_of_variables

    def run(self, RLBO = False):

        # Define Search Space
        self.space = sp.Space()
        self.RLBO = RLBO

        
        list_of_variables = self.BuildConfigurationSearchSpace(self.initial_state)

        self.space.add_variables(list_of_variables)
        history_list = []

        opt = Optimizer(
            self.function_to_be_optimized,
            self.space,
            max_runs = self.CONSTANTS['iterations'],
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