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
        sensorLocations, sensorTypes = self.GetSensorLocations()
        _, rooms = pf.ParseWorld(simworldname = '')
        summaryDict = Counter(sensorTypes)

        # TODO: DIFFERENT SENSOR TYPE DEFINITIONS SHOULD BE ADDED HERE:
        configurationSummary = []
        for key in summaryDict:
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
                if loc[0] >= rooms[r][0][0] and loc[0] <= rooms[r][1][0] and loc[1] >= rooms[r][0][1] and loc[1] <= rooms[r][1][1]:
                    room = r
                    break

            if (sensorTypes[index] == 1):
                configurationDetails.append(tuple([loc, room, 'motion sensors']))

            elif (sensorTypes[index] == 2):
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
        sensorTypes = []
        for index, sensorIndicator in enumerate(self.placeHolders):
            sensorLocations.append(self.placeHolders[index])

            # TODO: DIFFERENT SENSOR TYPE DEFINITIONS SHOULD BE ADDED HERE:
            sensorTypes.append(1)

        return sensorLocations, sensorTypes


class BOVariables:
    def __init__(self, base_path, testbed, epsilon, initSensorNum, maxLSSensorNum, maxISSensorNum, radius, ROS):
        self.epsilon = epsilon
        self.Data_path = base_path + testbed
        self.initSensorNum = initSensorNum
        self.maxLSSensorNum = maxLSSensorNum
        self.maxISSensorNum = maxISSensorNum
        self.radius = radius
        self.sensor_distribution, self.types, self.space, self.rooms, self.agentTraces = self.ModelsInitializations(ROS)
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
        space, rooms = pf.ParseWorld(simworldname)
        sim_sis.AddRandomnessToDatasets(self.epsilon, self.Data_path, rooms)

        space = [space[-1][0], space[1][1]]

        # User parameters 
        types, sensor_distribution = pf.GetUsersParameters()

        roomsList = []
        for room in sensor_distribution:
            roomsList.append(room)
              
        return sensor_distribution, types, space, rooms, agentTraces


def frange(start, stop, step):
    steps = []
    while start <= stop:
        steps.append(start)
        start +=step
        
    return steps


def PreProcessor(df):
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
          
      except:
        pass
    
      try:
          for i in range(len(I)):
            sensorNames.append(Name(i, 'IS'))
            if I[i] == 1:
                  if (previous_I != None):
                    if (previous_I[i] == 0):
                      ISSensor_Names.append(Name(i,'IS'))
                      ISSensor_Message.append('ON')

                  else:
                    ISSensor_Names.append(Name(i,'IS'))
                    ISSensor_Message.append('ON')

            if previous_I != None:
                  if I[i] == 0 and previous_I[i] == 1:
                    ISSensor_Names.append(Name(i,'IS'))
                    ISSensor_Message.append('OFF')

          previous_I = I

      except:
          pass 

      '''
      # Beacon Sensor
      try:
        for i in range(len(B)):
              sensorNames.append(Name(i, 'B'))
              if B[i] > -200:
                    if (previous_B != None):
                        if (previous_B[i] == 0):
                          BeaconSensor_Names.append(Name(i,'B'))
                          if B[i] >= -50:
                              BeaconSensor_Message.append('1')
                            
                          elif B[i] < -50:
                              BeaconSensor_Message.append('0.5')


                    else:
                        BeaconSensor_Names.append(Name(i,'B'))
                        if B[i] >= -50:
                           BeaconSensor_Message.append('1')
                        
                        elif B[i] < -50:
                           BeaconSensor_Message.append('0.5')

              if previous_B != None:
                    if B[i] <= -200 and previous_B[i] > -200:
                        BeaconSensor_Names.append(Name(i,'B'))
                        BeaconSensor_Message.append('0')

        previous_M = M
        
      except:
        pass
       '''
    
    
       

      for m in range(len(MotionSensor_Names)):
        output_file.append(time +' '+ MotionSensor_Names[m] + ' ' + MotionSensor_Names[m] + ' ' + MotionSensor_Message[m] + ' ' + Activity)
        
      ''' 
      for b in range(len(BeaconSensor_Names)):
        output_file.append(time +' '+ BeaconSensor_Names[b] + ' ' + BeaconSensor_Names[b] + ' ' + BeaconSensor_Message[b] + ' ' + Activity)
      '''
    
      for i_s in range(len(ISSensor_Names)):
        output_file.append(time +' '+ ISSensor_Names[i_s] + ' ' + ISSensor_Names[i_s] + ' ' + ISSensor_Message[i_s] + ' ' + Activity)
        
      for s in sensorNames:
          sensors.add(s)
    
    # for row in output_file:
    #     print(row)
    
    
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

def MakeDataBoundaries(height = 10.5, width = 6.6, MaxSensors = 15):
    from collections import defaultdict, OrderedDict

    d = dict()

    for idx in range(MaxSensors):
            d['x' + str(idx)] = (0.5, width - 0.5)
            d['y' + str(idx)] = (0.5, height - 0.5)

    return d

def black_box_function(sample, simulateMotionSensors = True, simulateEstimotes = False, simulateIS = False, Plotting = False):        
    files = []
    all_sensors = set([])

    for agentTrace in BOV.agentTraces:
        df_ = sim_sis.RunSimulator(BOV.space, 
                                   BOV.rooms, 
                                   agentTrace,
                                   sample.GetSensorConfiguration(), 
                                   simulateMotionSensors, 
                                   simulateEstimotes,
                                   simulateIS,
                                   Plotting, 
                                   BOV.Data_path)
        
        dataFile, sensors = PreProcessor(df_)
        all_sensors.update(sensors)
        files.append(dataFile)
        

    all_sensors = list(all_sensors)
    
    f1_score = (al.leave_one_out(files, all_sensors)[0]) * 100
    
    try:
        return f1_score[0]
    
    except:
        return f1_score
    
'''    
def calculate_confusion_matrix(sample, simulateMotionSensors = True, simulateEstimotes = False, Plotting = False):       
    files = []
    all_sensors = set([])

    for agentTrace in BOV.agentTraces:
        df_ = sim_sis.RunSimulator(BOV.space, 
                                   BOV.rooms, 
                                   agentTrace,
                                   sample.GetSensorConfiguration(), 
                                   simulateMotionSensors, 
                                   simulateEstimotes, 
                                   Plotting, 
                                   BOV.Data_path)
        
        dataFile, sensors = PreProcessor(df_)
        all_sensors.update(sensors)
        files.append(dataFile)
        
    all_sensors = list(all_sensors)
    
    return al.get_confusion_matrix(files, all_sensors)
   
'''
def function_to_be_optimized(config):
    sensorPositions = []
    sensorTypes = []
    sensor_xy = []
    excluded = []
    
    if (LSsensorTypesNum > 0):
        for i in range(1, CONSTANTS['max_LS_sensors'] + 1):
            sensor_xy.append(config['x' + str(i)] * CONSTANTS['epsilon'])
            sensor_xy.append(config['y' + str(i)] * CONSTANTS['epsilon'])
            sensorTypes.append(config['t' + str(i)])
            sensorPositions.append(sensor_xy)
            sensor_xy = []
            
    
    if (ISsensorTypesNum > 0):
        for i in range(1, CONSTANTS['max_IS_sensors'] + 1):
            object_location = config['object_location' + str(i)].split(',')
            sensor_xy.append(float(object_location[0]))
            sensor_xy.append(float(object_location[1]))
            sensorTypes.append(config['t_o' + str(i)])
            sensorPositions.append(sensor_xy)
            sensor_xy = []

    data = Data(sensorPositions, sensorTypes, BOV.space, CONSTANTS['epsilon'])
            
    return 100 - black_box_function(data, 
                                    simulateMotionSensors = sensor_types['model_motion_sensor'],
                                    simulateEstimotes = sensor_types['model_beacon_sensor'],
                                    simulateIS = (sensor_types['model_pressure_sensor'] and
                                                  sensor_types['model_accelerometer'] and
                                                  sensor_types['model_electricity_sensor'])
                                   )

def run(config,
        testbed = 'Testbed1/',
        run_on_colab = False, 
        epsilon = 1, # The distance between two nodes in the space grid:
        LSmaxSensorNum = 15,  # max location sensitive sensor numbers
        ISmaxSensorNum = 10,  # max location sensitive sensor numbers
        radius = 1, # radius of the motion sensors
        print_epochs = True,
        height = 8.0,
        width = 8.0,
        ROS = False,
        input_sensor_types = {'model_motion_sensor': True, 
                              'model_beacon_sensor': False,
                              'model_pressure_sensor': False,
                              'model_accelerometer': False,
                              'model_electricity_sensor': False},
      ):

    global multi_objective_flag
    global CONSTANTS
    global runningOnGoogleColab
    global sensor_types
    global LSsensorTypesNum
    global ISsensorTypesNum
    
    runningOnGoogleColab = run_on_colab
    multi_objective_flag = False
    
    CONSTANTS = {
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
    
    
    base_path = '../SensorDeploymentOptimization/'

    sys.path.append('..')

    finalResults = []
    w = CONSTANTS['width'] - 0.5
    h = CONSTANTS['height'] - 0.5

    dataBoundaries = MakeDataBoundaries(
                                        height = CONSTANTS['height'], 
                                        width = CONSTANTS['width'], 
                                        MaxSensors = CONSTANTS['max_LS_sensors']
                                       )

    global BOV
    BOV =  BOVariables(base_path, 
                       testbed,
                       CONSTANTS['epsilon'], 
                       CONSTANTS['initial_samples'],
                       CONSTANTS['max_LS_sensors'], 
                       CONSTANTS['max_IS_sensors'], 
                       CONSTANTS['radius'],
                       # CONSTANTS['initial_samples'],
                       ROS = True
                      )

    # from openbox import sp
    
    # Define Search Space
    # space = sp.Space()
    history = function_to_be_optimized(config)
    return history