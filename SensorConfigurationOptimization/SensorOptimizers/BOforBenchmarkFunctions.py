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
import pickle
from openbox import sp
import Config as cf


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
        self.acq_optimizer_type = acq_optimizer_type
        self.acquisition_function = acquisition_function
        self.task_id = task_id
        self.initial_state = initial_state

        self.CONSTANTS = {
            'iterations': iteration,
            'initial_samples': 10,
        }
    
    def function_to_be_optimized(self, config):
        next_point = []
        
        
        for i in range(len(config)):
            next_point.append(config['x' + str(i)])

        # simple_regret = np.abs(self.function.get_global_minimum(self.function.d)[1][0] - self.function(next_point))
        return self.function(next_point)
    
    def BuildConfigurationSearchSpace(self, D):
        if D == 10:
            dim = [5.582, 0.786, 4.595, 2.815, 1.068, 0.851, 3.076, 2.796, 0.860, 5.623]
        list_of_variables = []
        # list_of_variables.append(sp.Real("x", -5, 0, default_value = random.uniform(-5, 0)))
        # list_of_variables.append(sp.Real("y", 10, 15, default_value = random.uniform(10, 15)))
        
        for i in range(D):
           # list_of_variables.append(sp.Real("x" + str(i), cf.domain['lower_bound'], cf.domain['upper_bound'], default_value = random.uniform(-cf.domain['lower_bound'], cf.domain['upper_bound'])))
        
            list_of_variables.append(sp.Real("x" + str(i), cf.domain['lower_bound'], cf.domain['upper_bound'], default_value = random.uniform(-cf.domain['lower_bound'], cf.domain['upper_bound'])))

        return list_of_variables

    def run(self, RLBO = False):
        self.function = cf.function
        # Define Search Space
        self.space = sp.Space()
        self.RLBO = RLBO

        
        list_of_variables = self.BuildConfigurationSearchSpace(cf.D)

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
        )
        if RLBO == True:
            history, s, a, r = opt.run_RLBO(RLBO = self.RLBO)
            return history, s, a, r
        else:
            history = opt.run()
            return history, None, None, None