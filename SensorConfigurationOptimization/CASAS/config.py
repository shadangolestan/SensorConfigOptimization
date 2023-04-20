# config.py
# Global variables

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import argparse
import copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import Config as cf



MODE_TRAIN = 'TRAIN'
MODE_TEST = 'TEST'
MODE_CV = 'CV'
MODE_PARTITION = 'PARTITION'
MODE_ANNOTATE = 'ANNOTATE'
MODE_WRITE = 'WRITE'
MODE_LOO = 'LOO'
MODES = list([MODE_TRAIN,
              MODE_TEST,
              MODE_CV,
              MODE_PARTITION,
              MODE_ANNOTATE,
              MODE_WRITE,
              MODE_LOO])


class Config:

    def __init__(self):
        """ Constructor
        """
        '''
        self.activitynames = ['Prepare', 'Start Session', 'Exercise', 'Toilet', 'Sit', 'Bath',
                               'wash', 'Fill Kettle', 'Boil Water', 'find ingredients',
                               'Make tea/coffee', 'find frying pan', 'find eggs', 'Cook eggs',
                               'find plate', 'find bread', 'toast bread', 'find fork and knife',
                               'setup table', 'Eat', 'Take medicine', 'Rinse',
                               'return ingredients', 'Drain water', 'Wipe', 'Get Broom',
                               'Broom Kitchen', 'Broom Diningroom', 'Empty dustpan',
                               'Return broom', 'Laundry', 'Grab Iron', 'Iron', 'Return Iron',
                               'Pick Up Tablet', 'Work with Tablet', 'Watch TV']
                               
        '''
        
        self.activitynames = []
        
        #['Phone', 'Wash_hands', 'Cook', 'Eat', 'Clean']
        self.current_seconds_of_day = 0
        self.current_timestamp = 0
        self.day_of_week = 0
        self.dominant = 0
        self.sensornames = ['M09', 'M04', 'M12', 'M01', 'M03', 'M10', 'M06', 'M00', 'M07', 'M05', 'M11', 'M02', 'M08'] 
        #['M00', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M11', 'M12', 'M13',
                            #'B00', 'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'B13']

        #['M01' , 'M02' , 'M03' , 'M04' , 'M05' , 'M06' , 'M07' , 'M08' , 'M09' , 'M10' ,
        #               'M11' , 'M12' , 'M13' , 'M14' , 'M15' , 'M16' , 'M17' , 'M18' , 'M19' , 'M20' ,
        #                'M21' , 'M22' , 'M23' , 'M24' , 'M25' , 'M26',
        #                'I01' , 'I02' , 'I03' , 'I04' , 'I05' , 'I06' , 'I07' , 'I08' ,
        #                'D01' , 'AD1-A' , 'AD1-B' , 'AD1-C' , 'asterisk' , 'E01']

        self.sensortimes = []
        self.data = []
        self.dstype = []
        self.labels = []
        self.numwin = 0
        self.wincnt = 0
        self.data_filename = "data"
        self.filter_other = True  # Do not consider other in performance
        self.ignore_other = False
        self.cluster_other = False
        self.no_overlap = False  # Do not allow windows to overlap
        self.confusion_matrix = True
        self.mode = MODE_TRAIN  # TRAIN, TEST, CV, PARTITION, ANNOTATE, WRITE, LOO
        self.model_path = "./model/"
        self.model_name = "model"
        self.num_activities = 0
        self.num_clusters = 10
        self.num_sensors = 0
        self.num_set_features = 14
        self.features = 0
        self.num_features = 0
        self.seconds_in_a_day = 86400
        self.max_window = 30
        self.add_pca = False  # Add principal components to feature vector
        self.weightinc = 0.01
        self.windata = np.zeros((self.max_window, 3), dtype=np.int)

        self.clf = cf.clf

        '''
        self.clf = RandomForestClassifier(n_estimators=80,
                                          max_features=8,
                                          bootstrap=True,
                                          criterion="entropy",
                                          min_samples_split=20,
                                          max_depth=None,
                                          n_jobs=4,
                                          class_weight='balanced')
        '''
        
        
        # self.clf = KNeighborsClassifier(n_neighbors = 7, p = 1, n_jobs = -1, weights = 'distance', algorithm = 'brute')
        # self.clf = GradientBoostingClassifier()

    def set_parameters(self, _sensornames):
        """ Set parameters according to command-line args list.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode',
                            dest='mode',
                            type=str,
                            choices=MODES,
                            default=self.mode,
                            help=('Define the core mode that we will run in, default={}.'
                                  .format(self.mode)))
        parser.add_argument('--data',
                            dest='data',
                            type=str,
                            default=self.data_filename,
                            help=('Data file of sensor data, default={}'
                                  .format(self.data_filename)))
        parser.add_argument('--model',
                            dest='model',
                            type=str,
                            default=self.model_name,
                            help=('Specifies the name of the model to use, default={}'
                                  .format(self.model_name)))
        parser.add_argument('--ignoreother',
                            dest='ignoreother',
                            default=self.ignore_other,
                            action='store_true',
                            help=('Ignores all sensor events affiliated with activity '
                                  'Other_Activity, default={}'.format(self.ignore_other)))
        parser.add_argument('--clusterother',
                            dest='clusterother',
                            default=self.cluster_other,
                            action='store_true',
                            help=('Divides the Other_Activity category into subclasses using '
                                  'k-means clustering.  When activated this sets --ignoreother '
                                  'to False, default={}'.format(self.cluster_other)))
        parser.add_argument('--sensors',
                            dest='sensors',
                            type=str,
                            default=','.join(self.sensornames),
                            help=('Comma separated list of sensors that appear in the data file, '
                                  'default={}'.format(','.join(self.sensornames))))
        parser.add_argument('--activities',
                            dest='activities',
                            type=str,
                            default=','.join(self.activitynames),
                            help=('Comma separated list of activities to use, '
                                  'default={}'.format(','.join(self.activitynames))))
        # parser.add_argument('files',
        #                     metavar='FILE',
        #                     type=str,
        #                     nargs='*',
        #                     help='Data files for AL to process.')
        
        
        # print('parser:', parser)
        
        # for i in parser:
        #     print(i)
        
        # args = parser.parse_args()


        
        
        # self.mode = args.mode
        self.mode = 'LOO'
        # self.data_filename = args.data
        self.data_filename = ''
        # self.model_name = args.model
        self.model_name = 'model'
        # self.ignore_other = args.ignoreother
        self.ignore_other = True
        # self.cluster_other = args.clusterother
        self.cluster_other = False
        if self.cluster_other:
            self.ignore_other = False
        self.sensornames = _sensornames
        self.num_sensors = len(self.sensornames)
        for i in range(self.num_sensors):
            self.sensortimes.append(0)
            self.dstype.append('n')
        # self.activitynames = str(args.activities)
        # self.activitynames = self.activitynames
        self.num_activities = len(self.activitynames)
        # files = list()
        # if len(args.files) > 0:
        #     files = copy.deepcopy(args.files)
        # else:
        #     files.append(self.data_filename)

        # return files
