#--------------------------------------------------------------------- RL Parameters:
window_size = 10
buffer_size = 1
step_size = 1

performance = []


#--------------------------------------------------------------------- Experiment Parameters:
import pybenchfunction.function as benchmarks
D = 10
function = benchmarks.Powell(D)
gradient_fantacy = True
std_list = dict()

domain = {
    'lower_bound': 0,
    'upper_bound': 1
}

# [5.582, 0.786, 4.595, 2.815, 1.068, 0.851, 3.076, 2.796, 0.860, 5.623]


testbed = 'Testbed2/'
radius = 1
epsilon = 1


sensor_types = {
    'model_motion_sensor': True,
    'model_beacon_sensor': False,
    'model_pressure_sensor': False,
    'model_accelerometer': False,
    'model_electricity_sensor': False
}

'''
sensor_types = {
    'model_motion_sensor': True,
    'model_beacon_sensor': False,
    'model_pressure_sensor': True,
    'model_accelerometer': True,
    'model_electricity_sensor': True
}
'''

if testbed == 'Testbed1/':

    space = [(0.0, 0.0), (0.0, 8.0), (8.0, 8.0), (8.0, 0.0)]
    rooms = {'bedroom': [(3.9, 0.0), (8.0, 4.4)],
            'livingroom': [(0.0, 1.9), (6.3, 6.7)],
            'diningroom': [(0.0, 3.0), (2.9, 8.0)],
            'kitchen': [(0.0, 3.0), (2.9, 8.0)],
            'bathroom': [(6.1, 3.2), (8.0, 6.7)],
            'storage': [(2.8, 6.4), (8.0, 8.0)]} 
    objects = ['0.5, 2.7', '3.5, 2.7', '6.7, 1.4', '4.2, 3.2', '1.7, 6.0', '6.0, 3.6', '7.4, 3.6', '1.0, 5.5', '6.8, 5.5', '0.5, 7.1', '2.2, 7.1', '7.1, 6.8']

if testbed == 'Testbed2/':

    space = [(0.0, 0.0), (0.0, 8.0), (5.3, 8.0), (5.3, 0.0)]
    rooms = {'bedroom': [(0.0, 1.9), (3.0, 4.7)],
            'livingroom': [(1.6, 0.0), (5.3, 3.3)],
            'diningroom': [(3.1, 2.0), (5.3, 3.3)],
            'kitchen': [(3.0, 3.3), (5.3, 6.0)],
            'bathroom': [(0.0, 4.9), (2.4, 6.9)],
            'entryway': [(2.4, 6.0), (5.3, 8.0)]}
    objects = ['0.5, 2.7', '3.5, 2.7', '6.7, 1.4', '4.2, 3.2', '1.7, 6.0', '6.0, 3.6', '7.4, 3.6', '1.0, 5.5', '6.8, 5.5', '0.5, 7.1', '2.2, 7.1', '7.1, 6.8']


#--------------------------------------------------------------------- GA Parameters:
iteration = 100
population = 10
initSensorNum = 7
maxSensorNum = 10
mutation_rate = 0.005
crossover = 2
survival_rate = 0.1
reproduction_rate = 0.2

#--------------------------------------------------------------------- BO Parameters:
acquisition_function = 'kg'
# acquisition_function = 'lcb'
acq_optimizer_type = 'auto'
surrogate_model = 'prf'
ROS = True
error = 0.0
multi_objective = False
LSsensorsNum = 10
ISsensorsNum = 0
initial_state = 'random'
bo_iteration = 1000
RLBO = True
info_matrix = None
pivots_granularity = 1
pivots_granularity = epsilon
cutoff_treshold = 70
configuration_star = None
config_advisor = None
config_space = None
iteration_id = 0
#--------------------------------------------------------------------- CASAS Parameters:
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=80,
                             max_features=8,
                             bootstrap=True,
                             criterion="entropy",
                             min_samples_split=20,
                             max_depth=None,
                             n_jobs=4,
                             class_weight='balanced')