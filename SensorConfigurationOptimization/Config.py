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


testbed = ''
radius = 1
epsilon = None


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


#--------------------------------------------------------------------- GA Parameters:
iteration = 100
population = 10
initSensorNum = 7
maxSensorNum = 10
mutation_rate = 0.005
crossover = 2
survival_rate = 0.1
reproduction_rate = 0.2

#--------------------------------------------------------------------- BO/DGBO Parameters:
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
pivots_granularity = None
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