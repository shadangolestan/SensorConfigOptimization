import SensorOptimizers.BayesianOptimization as bo
import numpy as np
import pickle
import Config as cf

maxSensorNum = int(np.min([(cf.space[2][0] / cf.epsilon) * (cf.space[2][1] / cf.epsilon), cf.LSsensorsNum]))


print('----- Running BO with: \n \t - epsilon: ', cf.epsilon, 
      '\n \t - testbed: ', cf.testbed,
      '\n \t - LS sensors #: ', cf.LSsensorsNum, 
      '\n \t - IS sensors #: ', cf.ISsensorsNum, 
      '\n \t - initial state: ', cf.initial_state,
      '\n \t - Acquisition Function: ', cf.acquisition_function)

for i in range(0, 1):
    BO = bo.BayesianOptimization(testbed = cf.testbed,
                                 iteration = cf.bo_iteration, 
                                 epsilon = cf.epsilon, 
                                 error = cf.error,
                                 ROS = True, 
                                 LSmaxSensorNum = maxSensorNum,
                                 ISmaxSensorNum = cf.ISsensorsNum, 
                                 initial_state = cf.initial_state,
                                 input_sensor_types = cf.sensor_types,
                                 acquisition_function = cf.acquisition_function,
                                 surrogate_model = cf.surrogate_model,
                                 acq_optimizer_type = cf.acq_optimizer_type)

    history = BO.run()

    with open('Results_BO/AAAI new 10 _ history(LS' + str(cf.LSsensorsNum) +  'IS' + str(cf.ISsensorsNum) + ')_' + str(i), 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print(history)