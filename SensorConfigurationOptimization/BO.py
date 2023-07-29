import sys
import numpy as np
import pickle
import Config as cf


if __name__ == "__main__":
    if len(sys.argv) != 6:
        print('Please provide:',
              '\n \t testbed name: testbed= (Testbed1, Testbed2, aruba (this is a realworld dataset))',
              '\n \t acquisition function name: acq_func= (ei, dg (our method))',
              '\n \t number of sensors: sensor#=',
              '\n \t epsilon (0.25, 0.5, 1): epsilon=')
        sys.exit(1)

    if sys.argv[1].split('=')[-1] != 'aruba':
        import SensorOptimizers.BayesianOptimization as bo
    else:
        import SensorOptimizers.RealWorld_BayesianOptimization as bo

    cf.testbed = (sys.argv[1].split('=')[-1] + '/').replace(' ', '')
    cf.acquisition_function = sys.argv[2].split('=')[-1]
    cf.LSsensorsNum = int(sys.argv[3].split('=')[-1])
    cf.epsilon = float(sys.argv[4].split('=')[-1])
    i=int(sys.argv[5].split('=')[-1])

    print('----- Running BO with: \n \t - epsilon (not effective for aruba): ', cf.epsilon, 
        '\n \t - testbed:', cf.testbed,
        '\n \t - LS sensors #: ', cf.LSsensorsNum, 
        '\n \t - IS sensors #: ', cf.ISsensorsNum, 
        '\n \t - initial state: ', cf.initial_state,
        '\n \t - gradient analysis: ', cf.gradient_fantacy,
        ' \n \t - AF: ', cf.acquisition_function)

    # for i in range(0, 5):
    BO = bo.BayesianOptimization(testbed = cf.testbed,
                                iteration = cf.bo_iteration, 
                                epsilon = cf.epsilon, 
                                error = cf.error,
                                ROS = True, 
                                LSmaxSensorNum = cf.LSsensorsNum,
                                ISmaxSensorNum = cf.ISsensorsNum, 
                                initial_state = cf.initial_state,
                                input_sensor_types = cf.sensor_types,
                                acquisition_function = cf.acquisition_function,
                                surrogate_model = cf.surrogate_model,
                                acq_optimizer_type = cf.acq_optimizer_type)

    history = BO.run()

    with open('Results_BO/history(' + cf.acquisition_function + '_' + cf.testbed.replace('/', '') + '_' + str(cf.epsilon * 100) + '_' + 'LS' + str(cf.LSsensorsNum) + ')_' + str(i), 'wb') as handle:
        pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    print(history)