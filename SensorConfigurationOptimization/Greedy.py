# Example:
# python3 Greedy.py testbed=aruba sensor#=5


import SensorOptimizers.Greedy as gr
import numpy as np
import pickle
import Config as cf
import sys

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Please provide: \n \t testbed= (Testbed1, Testbed2, aruba (this is a realworld dataset)) \n \t sensor#=')
        sys.exit(1)

    if sys.argv[1].split('=')[-1] != 'aruba':
        import SensorOptimizers.Greedy as gr
        maxSensorNum = int(np.min([(cf.space[2][0] / cf.epsilon) * (cf.space[2][1] / cf.epsilon), cf.LSsensorsNum]))
        cf.LSsensorsNum = min(maxSensorNum, int(sys.argv[2].split('=')[-1]))
        cf.testbed = sys.argv[1].split('=')[-1]

    else:
        import SensorOptimizers.RealWorld_Greedy as gr
        maxSensorNum = 40
        cf.LSsensorsNum = min(maxSensorNum, int(sys.argv[2].split('=')[-1]))
        cf.testbed = sys.argv[1].split('=')[-1]

    cf.testbed = sys.argv[1].split('=')[-1] + '/'

    print('----- Running Greedy with: \n \t - epsilon (not effective for aruba): ', cf.epsilon, 
        '\n \t - testbed: ', cf.testbed,
        '\n \t - sensors #:', cf.LSsensorsNum)

    for i in range(0, 5):
        history = gr.run(testbed = cf.testbed,
                        iteration = cf.iteration, 
                        epsilon = cf.epsilon, 
                        ROS = cf.ROS, 
                        multi_objective = cf.multi_objective, 
                        LSSensorNum = cf.LSsensorsNum, 
                        ISSensorNum = cf.ISsensorsNum, 
                        initial_state = cf.initial_state, 
                        input_sensor_types = cf.sensor_types)

        with open('Results_SA/history(LS' + str(cf.LSsensorsNum) + ')_' + str(i), 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)