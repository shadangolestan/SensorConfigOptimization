import numpy as np
import pickle
import Config as cf
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Please provide: \n \t testbed= (Testbed1, Testbed2, aruba (this is a realworld dataset))')
        sys.exit(1)
    
    if sys.argv[1].split('=')[-1] != 'aruba':
        import SensorOptimizers.GeneticAlgorithm as ga

    else:
        import SensorOptimizers.RealWorld_GeneticAlgorithm as ga
        cf.testbed = sys.argv[1].split('=')[-1]

    cf.testbed = sys.argv[1].split('=')[-1] + '/'

    for i in range(0, 1):
        print('----- ', 'Running Genetic Algorithm #', i, ':')
        result, best_configuration_history = ga.run(testbed = cf.testbed,
                                                    input_sensor_types = cf.sensor_types,
                                                    iteration = cf.iteration, 
                                                    population = cf.population,
                                                    epsilon = cf.epsilon, 
                                                    initSensorNum = cf.initSensorNum, 
                                                    maxSensorNum = cf.maxSensorNum,  
                                                    radius = cf.radius, 
                                                    mutation_rate = cf.mutation_rate, 
                                                    crossover = cf.crossover,
                                                    survival_rate = cf.survival_rate, 
                                                    reproduction_rate = cf.reproduction_rate,
                                                    print_epochs = True,
                                                    ROS = True
                                                )

        with open('GA_results/history' + str(i + 1), 'wb') as handle:
            pickle.dump([result, best_configuration_history], handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        print('\t', 'Best accuracy found:', max(result[len(result) - 1])[0], ' sensors num used:', max(result[len(result) - 1])[1])




