# Introduction
This framework performs Bayesian optimization for sensor configuration in a testbed environment. The sensor configuration is optimized in terms of maximizing the accuracy of an activity recognition. The code uses the package SensorOptimizers.BayesianOptimization which uses OpenBox.

# Requirements Installation
```
pip install -r /SensorConfigurationOptimization/requirements.txt
```

# Inputs
1. **Config.space**: the space in which to optimize the sensor configuration.
2. **Config.epsilon**: the minimum manhattan distance between two sensors.
3. **Config.LSsensorsNum**: the number of Location Sensitive sensors (e.g., motion sensors) to be placed in the space.
4. **Config.ISsensorsNum**: the number of Interaction Sensitive sensors (e.g., electricity sensors) to be placed in the initial state of the space.
5. **Config.initial_state**: the initial state of the sensor configuration.
6. **Config.testbed**: the testbed environment in which the sensor configuration will be used.
7. **Config.bo_iteration**: the number of iterations for the Bayesian optimization algorithm.
8. **Config.ROS**: whether to use the ROS (Reagion Of Similarity) for the testbed environment.
9. **Config.input_sensor_types**: the types of sensors to be used in the sensor configuration.
10. **Config.acquisition_function**: the acquisition function to be used in the Bayesian optimization algorithm.
11. **Config.acq_optimizer_type**: the type of optimizer to be used for the acquisition function.

# Outputs
A pickle file, gets stored in Results_BO, containing the history of the sensor configuration optimization, including the sensor configuration and the corresponding activity recognition acciracy at each iteration.

# Usage
To run the code, simply execute BO.ipynb. Make sure that all the necessary input parameters are specified in the Config.py file. The pickle file with the history of sensor placement optimization will be saved in the Results_BO directory.

# Baseline Usage
This work is compared with conventional methods in the literature, i.e., Genetic and Greedy Algorithms.

## Genetic Algorithm
The Genetic Algorithm uses the SensorOptimizers.GeneticAlgorithm module to run a genetic algorithm for sensor optimization. The inputs are:

1. **Config.space**: the space in which to optimize the sensor configuration.
2. **Config.epsilon**: the minimum manhattan distance between two sensors.
3. **Config.initSensorNum**: the initial number of sensors to be placed in the space.
4. **Config.maxSensorNum**: the maximum number of sensors allowed to be placed in the space.
5. **Config.radius**: the radius of the sensors' sensing area.
6. **Config.mutation_rate**: the mutation_rate of the genetic algorithm.
7. **Config.crossover**: the number of splits in crossover function of genetic algorithm
8. **Config.survival_rate**: the survival_rate of the genetic algorithm.
9. **Config.survival_rate**: the reproduction_rate of the genetic algorithm.
10. **Config.ISsensorsNum**: the number of Interaction Sensitive sensors (e.g., electricity sensors) to be placed in the initial state of the space.
11. **Config.initial_state**: the initial state of the sensor configuration.
12. **Config.testbed**: the testbed environment in which the sensor configuration will be used.
13. **Config.ga_iteration**: the number of iterations for the genetic algorithm.
14. **Config.ROS**: whether to use the ROS (Reagion Of Similarity) for the testbed environment.
15. **Config.input_sensor_types**: the types of sensors to be used in the sensor configuration.

To run the genetic algoritm, run GA.ipynb. A pickle file, gets stored in GA_results, containing the history of the sensor configuration optimization, including the sensor configuration and the corresponding activity recognition acciracy at each iteration.

### Reference: 
Brian L Thomas, Aaron S Crandall, and Diane J Cook. A genetic algorithm approach to motion sensor placement in smart environments. Journal of reliable intelligent environments, 2(1):3–16, 2016

## Greedy Algorithm
The Greedy Algorithm uses the SensorOptimizers.Greedy module to run a greedy algorithm for sensor optimization. The inputs are:

1. **Config.space**: the space in which to optimize the sensor configuration.
2. **Config.epsilon**: the minimum manhattan distance between two sensors.
3. **Config.greedy_iteration**: the number of iterations for the greedy algorithm.
4. **Config.LSsensorsNum**: the number of Location Sensitive sensors (e.g., motion sensors) to be placed in the space.
5. **Config.ISsensorsNum**: the number of Interaction Sensitive sensors (e.g., electricity sensors) to be placed in the initial state of the space.
6. **Config.initial_state**: the initial state of the sensor configuration.
7. **Config.input_sensor_types**: the types of sensors to be used in the sensor configuration.

To run the greedy algoritm, run Greedy.ipynb. A pickle file, gets stored in results_SA, containing the history of the sensor configuration optimization, including the sensor configuration and the corresponding activity recognition acciracy at each iteration.

### Reference:
Andreas Krause, Jure Leskovec, Carlos Guestrin, Jeanne VanBriesen, and Christos Faloutsos. Efficient sensor placement optimization for securing large water distribution networks. Journal of Water Resources Planning and Management, 134(6):516–526, 2008.

# Packages Used:

1. **Activity Recognition**: Center of advanced studies in adaptive systems (casas) at washington state university. http://casas.wsu.edu/
2. **Intelligent Indoor Environment Simulation**: Shadan Golestan, Ioanis Nikolaidis, and Eleni Stroulia. Towards a simulation framework for smart indoor spaces. Sensors, 20(24):7137, 2020.


