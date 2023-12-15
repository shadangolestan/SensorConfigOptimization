# Introduction
This repository is the implementation of our paper published at AAAI 2024 titled "Grey-box Bayesian Optimization for Sensor Placement in Assisted Living Environments".

Preprint version of the paper can be found at Arxiv: https://arxiv.org/abs/2309.05784

Our algorithm, Distribution-Guided Bayesian Optimization (DGBO), is a sample-efficient algorithm that learns domain-specific knowledge about the objective function and integrates it into the iterative selection of query points in Bayesian optimization.

DGBO is applied for sensor placement in two simulation and one real-world testbed environments. The sensor placement is optimized in terms of maximizing the F1-score of an activity recognition. The code uses the package SensorOptimizers.BayesianOptimization which uses OpenBox.

Please note that, for more convenient reproducibility, we executed the SIM_SIS simulator and acquired and stored the results within this repository. In this context, we employ its sensor simulator in the optimization process to automatically produce synthetic sensor readings for simulation testbeds, given a candidate sensor placement. Therefore, there is no need to execute SIM_SIS.

# Requirements Installation
```
pip install -r /SensorConfigurationOptimization/requirements.txt
```

# Usage
Run the code with the following command:
```
python BO.py testbed=<TestbedName> acq_func=<AcquisitionFunction> sensor#=<NumberOfSensors> epsilon=<EpsilonValue> i=<IterationNumber>
```

provide the required arguments:

testbed: Choose from Testbed1, Testbed2, or aruba (real-world dataset).
acq_func: Choose ei (expected improvement) or dg (your method).
sensor#: Number of sensors for optimization.
epsilon: Choose from 0.25, 0.5, or 1 (not effective for aruba).
i: run number.

The code will execute the optimization process and save the results in the `Results_BO` directory.

# Baseline Usage
This work is compared with conventional methods in the literature, i.e., Genetic and Greedy Algorithms.

## Genetic Algorithm
The Genetic Algorithm uses the SensorOptimizers.GeneticAlgorithm module to run a genetic algorithm for sensor optimization. Simply run:

```
python GA.py testbed=<TestbedName> i=<RunNumber>
```
Provide the required arguments:
testbed: Choose from Testbed1, Testbed2, or aruba (real-world dataset).
i: run number.

The code will execute the Genetic Algorithm for sensor optimization and save the results in the GA_results directory.

### Reference: 
Brian L Thomas, Aaron S Crandall, and Diane J Cook. A genetic algorithm approach to motion sensor placement in smart environments. Journal of reliable intelligent environments, 2(1):3–16, 2016

## Greedy Algorithm
The Greedy Algorithm uses the SensorOptimizers.Greedy module to run a greedy algorithm for sensor optimization. Simply run:

```
python3 Greedy.py testbed=<TestbedName> sensor#=<NumberOfSensors>
```
provide the required arguments:

testbed: Choose from Testbed1, Testbed2, or aruba (real-world dataset).
sensor#: Number of sensors for optimization.

### Reference:
Andreas Krause, Jure Leskovec, Carlos Guestrin, Jeanne VanBriesen, and Christos Faloutsos. Efficient sensor placement optimization for securing large water distribution networks. Journal of Water Resources Planning and Management, 134(6):516–526, 2008.

# Packages Used:

1. **Activity Recognition**: Center of advanced studies in adaptive systems (casas) at washington state university. http://casas.wsu.edu/
2. **Intelligent Indoor Environment Simulation**: Shadan Golestan, Ioanis Nikolaidis, and Eleni Stroulia. Towards a simulation framework for smart indoor spaces. Sensors, 20(24):7137, 2020. https://github.com/shadangolestan/SIM_SIS-Simulator


