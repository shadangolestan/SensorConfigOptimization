# License: MIT
# This file is partially built on SMAC3(https://github.com/automl/SMAC3), which is licensed as follows,

# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)

# encoding=utf8

import abc
import logging
from typing import List, Union

import numpy as np
from scipy.stats import norm
import math

from openbox.utils.config_space import Configuration
from openbox.utils.config_space.util import convert_configurations_to_array
from openbox.surrogate.base.base_model import AbstractModel
from openbox.surrogate.base.gp import GaussianProcess
import Config as cf
import numpy as np
import ConfigSpace as CS
from scipy.optimize import linear_sum_assignment
import math

import time



class AbstractAcquisitionFunction(object, metaclass=abc.ABCMeta):
    """Abstract base class for acquisition function

    Attributes
    ----------
    model
    logger
    """
    def __str__(self):
        return type(self).__name__ + " (" + self.long_name + ")"

    def __init__(self, model: Union[AbstractModel, List[AbstractModel]], **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            Models the objective function.
        """
        self.lmda = 0
        self.model = model
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        
        
        
        if cf.acquisition_function == 'dg':
            self.info_map_plus = dict()
            for key in cf.info_matrix.keys():
                if '[' in key:
                    loc = eval(key)
                    self.info_map_plus[str([loc[0] - 1, loc[1] - 1])] = [cf.info_matrix[key] / 100]       
                else:
                    self.info_map_plus[key] = [cf.info_matrix[key] / 100]

            self.present_configurations_map = dict()
            self.absent_configurations_map = dict()
            self.all_configurations = []
            self.info_map_minus = dict()
            self.info_map_contributions = dict()
            self.limit1 = -1
            self.limit2 = -1
            self.flag = dict()
            self.log = {}
            self.expected_contribution = {}
            self.athar = {}

    def update(self, **kwargs):
        """Update the acquisition functions values.

        This method will be called if the surrogate is updated. E.g.
        entropy search uses it to update its approximation of P(x=x_min),
        EI uses it to update the current optimizer.

        The default implementation takes all keyword arguments and sets the
        respective attributes for the acquisition function object.

        Parameters
        ----------
        kwargs
        """
        
        self.lmda += 1
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __call__(self, configurations: Union[List[Configuration], np.ndarray], convert=True, **kwargs):
        """Computes the acquisition value for a given X

        Parameters
        ----------
        configurations : list
            The configurations where the acquisition function
            should be evaluated.
        convert : bool

        Returns
        -------
        np.ndarray(N, 1)
            acquisition values for X
        """

        import numpy as np
        if self.long_name == 'Distribution Guided':
            total_costs = []
            
            def dictionary_to_matrix(dictionary):
                max_row = int(np.ceil(cf.space[2][0]) / cf.pivots_granularity)
                max_col = int(np.ceil(cf.space[2][1]) / cf.pivots_granularity)
                
                
                matrix = [[-1] * (max_row) for _ in range(max_col)]
                
                
                for key, value in dictionary.items():
                    col, row = eval(key)
                    matrix[int(row / cf.pivots_granularity) - 1][int(col / cf.pivots_granularity) - 1] = np.mean(value)

                return matrix
                
            def calculate_g_star():
                Is = []
                for location in cf.configuration_star:
                    if location in self.expected_contribution.keys():
                        Is.append(self.expected_contribution[location])
                    else:
                        Is.append(cf.info_matrix[location])
                
                return Is
            
            
            self.S = []
            for index, c in enumerate(cf.config_advisor.history_container.configurations[len(self.all_configurations):], start=len(self.all_configurations)):
                c = c.get_dictionary()
                self.all_configurations.append(c)
                    
                for key in c.keys():
                    if key.startswith('ls') and not key.startswith('ls_t'):
                        if c[key] in self.info_map_plus.keys():
                            sensor_location = c[key]
                            sensor_initial = cf.info_matrix[sensor_location]
                            others_locations = [value for other_key, value in c.items() if other_key != key and other_key.startswith('ls') and not other_key.startswith('ls_t')]
                            others_initials = [cf.info_matrix[location] for location in others_locations]    
                            coeff = sensor_initial / (sum(others_initials) + sensor_initial)
                            inf = (100 - cf.config_advisor.history_container.perfs[index]) * coeff
                            # 3rd 4th and 5th tabs
                            # inf = (100 - cf.config_advisor.history_container.perfs[index]) * (sensor_initial / 100)
                            self.info_map_plus[sensor_location].append(inf)
                            # self.present_configurations_map[sensor_location].append(c)
                            

                        else:
                            sensor_location = c[key]
                            sensor_initial = cf.info_matrix[sensor_location]
                            others_locations = [value for other_key, value in c.items() if other_key != key and other_key.startswith('ls') and not other_key.startswith('ls_t')]
                            others_initials = [cf.info_matrix[location] for location in others_locations]    
                            coeff = sensor_initial / (sum(others_initials) + sensor_initial)
                            inf = (100 - cf.config_advisor.history_container.perfs[index]) * coeff
                            # 3rd 4th and 5th tabs
                            # inf = (100 - cf.config_advisor.history_container.perfs[index]) * (sensor_initial / 100)
                            self.info_map_plus.update({c[key]: [inf]})
                            # self.present_configurations_map.update({c[key]: [c]})
                
            for index, c in enumerate(configurations):
                seen_locations = []
                S_sensors = []
                c = c.get_dictionary()

                for key in c.keys():
                    if key.startswith('ls') and not key.startswith('ls_t'):
                        sensor_location = c[key]
                        
                        if sensor_location in seen_locations:
                            S_sensors.append(0)
                            continue
                            
                        seen_locations.append(sensor_location)
                        
                        if sensor_location in self.info_map_plus.keys():
                            if len(np.unique(self.info_map_plus[sensor_location])) > 1:
                                
                                normalized_info = self.info_map_plus[sensor_location]
                                
                                mu_plus = np.mean(normalized_info)
                                var_plus = np.var(normalized_info)      
                                std_plus = np.sqrt(var_plus)
                                Is = calculate_g_star()
                                G_star = np.mean(Is)


                                W = (mu_plus - G_star) / std_plus
                                sensor_contribution = (mu_plus - G_star) * norm.cdf(W) + std_plus * norm.pdf(W)

                                self.expected_contribution[sensor_location] = sensor_contribution
                                S_sensors.append(sensor_contribution)

                            else:
                                S_sensors.append(cf.info_matrix[sensor_location])

                        else:
                            S_sensors.append(cf.info_matrix[sensor_location])

                self.S.append(np.mean(S_sensors) / 100)
                
            self.S = np.array(self.S).reshape(-1, 1)
            

            
            if cf.testbed != 'aruba/':
                M = dictionary_to_matrix(self.expected_contribution)

                import os.path
                file_name = str(len(cf.config_advisor.history_container.configurations)) + '.png'

                if not os.path.isfile(file_name):
                    import numpy as np
                    import matplotlib.pyplot as plt

                    plt.imshow(M, cmap='hot', interpolation='nearest')
                    M = np.array(M)
                    for i in range(M.shape[0]):
                        for j in range(M.shape[1]):
                            plt.text(j, i, '{:.2f}'.format(M[i, j]), ha='center', va='center', color='blue')

                    plt.colorbar()
                    plt.savefig(file_name)
                    plt.clf()  
                    
            
            
            
        if convert:
            X = convert_configurations_to_array(configurations)
        else:
            X = configurations  # to be compatible with multi-objective acq to call single acq
        if len(X.shape) == 1:
            X = X[np.newaxis, :]


        acq = self._compute(X, **kwargs)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(np.float).max
        return acq

    @abc.abstractmethod
    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the acquisition value for a given point X. This function has
        to be overwritten in a derived class.

        Parameters
        ----------
        X : np.ndarray
            The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Acquisition function values wrt X
        """
        raise NotImplementedError()


class RLAF(AbstractAcquisitionFunction):
    r"""Computes for a given x the expected improvement as
    acquisition value.

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the incumbent.
    """

    def __init__(self,
                 model: AbstractModel,
                 par: float = 0.0,
                 num_fantasies: int = 64,
                 epsilon = 1,
                 error = 0.25,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        
        
        if num_fantasies is None:
                raise ValueError(
                    "Must specify `num_fantasies`."
                )
        
        super(RLAF, self).__init__(model)
        self.long_name = 'Reinforcement Learning-assisted Acquisition Function'
        self.par = par
        self.num_fantasies = num_fantasies
        self.eta = None
        self.epsilon = epsilon
        self.error = error
        self.iteration = 0
        self.rl_action = None

        
    def get_variance(self):
        return self.s
    
    def get_means(self):
        return self.m

    def get_incumbent_value(self):
        return self.eta

    def set_rl_action(self, action):
        self.rl_action = action


    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N, 1)
            Expected Improvement of X
        """

        

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        
        m, v = self.model.predict_marginalized_over_instances(X)
        self.s = np.sqrt(v)
        self.m = m
        
        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            # if not self.rl_action == None:
            #     z = (self.eta - m - self.rl_action) / self.s
            #     return (self.eta - m - self.rl_action) * norm.cdf(z) + self.s * norm.pdf(z)
            if not self.rl_action == None:
                z = (self.eta - m) / self.s
                
                PI = norm.cdf(z)
                EI =  (self.eta - m) * norm.cdf(z) + self.s * norm.pdf(z)
                
                result = self.rl_action * PI + (1 - self.rl_action) * EI
                
                
                
                # z = (self.rl_action * self.eta - m) / self.s
                # result  =  (self.rl_action * self.eta - m) * norm.cdf(z) + self.s * norm.pdf(z)
                
                return result
            
            else:
                z = (self.eta - m) / self.s
                return (self.eta - m) * norm.cdf(z) + self.s * norm.pdf(z)
                
            

        if np.any(self.s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            s_copy = np.copy(self.s)
            self.s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()
        # if (f < 0).any():
        #     raise ValueError(
        #         "Expected Improvement is smaller than 0 for at least one "
        #         "sample.")

        return f


class DG(AbstractAcquisitionFunction):
    r"""Computes for a given x the expected improvement as
    acquisition value.

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the incumbent.
    """

    def __init__(self,
                 model: AbstractModel,
                 par: float = 0.0,
                 num_fantasies: int = 64,
                 epsilon = 1,
                 error = 0.25,
                 **kwargs):
        
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        
        if num_fantasies is None:
                raise ValueError(
                    "Must specify `num_fantasies`."
                )
        
        super(DG, self).__init__(model)
        self.long_name = 'Distribution Guided'
        self.par = par
        self.num_fantasies = num_fantasies
        self.eta = None
        self.epsilon = epsilon
        self.error = error
        self.iteration = 0
        self.rl_action = None
        self.d = 0.5
        # Generate K random points within a hypercube of side length 2d centered at X
        self.K = 1000
        self.n_features = 10
        self.guide = 1

        # TODO:        
        # config = CS.Configuration(cf.config_space, values = cf.sorted_greedy_map.keys())
        

        # X = convert_configurations_to_array(cf.sorted_greedy_map)

    def get_variance(self):
        return self.s
    
    def get_means(self):
        return self.m

    def get_incumbent_value(self):
        return self.eta

    def set_rl_action(self, action):
        self.rl_action = action

    def sigma_neighbours(self, x):    
        import ast    
        num_sensors = int(len(x))
        x_prime = []

        for s in x:
            L = self.placeHolders[int(s)]            
            x_prime.append(L)

        # print('configuration is:', x_prime)

        def clamp(num, min_value, max_value):
           return max(min(num, max_value), min_value)

        import random
        import numpy as np
        neighbours = 100
        Ns = []
        tempNs = []
        for _ in range(neighbours):  
            N = []            
            tempN = []
            for s in range(num_sensors):
                N_x = clamp(x_prime[s][0] + (self.epsilon * random.randint(-1,1)), self.epsilon, 8 - self.epsilon)
                N_y = clamp(x_prime[s][1] + (self.epsilon * random.randint(-1,1)), self.epsilon, 8 - self.epsilon)

                tempN.append(list([N_x, N_y]))

                N.append(self.placeHolders.index(list([N_x, N_y])))
                
            
            Ns.append(N)
            tempNs.append(tempN)

        # print('neighbors are:', tempNs)
        return Ns
        
    def frange(self, start, stop, step):
        steps = []
        while start < stop:
            steps.append(start)
            start +=step
            
        return steps
    
    def find_neighbors(self, X, d, k, candidates):
        """
        Finds k-nearest neighbors of a query point X whose distances are less than or equal to d.
        
        Args:
            X (numpy.ndarray): Query point of shape (n_features,)
            d (float): Maximum distance threshold
            k (int): Number of nearest neighbors to return
            candidates (numpy.ndarray): Candidate neighbors of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Indices of the k-nearest neighbors whose distances are less than or equal to d
        """
        # Compute pairwise Euclidean distances between query point and candidate neighbors
        from scipy.spatial.distance import cdist

        # print('the query point is:', X)
        # print('The candidates are:', candidates)

        dists = cdist(X.reshape(1, -1), candidates, metric='euclidean')
        
        # print('d: ', d)
        # print('len dists are: ', len(dists))
        

        # Get indices of neighbors whose distances are less than or equal to d
        neighbors = np.where(dists <= d)[1]

        # print('len neighbors: ', len(neighbors))
        
        # Sort neighbors by distance and return the k-nearest neighbors
        
        neighbors_indices_sorted = neighbors[np.argsort(dists[0, neighbors])][:k]
        dist_of_neighbors = dists[0][neighbors_indices_sorted]
        weights = np.exp(-dist_of_neighbors / d)
        # weights /= np.sum(weights)  # Normalize weights to sum to 1

        return candidates[neighbors_indices_sorted], weights

    def get_neighbors_info(self, index):
        return self.neighbors_info[index]
    
    def plug_in_config(self, configurations: Union[List[Configuration], np.ndarray], convert=True, **kwargs):
        """Computes the DG acquisition value for a given X

        Parameters
        ----------
        configurations : list
            The configurations where the acquisition function
            should be evaluated.
        convert : bool

        Returns
        -------
        np.ndarray(N, 1)
            DG acquisition values for X
        """

        if convert:
            X = convert_configurations_to_array(configurations)
        else:
            X = configurations  # to be compatible with multi-objective acq to call single acq
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        

        acq = self._compute(X, **kwargs)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(np.float).max
        return acq

    def attention_function(self, time, cut_off = 100):        
        if time <= cut_off:
            return np.exp((time / cut_off) * np.log(2)) - 1  # Exponential growth phase
        else:
            return np.exp(-(time - cut_off) / 100)  # Exponential decay phase
    
    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the DG value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N, 1)
            Expected Improvement of X
        """
        
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        
        def greedy_map_analysis(X):
            import numpy as np
            m, v = self.model.predict_marginalized_over_instances(X)
            s = np.sqrt(v)
            
            E_I_plus = self.S
            EI = calculate_distance_alpha(m, s, E_I_plus)

            self.guide = self.attention_function(cf.iteration_id)
            self.guide = 0.0
            # third tab
            # alpha_c = self.guide * E_I_plus   +   (1 - self.guide) * EI
            
            # four-th tab
            # alpha_c = EI
            
            # fifth tab
            alpha_c = self.guide * E_I_plus   +   (1 - self.guide) * EI

            return alpha_c

        '''
        def gradient_analysis(x, n, w):
            m_ngbrs, v_ngbrs = self.model.predict_marginalized_over_instances(np.asarray(n))
            s_ngbrs = np.sqrt(v_ngbrs)
            neighbors_performance = w.reshape(-1, 1) * s_ngbrs

            m_c, v_c = self.model.predict_marginalized_over_instances(np.asarray([x]))
            s_c = np.sqrt(v_c)


            neighbors_performance = np.std(neighbors_performance)
            self.neighbors_info.append(neighbors_performance)


            if cf.gradient_fantacy:
                alpha_c = calculate_alpha(m_c, s_c) / self.total_costs

            else:
                alpha_c = calculate_alpha(m_c, s_c)

            return alpha_c
        '''
        
        def calculate_distance_alpha(MUs, STDs, E_I_plus):            
            m = MUs
            s = STDs
            z = (self.eta - m - self.par) / s
            # third tab
            # result = (self.eta - m - self.par)*norm.cdf(z) + s*norm.pdf(z)
            
            # four-th and fifth tab
            result = (self.eta - m - self.par)*E_I_plus* norm.cdf(z) + s *E_I_plus* norm.pdf(z)
            return result
        
        def calculate_alpha(MUs, STDs):
            m = MUs
            s = STDs
            z = (self.eta - m - self.par) / s
            result = (self.eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)
            return result

        # neighbors = []
        # weights = []
        
        # for x in X:
        #     self.n_features = x.shape[0]
        #     candidates = x.reshape(1, -1) + (np.random.rand(self.K, self.n_features) - 0.5) * self.d
        #     x_neighbors, x_weights = self.find_neighbors(x, self.d + 1, self.K, candidates)
        #     neighbors.append(x_neighbors)
        #     weights.append(x_weights)

        # self.neighbors_info = []
        # f = map(gradient_analysis, X, neighbors, weights)

        f = greedy_map_analysis(X)

        # f = list(f)

        if (np.asarray(f) < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one "
                "sample.")

        # f = np.asarray(f)

        return f
       
        
class EI(AbstractAcquisitionFunction):
    r"""Computes for a given x the expected improvement as
    acquisition value.

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the incumbent.
    """

    def __init__(self,
                 model: AbstractModel,
                 par: float = 0.0,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """

        super(EI, self).__init__(model)
        self.long_name = 'Expected Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N, 1)
            Expected Improvement of X
        """

        

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        
        m, v = self.model.predict_marginalized_over_instances(X)
        s = np.sqrt(v)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.eta - m - self.par) / s
            return (self.eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z)

        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()
        if (f < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one "
                "sample.")

        return f


class EIC(EI):
    r"""Computes for a given x the expected constrained improvement as
    acquisition value.

    :math:`\text{EIC}(X) := \text{EI}(X)\prod_{k=1}^K\text{Pr}(c_k(x) \leq 0 | \mathcal{D}_t)`,
    with :math:`c_k \leq 0,\ 1 \leq k \leq K` the constraints, :math:`\mathcal{D}_t` the previous observations.
    """

    def __init__(self,
                 model: AbstractModel,
                 constraint_models: List[GaussianProcess],
                 par: float = 0.0,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(EIC, self).__init__(model, par=par)
        self.constraint_models = constraint_models
        self.long_name = 'Expected Constrained Improvement'

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EIC value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N, 1)
            Expected Constrained Improvement of X
        """
        f = super()._compute(X)
        for model in self.constraint_models:
            m, v = model.predict_marginalized_over_instances(X)
            s = np.sqrt(v)
            f *= norm.cdf(-m / s)
        return f


class EIPS(EI):
    def __init__(self,
                 model: AbstractModel,
                 par: float = 0.0,
                 **kwargs):
        r"""Computes for a given x the expected improvement as
        acquisition value.
        :math:`EI(X) := \frac{\mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi\right] \} ]} {np.log(r(x))}`,
        with :math:`f(X^+)` as the incumbent and :math:`r(x)` as runtime.

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X) returning a tuples of
                   predicted cost and running time
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(EIPS, self).__init__(model, par=par)
        self.long_name = 'Expected Improvement per Second'

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EIPS value.

        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement per Second of X
        """
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self.model.predict_marginalized_over_instances(X)
        if m.shape[1] != 2:
            raise ValueError("m has wrong shape: %s != (-1, 2)" % str(m.shape))
        if v.shape[1] != 2:
            raise ValueError("v has wrong shape: %s != (-1, 2)" % str(v.shape))

        m_cost = m[:, 0]
        v_cost = v[:, 0]
        # The surrogate already predicts log(runtime)
        m_runtime = m[:, 1]
        s = np.sqrt(v_cost)

        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        def calculate_f():
            z = (self.eta - m_cost - self.par) / s
            f = (self.eta - m_cost - self.par) * norm.cdf(z) + s * norm.pdf(z)
            f = f / m_runtime
            return f

        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()

        if (f < 0).any():
            raise ValueError(
                "Expected Improvement per Second is smaller than 0 "
                "for at least one sample.")

        return f.reshape((-1, 1))


class LogEI(AbstractAcquisitionFunction):

    def __init__(self,
                 model: AbstractModel,
                 par: float = 0.0,
                 **kwargs):
        r"""Computes for a given x the logarithm expected improvement as
        acquisition value.

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(LogEI, self).__init__(model)
        self.long_name = 'Log Expected Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N, 1)
            Expected Improvement of X
        """
        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<int>) to inform the acquisition function '
                             'about the current best value.')

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)

        def calculate_log_ei():
            # we expect that f_min is in log-space
            f_min = self.eta - self.par
            v = (f_min - m) / std
            return (np.exp(f_min) * norm.cdf(v)) - \
                   (np.exp(0.5 * var_ + m) * norm.cdf(v - std))

        if np.any(std == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            self.logger.warning("Predicted std is 0.0 for at least one sample.")
            std_copy = np.copy(std)
            std[std_copy == 0.0] = 1.0
            log_ei = calculate_log_ei()
            log_ei[std_copy == 0.0] = 0.0
        else:
            log_ei = calculate_log_ei()

        if (log_ei < 0).any():
            raise ValueError(
                "Expected Improvement is smaller than 0 for at least one sample.")

        return log_ei.reshape((-1, 1))


class LPEI(EI):
    def __init__(self,
                 model: AbstractModel,
                 batch_configs=None,
                 par: float = 0.0,
                 estimate_L: float = 10.0,
                 **kwargs):
        r"""This is EI with local penalizer, BBO_LP. Computes for a given x the expected improvement as
        acquisition value.
        :math:`EI(X) := \frac{\mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi\right] \} ]} {np.log(r(x))}`,
        with :math:`f(X^+)` as the incumbent and :math:`r(x)` as runtime.

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X) returning a tuples of
                   predicted cost and running time
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(LPEI, self).__init__(model, par=par)
        self.estimate_L = estimate_L
        if batch_configs is None:
            batch_configs = []
        self.batch_configs = batch_configs
        self.long_name = 'Expected Improvement with Local Penalizer'

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the PenalizedEI value.

        Parameters
        ----------
        X: np.ndarray(N, D), The input point where the acquisition function
            should be evaluate. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N, 1)
            Expected Improvement per Second of X in log space
        """
        f = super()._compute(X)  # N*1
        f = np.log(np.log1p(np.exp(f)))  # apply g(z), soft-plus transformation
        for config in self.batch_configs:
            m, v = self.model.predict(config.get_array().reshape(1, -1))
            mu = m[0][0]
            var = v[0][0]
            sigma = math.sqrt(var)
            local_penalizer = np.apply_along_axis(self._local_penalizer, 1, X, config.get_array(),
                                                  mu, sigma, self.eta).reshape(-1, 1)
            f += local_penalizer
        return f

    def _local_penalizer(self, x, x_j, mu, sigma, Min):
        L = 5
        # L = self.estimate_L
        r_j = (mu - Min) / L
        s_j = sigma / L
        return norm.logcdf((np.linalg.norm(x - x_j) - r_j) / s_j)


class PI(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractModel,
                 par: float = 0.0,
                 **kwargs):

        """Computes the probability of improvement for a given x over the best so far value as
        acquisition value.

        :math:`P(f_{t+1}(\mathbf{X})\geq f(\mathbf{X^+})) :=
        \Phi(\frac{\mu(\mathbf{X}) - f(\mathbf{X^+})}{\sigma(\mathbf{X})})`,
        with :math:`f(X^+)` as the incumbent and :math:`\Phi` the cdf of the standard normal

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(PI, self).__init__(model)
        self.long_name = 'Probability of Improvement'
        self.par = par
        self.eta = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the PI value.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to evaluate PI. N is the number of points and D the dimension for the points

        Returns
        -------
        np.ndarray(N, 1)
            Expected Improvement of X
        """
        if self.eta is None:
            raise ValueError('No current best specified. Call update('
                             'eta=<float>) to inform the acquisition function '
                             'about the current best value.')

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)
        return norm.cdf((self.eta - m - self.par) / std)


class LCB(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractModel,
                 par: float = 1.0,
                 **kwargs):

        """Computes the lower confidence bound for a given x over the best so far value as
        acquisition value.

        :math:`LCB(X) = \mu(\mathbf{X}) - \sqrt(\beta_t)\sigma(\mathbf{X})`

        Returns -LCB(X) as the acquisition_function acq_maximizer maximizes the acquisition value.

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(LCB, self).__init__(model)
        self.long_name = 'Lower Confidence Bound'
        self.par = par
        self.eta = None  # to be compatible with the existing update calls in SMBO
        self.num_data = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the LCB value.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to evaluate LCB. N is the number of points and D the dimension for the points

        Returns
        -------
        np.ndarray(N, 1)
            (Negative) Lower Confidence Bound of X
        """
        if self.num_data is None:
            raise ValueError('No current number of Datapoints specified. Call update('
                             'num_data=<int>) to inform the acquisition function '
                             'about the number of datapoints.')
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)
        beta = 2 * np.log((X.shape[1] * self.num_data ** 2) / self.par)
        return -(m - np.sqrt(beta) * std)


class Uncertainty(AbstractAcquisitionFunction):
    def __init__(self,
                 model: AbstractModel,
                 par: float = 1.0,
                 **kwargs):

        """Computes half of the difference between upper and lower confidence bound (Uncertainty).

        :math:`LCB(X) = \mu(\mathbf{X}) - \sqrt(\beta_t)\sigma(\mathbf{X})`
        :math:`UCB(X) = \mu(\mathbf{X}) + \sqrt(\beta_t)\sigma(\mathbf{X})`
        :math:`Uncertainty(X) = \sqrt(\beta_t)\sigma(\mathbf{X})`

        Parameters
        ----------
        model : AbstractEPM
            A surrogate that implements at least
                 - predict_marginalized_over_instances(X)
        par : float, default=1.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        super(Uncertainty, self).__init__(model)
        self.long_name = 'Uncertainty'
        self.par = par
        self.eta = None  # to be compatible with the existing update calls in SMBO
        self.num_data = None

    def _compute(self, X: np.ndarray, **kwargs):
        """Computes the Uncertainty value.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to evaluate Uncertainty. N is the number of points and D the dimension for the points

        Returns
        -------
        tuple(np.ndarray(N, 1), np.ndarray(N, 1))
            Uncertainty of X
        """
        if self.num_data is None:
            raise ValueError('No current number of Datapoints specified. Call update('
                             'num_data=<int>) to inform the acquisition function '
                             'about the number of datapoints.')
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)
        beta = 2 * np.log((X.shape[1] * self.num_data ** 2) / self.par)
        uncertainty = np.sqrt(beta) * std
        if np.any(np.isnan(uncertainty)):
            self.logger.warning('Uncertainty has nan-value. Set to 0.')
            uncertainty[np.isnan(uncertainty)] = 0
        return uncertainty
