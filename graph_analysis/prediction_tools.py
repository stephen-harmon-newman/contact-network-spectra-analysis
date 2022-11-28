# A generalized tool for predicting disease spread based on linear regression on one or several types of input data.

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import copy

DEFAULT_DATA_NAME = "Data"





class CurveFit:
    def __init__(self, fit_x, fit_y):
        self.r2 = 0
        self._fit(fit_x, fit_y)
        self.mae = np.mean(np.abs(self.predict(fit_x) - fit_y))
        self.mse = np.mean(np.square(self.predict(fit_x) - fit_y))
        
    def _fit(self, fit_x, fit_y):
        raise NotImplementedError
    
    def predict(self, data):
        raise NotImplementedError
        
class LinearCurveFit(CurveFit):  # Just a linear regression
    def _fit(self, fit_x, fit_y):
        self.reg = LinearRegression().fit(fit_x, fit_y)
        self.r2 = self.reg.score(fit_x, fit_y)
    
    def predict(self, data):
        return self.reg.predict(data)
    
    
def exp_fit_fun(indata, *coeffs):
    augmented_indata = np.append(indata, np.ones((indata.shape[0], 1)), axis=1)
    return np.exp(np.dot(augmented_indata, np.array(coeffs))) # Coeffs needs to have length one more than the dimension of the input data. Note that we add a 1 to get the constant parameter.
    
def exp_fit_fun_jac(indata, *coeffs):
    augmented_indata = np.append(indata, np.ones((indata.shape[0], 1)), axis=1)
    return augmented_indata * np.exp(np.dot(augmented_indata, np.array(coeffs)))[:, np.newaxis]
    
class ExponentialCurveFit(CurveFit):
    def _fit(self, fit_x, fit_y):
        initial_guess = [0] * (fit_x.shape[1] + 1)
        self.true_params, _ = curve_fit(exp_fit_fun, fit_x, fit_y, initial_guess)
        #print(self.true_params)
        self.r2 = 1 - np.mean(np.square(self.predict(fit_x) - fit_y)) / np.mean(np.square(np.mean(fit_y) - fit_y))  # R^2 is dicey for nonlinear fits in general, but inasmuch as the dataset is quite large and it is an exponential fit, it's not horrible here -- we can reframe as a parametrically-weighted linear least-squares.
    def predict(self, data):
        return exp_fit_fun(data, *self.true_params)
    
"""
The theory in this function is that if the input data has a natural high-dimensional tensor ordering (e.g. it is indexed by time and eigenvalue index),
it should be the case that the coefficients are a rank-one tensor.
"""

class ExponentialTensorCurveFit(CurveFit):
    def __init__(self, fit_x, fit_y, tensor_shape):
        self.tensor_shape = tensor_shape
        
        def tensorized_exp_fit_fun(indata, *vector_coeffs):
            tensor_shape = self.tensor_shape  # By equality setting, we won't corrupt the original version of this.
            coeff_tensor = np.array(vector_coeffs[:tensor_shape[0]], dtype=float)
            vector_coeffs = vector_coeffs[tensor_shape[0]:]
            tensor_shape = tensor_shape[1:]
            while len(tensor_shape) > 0:
                coeff_tensor = coeff_tensor[..., np.newaxis] * np.array(vector_coeffs[:tensor_shape[0]], dtype=float)
                vector_coeffs = vector_coeffs[tensor_shape[0]:]
                tensor_shape = tensor_shape[1:]
            full_coeffs = list(coeff_tensor.flatten()) + [vector_coeffs[-1]]  # Vector_coeffs should be one long at this point.
            return exp_fit_fun(indata, *full_coeffs)
        
        self.teff = tensorized_exp_fit_fun
        
        super(ExponentialTensorCurveFit, self).__init__(fit_x, fit_y)
    
    def _fit(self, fit_x, fit_y):
        initial_guess = [.01] * (int(np.sum(self.tensor_shape)) + 1)  # This needs to be nonzero for this one.
        self.true_params, _ = curve_fit(self.teff, fit_x, fit_y, initial_guess)
        self.r2 = 1 - np.mean(np.square(self.predict(fit_x) - fit_y)) / np.mean(np.square(np.mean(fit_y) - fit_y))
    def predict(self, data):
        return self.teff(data, *self.true_params)

class Predictor:
    """
    Based on training data, compute the optimal offset. Verbose can be 0 (no printing), 1 (text printing), or 2 (text and graph printing)
    
    train_x_and_type is a dictionary of string_keyed tuples, the first element of which is a (l, ?) array of training data and the second of which is a string indicating model type. Alternately, it may be just such a tuple, in which case it will be converted to a one-element dictionary.
    
    The offset range is both left- and right-inclusive.
    We presume that case_counts and all data in train_x are time-aligned (i.e. start at the same time).
    If start_time and end_time are specified (as array indices), we will restrict the data to that range. This is left-inclusive, but not right inclusive.
    """
    def __init__(self, train_x_and_type, case_counts, offset_min=7, offset_max=100, start_time=None, end_time=None, verbose=0, graph_state_name='unknown'):
        if not isinstance(train_x_and_type, dict):
            train_x_and_type = {DEFAULT_DATA_NAME: train_x_and_type}
        
        
        self.keys = train_x_and_type.keys()
        
        train_x = {k:train_x_and_type[k][0] for k in self.keys}
        self.model_types = {k:train_x_and_type[k][1] for k in self.keys}
        
        train_x = copy.deepcopy(train_x)
        
        case_counts = case_counts[start_time:end_time].astype(np.float64)
        for k in self.keys:
            train_x[k] = train_x[k][start_time:end_time]
        
        self.optimal_offsets = {k:None for k in self.keys}
        self.optimal_models = {k:None for k in self.keys}
        self.r2_values = {k:[np.nan] * offset_min for k in self.keys}
        self.mses = {k:[np.nan] * offset_min for k in self.keys}
        self.maes = {k:[np.nan] * offset_min for k in self.keys}
        
        self.offsets = np.arange(offset_min, offset_max+1)
        
        for offset in self.offsets:
            case_count_ratios = np.minimum(case_counts[offset:]/case_counts[:-offset], 10)
            for k in self.keys:
                prediction_data_length = min(train_x[k].shape[0], case_count_ratios.shape[0])
                train_x_trunc = train_x[k][:prediction_data_length].astype(np.float64)
                case_count_ratios_trunc = case_count_ratios[:prediction_data_length]
                
                model_args = dict()
                if self.model_types[k] == 'exponential':
                    MODELTYPE = ExponentialCurveFit
                elif self.model_types[k].startswith('exponential-tensor'):  # This type wants to be passed with a colon between the type name and parameters. For instance, "exponential-tensor:4,6"
                    MODELTYPE = ExponentialTensorCurveFit
                    model_args['tensor_shape'] = [int(i) for i in self.model_types[k].split(':')[1].split(',')]
                else:
                    MODELTYPE = LinearCurveFit

                model = MODELTYPE(train_x_trunc, case_count_ratios_trunc, **model_args)
                
                if (self.optimal_offsets[k] is None) or (model.r2 > self.r2_values[k][self.optimal_offsets[k]]):
                    self.optimal_offsets[k] = offset
                    self.optimal_models[k] = model
                    
                self.r2_values[k].append(model.r2)
                self.mses[k].append(model.mse)
                self.maes[k].append(model.mae)
        
        for k in self.keys:
            self.r2_values[k] = np.array(self.r2_values[k])
            self.mses[k] = np.array(self.mses[k])
            self.maes[k] = np.array(self.maes[k])
                
        if verbose >= 1:
            for k in self.keys:
                print(k + ": Optimal R^2 " + str(self.r2_values[k][self.optimal_offsets[k]]) + " achieved at offset " + str(self.optimal_offsets[k]))
        
        NUM_PLOTS = 2
        if verbose >= 2:
            SIZE=12
            fig, axs = plt.subplots(NUM_PLOTS, 1, figsize=(SIZE, SIZE * 10/16 * NUM_PLOTS))
            axs[0].set_ylabel('R^2')
            axs[0].set_ylim(0, 1)
            axs[1].set_ylabel('MAE')
            axs[1].set_ylim(0, 1.1*np.max(self.maes[k][~np.isnan(self.maes[k])]))
            #axs[2].set_ylabel('MSE')
            #axs[2].set_ylim(0, np.max(self.mses[k][~np.isnan(self.mses[k])]))
            for i in range(NUM_PLOTS):
                axs[i].set_xlabel('Prediction Offset')
                
            cgen = iter(cm.rainbow(np.linspace(0, 1, len(self.keys))))
            for k in self.keys:
                plot_range = np.arange(0, offset_max+1)
                c = next(cgen)
                axs[0].plot(plot_range, self.r2_values[k], color=c, marker='.')
                axs[1].plot(plot_range, self.maes[k], color=c, marker='.')
                #axs[2].plot(plot_range, self.mses[k], color=c, marker='.')
                
                
            axs[0].set_ylabel('R^2')
            axs[1].set_ylabel('MAE')
            #axs[2].set_ylabel('MSE')
            for i in range(NUM_PLOTS):
                axs[i].set_xlabel('Prediction Offset')
                axs[i].legend(self.keys)
            
            plt.savefig("/home/ec2-user/Plots/stats_" + graph_state_name + ".eps", dpi=400, bbox_inches='tight')
            
    def predict(self, input_interval, data_name=None):  
        """
        Return predictions (case ratios) for input data for a specific model. 
        Can either pass an array of input data (i.e. of shape (k, ?)), in which case we will return an array of length k,
        or a single point of input data of shape (?), in which case we return a scalar
        """
        if data_name is None:
            data_name = DEFAULT_DATA_NAME
        
        as_array = (len(input_interval.shape) == 2)
        if not as_array:
            input_interval = input_interval[np.newaxis]
        output = self.optimal_models[data_name].predict(input_interval)
        if not as_array:
            output = output[0]
        return output
            
        

            
        
        
        
        