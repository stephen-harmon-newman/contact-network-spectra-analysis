# A generalized tool for predicting disease spread based on linear regression on one or several types of input data.

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import copy

from numba import jit

import data_loading

DEFAULT_DATA_NAME = "Data"



class CurveFit:
    def __init__(self, fit_x, fit_y):
        self.fit_x = fit_x
        self.fit_y = fit_y
        self.r2 = 0
        self._fit(fit_x, fit_y)
        self.mae = np.mean(np.abs(self.predict(fit_x) - fit_y))
        self.mse = np.mean(np.square(self.predict(fit_x) - fit_y))
        self.rss = np.mean(np.square(self.predict(fit_x) - fit_y))
    def _fit(self, fit_x, fit_y):
        raise NotImplementedError
    
    def predict(self, data):
        raise NotImplementedError
        
    def bootstrap_mse(self):
        sum_mse = 0
        sample_indices = range(0,self.fit_x.shape[0], 7)  # Once a week for speed.
        for i in sample_indices:  # This is mildly abusive -- we do the refitting in place. We reset after the loop.
            self._fit(np.concatenate((self.fit_x[:i], self.fit_x[i+1:]), axis=0), np.concatenate((self.fit_y[:i], self.fit_y[i+1:]), axis=0))
            sum_mse += np.square(self.predict(self.fit_x[i:i+1]) - self.fit_y[i:i+1])[0]
        self._fit(self.fit_x, self.fit_y)  # Reset the fit so that it's actually correct for the original input.
        return sum_mse / len(sample_indices)

    def bootstrap_msle(self):  # Mean-squared-logarithmic-error. Used for exponential models.
        sum_msle = 0
        sample_indices = range(0,self.fit_x.shape[0], 7)  # Once a week for speed.
        for i in sample_indices:  # This is mildly abusive -- we do the refitting in place. We reset after the loop.
            self._fit(np.concatenate((self.fit_x[:i], self.fit_x[i+1:]), axis=0), np.concatenate((self.fit_y[:i], self.fit_y[i+1:]), axis=0))
            sum_msle += np.square(np.log(self.predict(self.fit_x[i:i+1])) - np.log(self.fit_y[i:i+1]))[0]
        self._fit(self.fit_x, self.fit_y)  # Reset the fit so that it's actually correct for the original input.
        return sum_msle / len(sample_indices)
        
class LinearCurveFit(CurveFit):  # Just a linear regression
    def _fit(self, fit_x, fit_y):
        self.reg = LinearRegression().fit(fit_x, fit_y)
        self.r2 = self.reg.score(fit_x, fit_y)
        self.ar2 = 0 if fit_x.shape[0] <= fit_x.shape[1] - 1 else 1 - ((1-self.r2) * (fit_x.shape[0] - 1)) / (fit_x.shape[0] - fit_x.shape[1] - 1)
    def predict(self, data):
        return self.reg.predict(data)
    
    
def exp_fit_fun(indata, *coeffs):
    augmented_indata = np.append(indata, np.ones((indata.shape[0], 1)), axis=1)
    return np.exp(np.dot(augmented_indata, np.array(coeffs)))  # Coeffs needs to have length one more than the dimension of the input data. Note that we add a 1 to get the constant parameter.
    
def exp_fit_fun_jac(indata, *coeffs):
    augmented_indata = np.append(indata, np.ones((indata.shape[0], 1)), axis=1)
    return augmented_indata * np.exp(np.dot(augmented_indata, np.array(coeffs)))[:, np.newaxis]
    
class ExponentialCurveFit(CurveFit):
    def _fit(self, fit_x, fit_y):
        initial_guess = [0] * (fit_x.shape[1] + 1)
        self.true_params, _ = curve_fit(exp_fit_fun, fit_x, fit_y, initial_guess)
        self.r2 = 1 - np.mean(np.square(self.predict(fit_x) - fit_y)) / np.mean(np.square(np.mean(fit_y) - fit_y))  # R^2 is dicey for nonlinear fits in general, but inasmuch as the dataset is quite large and it is an exponential fit, it's not horrible here -- we can reframe as a parametrically-weighted linear least-squares.
        self.ar2 = 0 if fit_x.shape[0] <= fit_x.shape[1] - 1 else 1 - ((1-self.r2) * (fit_x.shape[0] - 1)) / (fit_x.shape[0] - fit_x.shape[1] - 1)
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
        self.ar2 = 0 if fit_x.shape[0] <= fit_x.shape[1] - 1 else 1 - ((1-self.r2) * (fit_x.shape[0] - 1)) / (fit_x.shape[0] - fit_x.shape[1] - 1)
    def predict(self, data):
        return self.teff(data, *self.true_params)

@jit(nopython=True)
def seir_predict_marginal_infections(time_off, *args):
    alpha, beta, gamma, ls_0, le_0, li_0 = args
    os = 10 ** ls_0
    oe = 10 ** le_0
    oi = 10 ** li_0
    for i in range(int(time_off)+1):
        ns = os - beta * os * oi
        ne = oe + beta * os * oi - alpha * oe
        ni = oi + alpha * oe - gamma * oi
        a = alpha * oe
        os, oe, oi = ns, ne, ni
    return a

seir_pmi_vectorized = np.vectorize(seir_predict_marginal_infections)


class LocalSEIRFit(CurveFit):
    def predict(self, data):
        pred = np.zeros(data.shape[0])
        k = data.shape[1]-1
        for i in range(data.shape[0]):
            past_days_marginal_cases = data[i, :-1][::-1]
            fit_params, _ = curve_fit(seir_pmi_vectorized, list(range(k)), past_days_marginal_cases, [.5, 1e-7, .1, 7, np.log10(past_days_marginal_cases[0]+1)+0.5, np.log10(past_days_marginal_cases[0]+1)+0.5], method='trf', bounds=([.2, 0, .05, 5, 0, 0], [.5, 1e-6, 1, 8, 6, 6]), max_nfev=50000)
            pred[i] = seir_predict_marginal_infections(data[i, -1] + k-1, *fit_params) / np.maximum(past_days_marginal_cases[-1], 1)  # Divide to get the case ratio, which is what we use for output in the Predictor class
        return pred
    
    def _fit(self, fit_x, fit_y):  # Since this is a local model, it doesn't actually fit.
        self.r2 = 1 - np.mean(np.square(self.predict(fit_x) - fit_y)) / np.mean(np.square(np.mean(fit_y) - fit_y))
        self.ar2 = 0
        return
        

class Predictor:
    """
    Based on training data, compute the optimal offset. Verbose can be 0 (no printing), 1 (text printing), or 2 (text and graph printing)
    
    train_x_and_type is a dictionary of string_keyed tuples, the first element of which is a (l, ?) array of training data and the second of which is a string indicating model type. Alternately, it may be just such a tuple, in which case it will be converted to a one-element dictionary.
    
    The offset range is both left- and right-inclusive.
    We presume that case_counts and all data in train_x are time-aligned (i.e. start at the same time).
    If start_time and end_time are specified (as array indices), we will restrict the *input* data to that range. This is left-inclusive, but not right inclusive.
    """
    def __init__(self, train_x_and_type, case_counts, offset_min=7, offset_max=100, start_time=None, end_time=None, verbose=0, graph_state_name='unknown', colors=None):
        if not isinstance(train_x_and_type, dict):
            train_x_and_type = {DEFAULT_DATA_NAME: train_x_and_type}
        
        
        self.keys = train_x_and_type.keys()
        
        train_x = {k:train_x_and_type[k][0] for k in self.keys}
        self.model_types = {k:train_x_and_type[k][1] for k in self.keys}
        
        train_x = copy.deepcopy(train_x)
        case_counts = case_counts.astype(np.float64)
        
        self.optimal_offsets = {k:None for k in self.keys}
        self.optimal_models = {k:None for k in self.keys}
        self.r2_values = {k:[np.nan] * offset_min for k in self.keys}
        self.ar2_values = {k:[np.nan] * offset_min for k in self.keys}
        self.mses = {k:[np.nan] * offset_min for k in self.keys}
        self.maes = {k:[np.nan] * offset_min for k in self.keys}
        
        self.offsets = np.arange(offset_min, offset_max+1)
        
        for offset in self.offsets:
            case_count_ratios = np.minimum(case_counts[offset:]/case_counts[:-offset if offset > 0 else None], 25.0)  # Hardcoded to prevent data anomalies with low cases.
            for k in self.keys:
                
                if self.model_types[k].startswith('Autoregression'):     
                    num_steps, step_spacing = [int(i) for i in self.model_types[k].split(':')[1].split(',')]
                    train_x[k] = data_loading.left_looking_multiaverage(np.concatenate((np.zeros((offset,)), case_count_ratios)), step_spacing, num_steps)  # We left-pad case count ratios. This will make autoregression act poorly if start_time is too small, since it won't actually know the x data it needs.
                prediction_data_length = end_time-start_time
                train_x_trunc = train_x[k][start_time:start_time+prediction_data_length]
                case_count_ratios_trunc = case_count_ratios[start_time:start_time+prediction_data_length]
                model_args = dict()
                if self.model_types[k] == 'exponential':
                    MODELTYPE = ExponentialCurveFit
                elif self.model_types[k].startswith('exponential-tensor'):  # This type wants to be passed with a colon between the type name and parameters. For instance, "exponential-tensor:4,6"
                    MODELTYPE = ExponentialTensorCurveFit
                    model_args['tensor_shape'] = [int(i) for i in self.model_types[k].split(':')[1].split(',')]
                elif self.model_types[k] == 'SEIR':
                    MODELTYPE = LocalSEIRFit
                elif self.model_types[k].startswith('Autoregression'):
                    MODELTYPE = LinearCurveFit
                else:
                    MODELTYPE = LinearCurveFit
                    
                model = MODELTYPE(train_x_trunc, case_count_ratios_trunc, **model_args)
                
                if (self.optimal_offsets[k] is None) or (model.r2 > self.r2_values[k][self.optimal_offsets[k]]):
                    self.optimal_offsets[k] = offset
                    self.optimal_models[k] = model
                    
                self.r2_values[k].append(model.r2)
                self.ar2_values[k].append(model.ar2)
                self.mses[k].append(model.mse)
                self.maes[k].append(model.mae)
        
        for k in self.keys:
            self.r2_values[k] = np.array(self.r2_values[k])
            self.ar2_values[k] = np.array(self.ar2_values[k])
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
                c = (next(cgen) if colors is None else colors[k])
                axs[0].plot(plot_range, self.r2_values[k], color=c, marker='.')
                axs[1].plot(plot_range, self.maes[k], color=c, marker='.')
                #axs[2].plot(plot_range, self.mses[k], color=c, marker='.')
                
            for i in range(NUM_PLOTS):
                axs[i].set_xlabel('Prediction Offset')
                axs[i].legend(self.keys)
            
            plt.savefig("/home/ec2-user/Plots/stats_" + graph_state_name + ".png", dpi=400, bbox_inches='tight')
            
    def predict(self, input_interval, return_ci=False, data_name=None):  
        """
        Return predictions (case ratios) for input data for a specific model. 
        Can either pass an array of input data (i.e. of shape (k, ?)), in which case we will return an array of length k,
        or a single point of input data of shape (?), in which case we return a scalar
        If return_ci is p>0, it will return p-confidence intervals given by MSE^(1/2)
        """
        if data_name is None:
            data_name = DEFAULT_DATA_NAME
        
        as_array = (len(input_interval.shape) == 2)
        if not as_array:
            input_interval = input_interval[np.newaxis]
        output = self.optimal_models[data_name].predict(input_interval)
        
        if not as_array:
            output = output[0]
        if return_ci:
            N_STD_DEV = 1.96
            if self.model_types[data_name] == 'exponential':
                log_stddev = (self.optimal_models[data_name].bootstrap_msle())**.5
                return (output, np.stack((output / np.exp(1.96 * log_stddev), output * np.exp(1.96 * log_stddev)), axis=-1))
            else:
                stddev = (self.optimal_models[data_name].bootstrap_mse())**.5
                return (output, np.stack((output - N_STD_DEV * stddev, output + N_STD_DEV * stddev), axis=-1))
        else:
            return output
            
        

            
        
        
        
        