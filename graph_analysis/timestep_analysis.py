import numpy as np
import pandas as pd
import datetime
import data_loading
import os

import dask
from dask.delayed import delayed


    
def rolling_day_data(state, 
                     start_date, 
                     end_date,
                     external_data_generator_functions,
                     per_day_processing_functions,
                     final_processing_function,
                     fpref=None, 
                     num_parallel=1,
                     history_length=1,
                     final_delayed=True):
    """
    The idea of this is that it should provide a wrapper to easily do *all* of the things that were being done with similar functions in radius_experiments.ipynb. 
    In particular, this lets us compute a per-day function which may call on data from the several days prior to the given date. We can then load data serially, doing each day as we load the data needed for it and unloading data once no longer needed.
    start/end_date are the start and end dates of the prediction interval. Note that we may attempt to pull data from before start_date if history_length>1
    external_data_generators are generator functions that, when called under arguments (state, start_date, end_date) will return generators that will yield days of data in order starting from start_date and going at least to end_date
    per_day_processing functions are functions that will be computed for each day of data individually. They should be designed to take **kwargs as argument, 
        and will be given the return of each external data generator as the argument corresponding to that function name
    final_processing_function is a function that will produce the output for the day. It should take 2 list-of-lists arguments:
        one where each list is the time-series output of the per_day processing functions and the other where each list is the pe
    They will be either the outputs of the relevant per_day function if the data for that day exists, or None if it does not.
    It will contain fewer than that many if a day's worth of data is missing.
    While per_day functions do not need to handle Nones as arguments, final_processing_function does.
    
    We require that final_processing_function have return equal to either a scalar or a nested list/tuple of scalars.
    """
    
    
    if fpref is None:
        fpref = final_processing_function.__name__
    fname = '/home/ec2-user/Contacts-sensitive/' + fpref + '__' + state + '_' + '-'.join([str(i) for i in start_date]) + '_' + '-'.join([str(i) for i in end_date]) + '_' + str(history_length)
    if not fname.endswith('.txt'):
        fname += '.txt'
    if os.path.exists(fname):
        days_already_done = sum(1 for line in open(fname))
        print("Found", days_already_done, "days of already computed data!")
    else:
        print("No preexisting computed data found!")
        with open(fname, "w") as f:
            pass
        days_already_done = 0
    
    dates = []
    end_date = tuple((datetime.datetime(*end_date) + datetime.timedelta(days=1)).timetuple())[:3]
    di = data_loading.date_iterator(datetime.datetime(*start_date) - datetime.timedelta(days=history_length - 1), end_date, spacing=1)
    while True:
        try:
            dates += [next(di)]
        except StopIteration:
            break
        temp_dates = list(dates)  # Deepcopy
    temp_dates = temp_dates[days_already_done:]
    
    dfs = []
    per_day_processing_results = [[] for _ in per_day_processing_functions]
    
    external_data_generators = [edgf(state=state, start_date=data_loading.str_date_to_tup(temp_dates[0]), end_date=end_date) for edgf in external_data_generator_functions]
    
    print("Starting processing loop...")
    while len(temp_dates) > 1:
        while len(dfs) < num_parallel + history_length - 1 and len(temp_dates) > 1:
            dfs += [data_loading.load_date_interval(state, temp_dates[0], temp_dates[1], delay=True)]  # There is an view-setting issue here?
            external_day_data = {external_data_generator_functions[i].__name__: next(external_data_generators[i]) for i in range(len(external_data_generators))}
            for i, f in enumerate(per_day_processing_functions):
                per_day_processing_results[i] += [None] if dfs[-1] is None else [delayed(f)(dfs[-1], **external_day_data)]
            temp_dates = temp_dates[min(len(temp_dates), 1):]
        # At this point, we have the appropriate dfs delay-loaded in chronological order starting from the first day of prediction in this batch minus (num_parallel-1) and going through the last day of prediction
        # Moreover, we have their per-day processing results delay-loaded
        # We now just need to do the computation
        if not final_delayed:
            per_day_processing_results = dask.compute(per_day_processing_results)[0]
            print("Starting final")
            day_results = [final_processing_function([x[i:i+history_length] for x in per_day_processing_results]) for i in range(len(dfs)-(history_length-1))]
        else:
            day_results = [delayed(final_processing_function)([x[i:i+history_length] for x in per_day_processing_results]) for i in range(len(dfs)-(history_length-1))]
            print("Pre-compute")
            day_results = dask.compute(day_results)[0]
            print("Post-compute")
        with open(fname, "a") as f:
            for r in day_results:
                f.write(str(r).replace('\n', ' ') + '\n')
        to_cut = len(dfs) - (history_length - 1)  # This should always be the same, but redundancy doesn't hurt
        dfs = dfs[to_cut:]
        per_day_processing_results = [x[to_cut:] for x in per_day_processing_results]
        if len(temp_dates) > 0:
            print("Finished computes up to", temp_dates[0])
        
   
def sum_contacts_by_pairs(df, weighting='per_contact', **kwargs):
    # Can take kwargs for compatibility with the above
    summation_df = pd.DataFrame()
    summation_df[['device_id_1', 'device_id_2']] = df[['device_id_1', 'device_id_2']].values
    if weighting == 'per_contact':
        summation_df['weight'] = 1.0
    else:
        print("Weighting not recognized!")
        return
    return data_loading.order_device_ids(summation_df).groupby(['device_id_1', 'device_id_2']).sum().reset_index()

import re

def read_arbitrary_output(fname):  # Helper function for reading arbitrary time-series output of the form produced by rolling_day_data
    with open(fname, 'r') as f:
        data = f.read().split('\n')[:-1]
    for i in range(len(data)):
        data[i] = re.sub(' +', ' ', data[i])
        data[i] = data[i].replace('array', 'np.array')
        if ',' not in data[i]:
            data[i] = data[i].replace(' ', ',')
        data[i] = eval(data[i])
    return data

def right_shift_data(arr, n):  # Performs right-shift by n along the first axis, returning an array of same dimensions. Adds NaNs to the now-undefined indices.
    if n == 0:
        return arr
    elif n < 0:
        return np.concatenate([arr[-n:], np.full((-n,) + arr.shape[1:], np.nan)], axis=-1)
    else:
        return np.concatenate([np.full((n,) + arr.shape[1:], np.nan), arr[:-n]], axis=-1)
    
def concatenate_right_offsets(arr, offset_list):  # Given a 2D data array, where the first dimension is time and the second is data, return a data array whose data at time t is the data from the original array at each time t+t_off for t_off in offset_list
    return np.concatenate([right_shift_data(arr, n) for n in offset_list], axis=1)



    

    