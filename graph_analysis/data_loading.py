"""
This file contains various tools for loading data useful to the project. The two major types of data loaded are COVID case data (formatted as https://github.com/nytimes/covid-19-data) and UberMedia close contact data, formatted as it was given to us by UberMedia.
"""


import numpy as np
import pandas as pd
import datetime
import os
from datetime import date, timedelta
import censusdata
from functools import reduce

from dask import delayed, compute
from dask.delayed import delayed
import dask.dataframe as dd
import dask.bag as db

os.chdir("/home/ec2-user/Contacts-sensitive/functions")
import aws_um_funcs as fcombo


state_dict = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

state_dict_inv = {v: k for k, v in state_dict.items()}

def str_date_to_tup(d):  # Assumes a string yyyymmdd, possibly with dashes, and returns (int(yyyy), int(mm), int(dd))
    d = d.replace('-', '')
    return (int(d[0:4]), int(d[4:6]), int(d[6:8]))

def tup_date_difference(first, second):  # Yields number of days from first date to second
    return round((datetime.datetime(*second) - datetime.datetime(*first)).days)

def time_average_cases(data, period):  # Returns an averaged case count, as opposed to the raw daily. Averages backwards in time, so we never know anything beyond the current index
    averaged = np.zeros(shape=data.shape)
    for i in range(averaged.shape[1]):
        scpdi = data[:, i]
        averaged[:, i] = np.convolve(np.pad(scpdi, (period - 1, 0), mode='edge'), np.ones((period,)) / period, mode='valid')
    return averaged



def get_cty_row(df):
    if 'fips' in df.columns:
        return (df['fips'] // 1) % 10 ** 3
    elif 'cbg' in df.columns:
        return (df['cbg'] // (10 ** 7)) % 10 ** 3
    else:
        raise KeyError

def add_cty_row(df):  # With the above, adds an integer county index row to the dataframe (derived from the CBG row)
    df['cty'] = get_cty_row(df)
    return df
    
def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def geo_interp_nans(arr):  # Interpolates NaN values with the geomean
    log_arr = np.empty(shape=arr.shape)
    log_arr[np.isnan(arr)] = np.nan
    log_arr[~np.isnan(arr)] = np.log(arr[~np.isnan(arr)])
    nans, nz = nan_helper(log_arr)
    log_arr[nans] = np.interp(nz(nans), nz(~nans), log_arr[~nans])
    return np.exp(log_arr)

def cty_case_data(state, start_date, end_date, time_average_period=1, interpolate_nans=True):
    """
    Returns a dataframe of per-county cases per day over the period [start_date, end_date) and a dictionary of counties to populations. Can time-average (again, backwards in time).
    """
    
    if len(state) > 2:
        state_abbr = state_dict[state]
    else:
        state_abbr = state
        state = state_dict_inv[state_abbr]
    all_case_data = pd.read_csv("~/us-counties.csv")
    current_state_case_data = all_case_data[all_case_data['state'] == state].copy()
    current_state_case_data = current_state_case_data[~current_state_case_data['fips'].isnull()]
    current_state_case_data = add_cty_row(current_state_case_data)
    
    """
    We start counting cases one day before start_date, so that we can take the consecutive day difference to get day-to-day cases
    """
    
    county_case_df = pd.DataFrame({int(x) : [np.nan] * (tup_date_difference(start_date, end_date)+1) for x in current_state_case_data['cty'].unique()})
    
    for idx, row in current_state_case_data.iterrows():
        row_cty = int(row['cty'])
        row_time = tup_date_difference(start_date, str_date_to_tup(row['date']))+1
        if row_time >= 0 and row_time < tup_date_difference(start_date, end_date)+1:
            county_case_df.at[row_time, row_cty] = row['cases']
            
    processed_cty_data = pd.DataFrame()
    for cty in county_case_df.columns:
        cty_data = county_case_df[cty].to_numpy()
        cty_data = cty_data[1:] - cty_data[:-1]
        if interpolate_nans:
            cty_data = geo_interp_nans(np.maximum(cty_data, 1))
        if time_average_period > 1:
            cty_data = time_average_cases(cty_data[:, np.newaxis], time_average_period)[:, 0]  # Goes backwards in time, so no future information gained at given index.
        processed_cty_data[cty] = cty_data
    
    current_state_fips = str(current_state_case_data['fips'].iloc[0] // 10 ** 3).split('.')[0]
    if len(current_state_fips) == 1:
        current_state_fips = '0' + current_state_fips
    COLUMN_ID = 'B01003_001E'  # Total population
    state_census_data = censusdata.download('acs5', 2015, censusdata.censusgeo([('state', current_state_fips), ('county', '*')]), [COLUMN_ID]) 
    state_census_data.columns = ['pop']
    state_census_data.index = [int(x.geo[1][1]) for x in state_census_data.index]
    
    pops = {}
    for idx, row in state_census_data.iterrows():
        try:
            pops[idx] = row['pop']
        except:
            continue
    return processed_cty_data, pops

def cty_case_rate_generator(state, start_date, end_date, time_average_period=1, interpolate_nans=True):  # Generator version of the above
    # Returns case frequencies (i.e. between 0 and 1) in the given locations
    county_case_df, pops = cty_case_data(state, start_date, end_date, time_average_period, interpolate_nans)
    for cty in county_case_df.columns:
        county_case_df[cty] /= pops[cty]
    
    for i in range(len(county_case_df)):
        yield county_case_df.loc[i]

        
def rename(f, name):
    f.__name__ = name
    return f


def generator_function_arg_setter(funct, args_dict, name=None):
    # Given a generator function, makes one with the arguments in **kwargs set
    def gf(*args, **kwargs):
        for k, v in args_dict.items():
            kwargs[k] = v
        return funct(*args, **kwargs)
    if name is not None:
        return rename(gf, name)
    else:
        return gf
        
        
def swap_str_1s_2s(s):
    """
    Swaps 1s and 2s in a string. Used in next function
    """
    return s.translate({ord('1'):'2', ord('2'):'1'})

def sum_dfs(dfs):  # Sums like indices. Useful for aggregating weights of contacts.
    return reduce(lambda a, b: a.add(b, fill_value=0), dfs)
    
def symmetrize_df(df):
    """
    For every contact, makes a contact with the device-ids switched. Storage-inefficient, but helpful in other ways
    """
    """
    Presumes that indexing is purely numeric (i.e. not by device_id). Will add a version of each contact in each direction.
    """
    return pd.concat([df, df.rename(columns={c: swap_str_1s_2s(c) for c in df.columns})])

def order_device_ids(df):
    """
    Returns df with same contacts as previous, but now device_id_1 is always lexicographically before device_id_2
    """
    device_id_ooo = df['device_id_2'] < df['device_id_1']
    df.loc[device_id_ooo, df.columns] = df.loc[device_id_ooo, [swap_str_1s_2s(c) for c in df.columns]].values
    return df
    
def date_iterator(start_date, end_date, spacing):
    # Provides an iterator for the given date range, with a sample rate given by spacing
    # Each of start_date, end_date should be integer tuples of the form (Y, M, D) or just date() objects.
    if type(start_date) is tuple:
        start_date = date(*start_date)
    elif isinstance(start_date, datetime.datetime):
        start_date = start_date.date()
    if type(end_date) is tuple:
        end_date = date(*end_date)
    elif isinstance(end_date, datetime.datetime):
        end_date = end_date.date()
    
    for n in range(int((end_date - start_date).days / spacing)):
        yield (start_date + timedelta(n * spacing)).strftime("%Y%m%d")

def load_date_interval(state, start_date, end_date, delay=False):  # Load data for a date interval as a single dataframe.
    if type(start_date) is str:
        start_date = (int(start_date[:4]), int(start_date[4:6]), int(start_date[6:]))
    if type(end_date) is str:
        end_date = (int(end_date[:4]), int(end_date[4:6]), int(end_date[6:]))
    di = date_iterator(start_date, end_date, 1)
    combined_data = []
    sql_exp = ("SELECT * FROM s3Object")
    while 1:
        try:
            cd = next(di)
        except StopIteration:
            break
        day_dfs = []
        BUCKET = 'yse-bioecon-um3-dc'
        state_day = 'um2-dc/third/state=' + (state_dict[state] if len(state) > 2 else state) + '/local_date=' + cd
        try:
            state_day_keys = fcombo.get_s3_keys(BUCKET, state_day)
        except:
            print("KEYGRAB FAILED FOR", cd)
            continue
        day_dfs = []
        for key in state_day_keys:
            df = dd.read_csv('s3://' + BUCKET + '/' + key, 
                         compression = 'gzip', 
                         sep = '\t', 
                         blocksize = None,
                         dtype = {'lat':'float64',
                                  'lon':'float64',
                                  'distance_m':'int64',
                                  'cbg':'int64',
                                  'device_id_1':'object',
                                  'unixtime_1':'int64',
                                  'datasource_id_1':'int64',
                                  'cel_distance_m_1':'float64',
                                  'local_time_1':'object',
                                  'speed_kph_1':'float64',
                                  'device_id_2':'object',
                                  'unixtime_2':'int64',
                                  'datasource_id_2':'int64',
                                  'cel_distance_m_2':'float64',
                                  'local_time_2':'object',
                                  'speed_kph_2':'float64'})
            day_dfs.append(df)
        day_df = dd.concat(day_dfs, axis=0, interleave_partitions=True)
        day_df = day_df.drop(['local_time_1', 'speed_kph_1', 'local_time_2', 'speed_kph_2'], axis=1)
        #filtered_df = delayed(clump_finding.find_clumps)(, drop_clump_contacts=True, return_clumps=False)  # We now have predeclumped data
        combined_data.append(day_df)
    if len(combined_data) == 0:
        return None
    
    if delay:
        return dd.concat(combined_data)
    else:
        return compute(dd.concat(combined_data))[0]
    
def left_looking_moving_average(arr, duration=1):
    """
    Assuming that the first dimension is time, gets the array containing moving average of the last duration points, or fewer if we are less than duration steps in
    """
    avg_arr = np.zeros(arr.shape)
    for i in range(duration):
        avg_arr[i:] += arr[:-i if i>0 else None]
    for i in range(1, duration):
        avg_arr[i-1] /= i
    avg_arr[duration-1:] /= duration
    return avg_arr
        
def left_looking_sample(arr, step=7, num_samples=4):
    """
    Returns an array of the same data, but with last dimension a timeshift on the data (i.e. instead of having an eigenvalue, we have a vector of the eigenvalue at that time step, STEP time before it, 2*STEP time before it, and so on). If a datapoint is unknown, returns the earliest datapoint known there.
    """
    samp_arr = np.empty(shape = arr.shape + (num_samples,))
    for i in range(num_samples):
        samp_arr[i * step:, ..., i] = arr[:-i * step if i > 0 else None]
    return samp_arr

def left_looking_multiaverage(arr, averaging_period=7, num_averaging_periods=1, replacement=None):
    """
    Given an initial array of shape (n, e) of e eigs over time, average back in time over the past averaging_period and concatenate the past num_averaging periods.
    Example: if num_eigs_returned=2, averaging_period=7, num_averaging_periods=4, then the output array is dimension (n, 8): we get the weeklong averages of the first two eigenvalues for each of the past four weeks.
    If replacement is 'lin', we replace by linear interpolation (or nearest on the edges)
    """
    data = flatten_all_but_first(left_looking_sample(left_looking_moving_average(arr, averaging_period), step=averaging_period, num_samples=num_averaging_periods))
    if replacement == 'lin':  # Not sure this works -- might have to do by row.
        for i in range(data.shape[1]):
            mask = np.isnan(data[:, i])
            data[mask, i] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask, i])
    return data
        
        

def flatten_all_but_first(arr):
    # Flattens all dimensions but the first. Useful for timeseries data
    return arr.reshape((arr.shape[0], np.prod(np.array(arr.shape[1:]))))


def load_cdc_csv(path):
    raw_case_data = pd.read_csv(path)
    raw_case_data['submission_date'] = pd.to_datetime(raw_case_data['submission_date'], format='%m/%d/%Y')
    return raw_case_data.sort_values(by=['state','submission_date']).groupby('state')['submission_date', 'new_case']

def days_before(pd_timestamp):  # Computes days since the start of 2020. Useful baseline. Could (and maybe should) be rewritten to use days since the epoch everywhere, but this is a little simpler for experimentation.
    return (pd_timestamp.date() - datetime.date(2020, 1, 1)).days

def state_case_data(all_state_case_data, state):
    current_state_case_data = all_state_case_data.get_group(state_dict[state])
    current_state_uniform_data = current_state_case_data.set_index('submission_date')[['new_case']]
    current_state_uniform_data = current_state_uniform_data.reindex(pd.date_range(start=current_state_case_data['submission_date'].min(), end=current_state_case_data['submission_date'].max(),freq='1D')).interpolate(method='linear')
    current_state_first_day_idx = days_before(current_state_case_data['submission_date'].iloc[0])
    current_state_cases = current_state_uniform_data.to_numpy()[:, 0]
    if current_state_first_day_idx > 0:
        current_state_cases = np.concatenate((np.zeros((current_state_first_day_idx, )), current_state_cases))
    return current_state_cases