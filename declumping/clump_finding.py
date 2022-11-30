import pandas as pd
import numpy as np
import boto3

class LatLonSquareSearcher:
    """
    Allows fast grabbing of all indices within lat/lon interval
    """
    def __init__(self, df):
        self.lats = df['lat'].to_numpy()  # Sort all latitudes, longitudes for fast interval grabbing
        self.lons = df['lon'].to_numpy()
        self.lats = np.stack([self.lats, np.arange(self.lats.shape[0])], axis=1)
        self.lons = np.stack([self.lons, np.arange(self.lons.shape[0])], axis=1)
        self.lats = self.lats[self.lats[:, 0].argsort()]
        self.lons = self.lons[self.lons[:, 0].argsort()]
        
    def point_indices(self, lat_interval, lon_interval):
        # Finds elements which are in the INCLUSIVE interval
        lat_interval[1] += 1e-10  # For inclusivity
        lon_interval[1] += 1e-10
        lat_bounds = np.searchsorted(self.lats[:, 0], lat_interval)
        lon_bounds = np.searchsorted(self.lons[:, 0], lon_interval)
        lat_idxs = self.lats[lat_bounds[0]:lat_bounds[1], 1]
        lon_idxs = self.lons[lon_bounds[0]:lon_bounds[1], 1]
        return np.intersect1d(lat_idxs, lon_idxs).astype(np.int)
        #return list(lat_idxs.intersection(lon_idxs))

        


def find_clumps(contacts, 
                clump_radius=3e-5, 
                min_clump_size=2000, 
                min_density_ratio=25, 
                NRM=3, 
                verbose=False, 
                add_is_clump_row=False, 
                drop_clump_contacts=False, 
                return_clumps=True):
    """
    Finds the coordinates of a set of clumps given a dataframe of contacts in a single day.
    Requires a dataframe of all relevant contacts, containing at least the lat and lon coordinates of those contacts, in columns 'lat' and 'lon' respectively.
    Returns a list of lat/lon pairs.
    
    A clump is defined to be more than min_clump_size coordinates whose lat, lon differ from a central point by no more than 2 * clump_radius.
    Additionally, we require that expanding the lat/lon square defining the clump by a factor of NRM sends the average density down by a factor of at least min_density_ratio
    """
    
    is_clump_contact_array = np.zeros((len(contacts.index),), dtype=np.int)
    contacts_lat_lon_indexed = contacts.groupby(['lat', 'lon']).size()
    searcher = LatLonSquareSearcher(contacts)
    clumps = []
    ct = 0
    min_coord_size_to_check = min_clump_size / ((clump_radius * 2 / 1e-5 + 1) ** 2) # The smallest number s.t. if a clump is of size >= min_clump_size
    # in the given radius, at least one lattice (assuming lattice spacing 1e-5) point in the clump will have this many contacts
    poss_loc_index = contacts_lat_lon_indexed.index[contacts_lat_lon_indexed > 200]
    if verbose:
        print("Total locs to check:", poss_loc_index.size)
    for lat, lon in poss_loc_index:
        ct += 1
        if verbose and ct % 100 == 0:
            print("Checked", ct)
        taxi_dist_to_prev = [np.abs(lat - c[0]) + np.abs(lon - c[1]) for c in clumps] + [1e8]
        if min(taxi_dist_to_prev) < 2 * clump_radius:  # Screen duplicates
            continue
        clump_point_indices = searcher.point_indices([lat - clump_radius, lat + clump_radius], [lon - clump_radius, lon + clump_radius])
        clump_size = clump_point_indices.shape[0]
        if clump_size < min_clump_size:
            continue
        neighborhood_size = searcher.point_indices([lat - clump_radius * NRM, lat + clump_radius * NRM], [lon - clump_radius * NRM, lon + clump_radius * NRM]).shape[0] - clump_size
        if clump_size / (max(neighborhood_size, 1) / (NRM ** 2 - 1)) > min_density_ratio:
            is_clump_contact_array[[int(i) for i in clump_point_indices]] = 1
            clump_center = contacts[['lat', 'lon']].iloc[clump_point_indices].mean(axis=0).tolist() # Recalculate actual center -- previous coordinates were approximate
            clump_center = np.round(np.array(clump_center), 5).tolist()
            clumps += [clump_center]
    
    if add_is_clump_row:
        contacts = contacts.copy()  # To make dask happy so we don't in-place modify inputs
        contacts['is_clump_contact'] = is_clump_contact_array.tolist()
    if drop_clump_contacts:
        contacts = contacts[[(i == 0) for i in is_clump_contact_array.tolist()]]
        
    if verbose:
        tc = len(contacts.index)
        tcc = np.sum(is_clump_contact_array)
        print("Total contacts:", tc)
        print("Total clump contacts:", tcc)
        print("Fraction of total:", tcc/tc)
        
    if return_clumps:
        return contacts, clumps
    else:
        return contacts
    
    # CHANGE THE RETURN DF, DON'T IN-PLACE MOD IT