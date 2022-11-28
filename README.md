# contact-network-spectra-analysis
Code ancillary to "Contact Network Spectra Predict Infectious Disease Spread."

This is a system for predicting disease spread based on eigenvalues of the graph Laplacian of a measured day-to-day close contact network, as described in the above paper. It is fast, relatively easy to use (feel free to contact the corresponding author with questions if you wish to implement this system for public health purposes!), and effective. With it, and given the contact networks used for initial generation, you can reproduce the entire result stack of the paper. Alternately, the intermediate calculated eigenvalues for contact networks in 2020 are included in this repository, so you may perform data analysis for yourself if you wish.

# Files in this repo

 - graph_analysis/radius_experiments.ipynb is used for generating the day-to-day eigenvalues and other statistics of contact graph data. This will need major modifications, as it depends on dataloading functions specific to our dataset and data storage setup. Their calls have been left in for context.
 - graph_analysis/spectral_result_analysis.ipynb is used for analysis of the results from graph_analysis and performing predictions. This will need much fewer modifications -- the only major ones should be modifying imports and input/output locations.
 - All other files in graph_analysis contain supporting functions for the above two notebooks.
 - eig_data contains the first several per-day eigenvalues for each state and day in 2020. Days for which we were missing data are omitted. Each list corresponds to a single day, listed in chronological order (see rolling_day_data in graph_analysis/timestep_analysis.py and the relevant call in graph_analysis/radius_experiments.ipynb for details on the generation).
 - summary_stats contains various summary statistics of the data. See the paper for a more thorough description. All are generated in graph_analysis/spectral_result_analysis.ipynb.