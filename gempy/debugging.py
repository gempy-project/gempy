# These two lines are necessary only if gempy is not installed
import sys, os
sys.path.append("../")

# Importing gempy
import gempy as gp

# Embedding matplotlib figures into the notebooks
geo_data = gp.read_pickle('../Tutorial/geo_data.pickle')
geo_data.n_faults = 0
print(geo_data)
# Embedding matplotlib figures into the notebooks

gp.visualize(geo_data)
# Aux imports
