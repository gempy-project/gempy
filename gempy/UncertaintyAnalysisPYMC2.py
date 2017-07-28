import pymc
import theano
import numpy as np

class Posterior:
    """Posterior database analysis for pymc2."""
    def __init__(self, db_name, verbose=False):
        # load db
        self.db = pymc.database.hdf5.load(db_name)  # TODO: Compatibility with different database types?
        # get trace names
        self.trace_names = self.db.trace_names[0]
        # get gempy block models
        try:
            self.sols = self.db.gempy_model.gettrace()
        except AttributeError:
            print("No GemPy models tallied.")
            self.sols = None
        # load graphs
        self.graphs = self.db.gempy_topo.gettrace()[:, 0]
        # load centroids
        self.centroids = self.db.gempy_topo.gettrace()[:, 1]
        # load input data
        self.input_data = self.db.input_data.gettrace()

    def change_input_data(self, interp_data, i):
        interp_data.geo_data_res = self.input_data[i]
        interp_data.interpolator.tg.final_potential_field_at_formations = theano.shared(np.zeros(
            interp_data.interpolator.tg.n_formations_per_serie.get_value().sum(), dtype='float32'))
        interp_data.interpolator.tg.final_potential_field_at_faults = theano.shared(np.zeros(
            interp_data.interpolator.tg.n_formations_per_serie.get_value().sum(), dtype='float32'))
        interp_data.update_interpolator()