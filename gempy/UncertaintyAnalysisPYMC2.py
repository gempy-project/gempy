import pymc

class Posterior:
    """Posterior database analysis for pymc2."""
    def __init__(self, db_name, verbose=False):
        # load db
        # TODO: Compatability with different database types?
        self.db = pymc.database.hdf5.load(db_name)
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