""" NASBench 101 Search Space """
import numpy as np

from nasbench import api
from search.search_spaces import SearchSpace


class NASBench101(SearchSpace):
    """
        NASBench101 API need to be first installed following the official instructions from
        https://github.com/google-research/nasbench
    """
    def __init__(self,
                 nasbench_api: api,  # NASbench101 API
                 fidelity=108,  # options = [4, 12, 36, 108], i.e. acc measured at different epochs
                 feature_encoding='one-hot', **kwargs
                 ):
        super().__init__(**kwargs)
        self.engine = nasbench_api
        self.num_vertices = 7
        self.max_edges = 9
        self.edge_spots = int(self.num_vertices * (self.num_vertices - 1) / 2)  # Upper triangular matrix
        self.op_spots = int(self.num_vertices - 2)  # Input/output vertices are fixed
        self.allowed_ops = ['conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3']
        self.allowed_edges = [0, 1]  # Binary adjacency matrix
        self.fidelity = fidelity
        self.feature_encoding = feature_encoding

        # upper and lower bound on the decision variables
        self.n_var = int(self.edge_spots + self.op_spots)
        self.lb = [0] * self.n_var
        self.ub = [1] * self.n_var
        self.ub[-self.op_spots:] = 2

        # create the categories for each variable
        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

    @property
    def name(self):
        return 'nasbench101'

    def _sample(self, **kwargs):
        pass


    def _encode(self, subnet):
        pass

    def _decode(self, x):
        pass

    def _features(self, X):
        pass


if __name__ == '__main__':
    nasbench101 = api.NASBench('data/meta_data/nasbench_full.tfrecord')

