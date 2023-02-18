import numpy as np
from pybads.poll import poll_mads_2n

def poll_mads_2n_ones_test():
    poll_scale = np.ones((1, 3))
    search_mesh_size = 9.7656e-4
    mesh_size = 1.0
    D = 3
    B = poll_mads_2n(D, poll_scale, search_mesh_size, mesh_size)
    assert B.shape[0] == 6 and B.shape[1] == D

def poll_mads_2n_test():
    poll_scale = np.array([[0.5133, 0.493, 3.9511]])
    search_mesh_size = 9.7656e-4
    mesh_size = 0.0312
    D = 3
    B = poll_mads_2n(D, poll_scale, search_mesh_size, mesh_size)
    assert B.shape[0] == 6 and B.shape[1] == D