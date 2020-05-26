
# Installed
from scipy.optimize import linear_sum_assignment as linear_assignment
from numba import jit
import numpy as np


def mahalanobis(a, b, Vi, **kargs):
	@jit
	def do_jit(a, b, Vi):
		N, M = a.shape[0], b.shape[0]
		c = np.empty((N,M))
		d = np.empty((N,M))
		for n in range(N):
			c[n] = a - b
			d[n] = c[n] * Vi[n]
		return d * c.T
	
	return do_jit(a, b, Vi)
	

def greedy_threshold(cost, gth=0.1, **kargs):
	mask = cost <= gth
	am = np.any(mask, axis=0)
	bm = np.any(mask, axis=1)
	return np.nonzero(mask), am, bm


def hungarian(cost, **kargs):
    indices = linear_assignment(cost)
    am = np.in1d(indices[0], range(cost.shape[0]))
    bm = np.in1d(indices[1], range(cost.shape[1]))
    return indices, am, bm