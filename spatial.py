
# Installed
from scipy.optimize import linear_sum_assignment as linear_assignment
from numba import jit
import numpy as np

@jit
def inv_cov(H, P, R):
	return np.linalg.inv(H @ P @ H.T + R)


def mahalanobis(a, b, V, **kargs):
	#@jit
	def do_jit(a, b, V):
		N, M = len(a), len(b)
		A = np.repmat(a, M)
		c = np.empty((N,M))
		d = np.empty((N,M))
		for n in range(N):
			c[n] = a[n] - b[m]
			d[n] = c[n] @ V[n]
		return d * c.T
	return do_jit(a, b, V)
	

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