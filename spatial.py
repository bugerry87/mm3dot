
# Installed
from scipy.optimize import linear_sum_assignment as linear_assignment
from numba import jit
import numpy as np

@jit
def S_cov(H, P, R):
	return np.linalg.inv(H @ P @ H.T + R)


def mahalanobis(a, b, S, **kargs):
	@jit
	def do_jit(a, b, S):
		N, M = len(a), len(b)
		d = np.empty((N,M))
		for n in range(N):
			for m in range(M):
				c = a[n] - b[m]
				d[n,m] = c @ S[n] @ c.T
		return d
	return do_jit(a, b, S)
	

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