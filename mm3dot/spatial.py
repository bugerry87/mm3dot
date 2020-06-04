"""
"""
# Installed
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial.transform import Rotation as R
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


@jit
def yaw_to_vec(yaw):
	x = np.cos(yaw)
	y = np.sin(yaw)
	z = 0.0
	return x,y,z


@jit
def vec_to_yaw(x, y, z=0):
	pi = np.where(x > 0.0, np.pi, -np.pi)
	with np.errstate(divide='ignore', over='ignore'):
		yaw = np.arctan(x / y) + (y < 0) * pi
	return yaw


def quat_to_vec(x, y, z, w):
	r = R.from_quat((x, y, z, w))
	yaw, pitch, roll = r.as_euler('zyx')
	x = np.cos(yaw)
	y = np.sin(yaw)
	z = np.sin(pitch)
	return x,y,z
	

def greedy_threshold(cost, gth=0.1, **kargs):
	mask = cost <= gth
	am = np.any(mask, axis=0)
	bm = np.any(mask, axis=1)
	return np.nonzero(mask), am, bm


def hungarian(cost, **kargs):
	indices = linear_assignment(cost)
	am = np.in1d(range(cost.shape[0]), indices[0])
	bm = np.in1d(range(cost.shape[1]), indices[1])
	return indices, am, bm