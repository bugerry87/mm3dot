"""
"""
# Installed
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial.transform import Rotation as R
from numba import jit
import numpy as np


DISTANCES = {}
ASSIGNMENTS = {}


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


def euclidean(a, b, **kargs):
	@jit
	def do_jit(a, b):
		N, M = len(a), len(b)
		d = np.empty((N,M))
		for n in range(N):
			d[n] = np.sum((a[n] - b)**2, axis=-1)
		return d
	return do_jit(a, b)


@jit
def yaw_to_vec(yaw):
	x = np.cos(yaw)
	y = np.sin(yaw)
	z = 0.0
	return x,y,z


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
	

def threshold(cost, th=0.1, **kargs):
	mask = cost <= th
	am = np.any(mask, axis=0)
	bm = np.any(mask, axis=1)
	return np.nonzero(mask), am, bm


def hungarian(cost, **kargs):
	indices = linear_assignment(cost)
	am = np.in1d(range(cost.shape[0]), indices[0])
	bm = np.in1d(range(cost.shape[1]), indices[1])
	return indices, am, bm


DISTANCES['mahalanobis'] = mahalanobis
ASSIGNMENTS['threshold'] = threshold
ASSIGNMENTS['hungarian'] = hungarian


# Test
if __name__ == '__main__':
	np.random.seed(0)
	a = np.random.rand(10,3)
	mask = euclidean(a, a) < 0.25
	print(mask)
	x, y = np.nonzero(mask)
	merge = np.zeros(a.shape)
	merge[x] += a[y]
	merge, n = np.unique(merge, return_counts=True, axis=0)
	merge /= n.reshape(-1,1)
	print(merge)