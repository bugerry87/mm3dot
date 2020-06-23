"""
"""
# Installed
from scipy.optimize import linear_sum_assignment as linear_assignment
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon
from numba import jit
import numpy as np


DISTANCES = {}
ASSIGNMENTS = {}


@jit
def S_cov(H, P, R):
	return np.linalg.inv(H @ P @ H.T + R)


def euclidean(trackers, frame, match_idx, **kargs):
	@jit
	def do_jit(a, b):
		N, M = len(a), len(b)
		d = np.empty((N,M))
		for n in range(N):
			d[n] = np.sum((a[n] - b)**2, axis=-1)
		return d
	
	if match_idx is None:
		M = frame.shape[-1]
		match_idx = slice(M)
	else:
		M = len(match_idx)
	
	N = len(trackers)
	a = np.empty((N,M))
	b = frame.data[:,match_idx]
	trk_idx = np.empty(N)
	
	for i, (idx, tracker) in enumerate(trackers.items()):
		a[i] = tracker.x[match_idx,].flatten()
		trk_idx[i] = idx
	
	return do_jit(a, b), trk_idx


def mahalanobis(trackers, frame, match_idx, **kargs):
	@jit
	def do_jit(a, b, S):
		N, M = len(a), len(b)
		d = np.empty((N,M))
		for n in range(N):
			for m in range(M):
				c = a[n] - b[m]
				d[n,m] = c @ S[n] @ c.T
		return d
	
	if match_idx is None:
		M = frame.shape[-1]
		match_idx = slice(M)
	else:
		M = len(match_idx)
	
	N = len(trackers)
	a = np.empty((N,M))
	b = frame.data[:,match_idx]
	S = np.empty((N,M,M))
	trk_idx = np.empty(N)
	
	for i, (idx, tracker) in enumerate(trackers.items()):
		a[i] = tracker.x[match_idx,].flatten()
		S[i] = spatial.S_cov(tracker.H, tracker.P, tracker.R)[match_idx, match_idx] #tracker.SI[match_idx, match_idx]
		trk_idx[i] = idx
	
	return do_jit(a, b, S), trk_idx


def bbox_to_poly_2d(bbox):
	if bbox.shape[-1] == 5:
		x, y, l, w, yaw = bbox.T
		r = R.from_euler('zyx', (yaw, 0, 0))
	elif bbox.shape[-1] == 6:
		x, y, l, w, xr, yr = bbox.T
		yaw = vec_to_yaw(xr, yr)
		r = R.from_euler('zyx', (yaw, 0, 0))
	elif bbox.shape[-1] == 8:
		x, y, l, w, xq, yq, zq, wq = bbox.T
		r = R.from_quat((xq, yq, zq, wq))
	else:
		raise ValueError("BBox with dimension {} not understood!".format(bbox.shape[-1]))
	
	l = l * 0.5
	w = w * 0.5
	corners = ((l, w, 0.0), (l, -w, 0.0), (-l, -w, 0.0), (-l, w, 0.0))
	corners = r.apply(corners)
	corners += (x, y, 0.0)
	return Polygon(corners[:,:2])


def iou_2d(trackers, frame, match_idx, **kargs):
	"""
	"""
	N = len(trackers)
	M = len(frame)
	iou = np.empty((N,M))
	trk_idx = np.empty(N)
	
	for n, (trk_id, tracker) in enumerate(trackers.items()):
		a = bbox_to_poly_2d(tracker.x[match_idx,].flatten())
		trk_idx[n] = trk_id
		for m, data in enumerate(frame.data):
			b = bbox_to_poly_2d(data[match_idx,])
			inter = a.intersection(b).area
			iou[n,m] = inter / (a.area + b.area - inter)
	return 1.0-iou, trk_idx


@jit
def yaw_to_vec(yaw):
	x = np.cos(yaw)
	y = np.sin(yaw)
	z = 0.0
	return x,y,z


def vec_to_yaw(x, y, z=0):
	pi = np.where(x > 0.0, np.pi, -np.pi)
	with np.errstate(divide='ignore', over='ignore'):
		if x == 0:
			a = np.inf
		else:
			a = y / x
		yaw = np.arctan(a) + (y < 0) * pi
	return yaw


def quat_to_vec(x, y, z, w, order='zyx'):
	r = R.from_quat((x, y, z, w))
	yaw, pitch, roll = r.as_euler(order)
	x = np.cos(yaw)
	y = np.sin(yaw)
	z = np.sin(pitch)
	return x,y,z


def hungarian_match(cost, **kargs):
	indices = linear_assignment(cost)
	am = np.in1d(range(cost.shape[0]), indices[0])
	bm = np.in1d(range(cost.shape[1]), indices[1])
	return indices, am, bm


def greedy_match(cost, **kargs):
	"""
	"""
	matched_indices = []
	N, M = cost.shape
	rank = np.argsort(cost.flatten())
	aidx, bidx = np.unravel_index(rank, cost.shape)
	am = np.zeros(N, dtype=bool)
	bm = np.zeros(M, dtype=bool)
	taken = np.zeros(N*M, dtype=bool)
	for i, (a, b) in enumerate(zip(aidx, bidx)):
		if np.all(am): #all trackers already taken?
			break
		elif np.all(bm): #all detections already taken?
			break
		elif ~am[a] and ~bm[b]:
			taken[i] = True
			am[a] = True #mark as taken
			bm[b] = True
	return (aidx[taken], bidx[taken]), am, bm


DISTANCES['mahalanobis'] = mahalanobis
DISTANCES['euclidean'] = euclidean
DISTANCES['iou_2d'] = iou_2d
ASSIGNMENTS['hungarian'] = hungarian_match
ASSIGNMENTS['greedy'] = greedy_match


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