
# Installed
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import numpy as np

# Local
from .argoverse import ArgoDetectionLoader
from ..spatial import euclidean


class ArgoDetectionFilter(ArgoDetectionLoader):
	"""
	"""
	def __init__(self, dataroot,
		score_filter=0.0,
		off_ground_filter=0.0,
		merge=0.0,
		**kwargs
		):
		"""
		"""
		super().__init__(**kwargs)
		self.score_filter = score_filter
		self.off_ground_filter = off_ground_filter
		self.merge = merge
		self.map = ArgoverseMap()
		self.argo_loader = ArgoverseTrackingLoader(dataroot)
		pass
	
	def get_map_positions(self, frame):
		idx = self.argo_loader.get_idx_from_timestamp(frame.timestamp, frame.context)
		ego = self.argo_loader.get_pose(idx, frame.context)
		return ego.transform_point_cloud(frame.data[:,self.pos_idx])

	def filter(self, frame):
		"""
		"""
		map_pos = None
		def get_map_pos():
			if map_pos is None:
				return self.get_map_positions(frame)
			else:
				return map_pos
			pass
		
		if self.score_filter:
			mask = frame.data.T[self.score_idx] >= self.score_filter
			frame.data = frame.data[mask]
			frame.labels = frame.labels[mask]
			frame.uuids = frame.uuids[mask]
		
		if self.off_ground_filter:
			map_pos = get_map_pos()
			city = self.argo_loader.get(frame.context).city_name
			ground = self.map.get_ground_height_at_xy(map_pos, city)
			height = frame.data.T[self.shape_idx[-1]]
			off_ground = map_pos[:,-1] - ground + height * 0.5
			mask = off_ground >= self.off_ground_filter
			frame.data = frame.data[mask]
			frame.labels = frame.labels[mask]
			frame.uuids = frame.uuids[mask]
		
		if self.merge:
			m = len(frame)
			pos = frame.data[:,self.pos_idx]
			width = frame.data.T[self.shape_idx[1]]
			mask = euclidean(pos, pos) - (width * self.merge * 0.5)**2 <= 0
			x, y = np.nonzero(mask)
			merge = np.zeros(frame.data.shape)
			merge[x] += frame.data[y]
			merge, idx, n = np.unique(merge, return_counts=True, return_index=True, axis=0)
			frame.data = merge / n.reshape(-1,1)
			frame.labels = frame.labels[idx]
			frame.uuids = frame.uuids[idx]
			print('Merged:', m-len(frame))
			pass
		
		return frame
	
	def __getitem__(self, file):
		return self.filter(super().__getitem__(file))
	pass