

# Build In
import os
import json

# Installed
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import numpy as np

# Local
from .argoverse import ArgoDetectionLoader, ArgoRecorder, init_argoverse_arg_parser
from .. import spatial


def init_argoverse_filter_arg_parser(parents=[]):
	from argparse import ArgumentParser
	parser = ArgumentParser(
		parents=parents,
		description='Arguments for a Agroverse Filter lib',
		add_help=len(parents) == 0
		)
	parser.add_argument('--score_filter', type=float, metavar='FLOAT', default=0.3,
		help="Consider detections with scores above only.")
	parser.add_argument('--off_ground_filter', type=float, metavar='FLOAT', default=1.0,
		help="Consider detections on the ground only (fly-tolerance).")
	parser.add_argument('--merge', type=float, metavar='FLOAT', default=2.0,
		help="Merge and mean detections intersecting its width*x.")
	parser.add_argument('--city_space', type=bool, nargs='?', metavar='FLAG', default=False, const=True,
		help="Merge and mean detections intersecting its width*x.")
	return parser


class ArgoDetectionFilter(ArgoDetectionLoader):
	"""
	"""
	def __init__(self, dataroot,
		score_filter=0.0,
		off_ground_filter=0.0,
		merge=0.0,
		city_space=False,
		argo_loader=None,
		map=None,
		**kwargs
		):
		"""
		"""
		super().__init__(**kwargs)
		self.score_filter = score_filter
		self.off_ground_filter = off_ground_filter
		self.merge = merge
		self.city_space = city_space
		
		if isinstance(map, ArgoverseMap):
			self.map = map
		else:
			self.map = ArgoverseMap()
			
		if isinstance(argo_loader, ArgoverseTrackingLoader):
			self.argo_loader = argo_loader
		else:
			self.argo_loader = ArgoverseTrackingLoader(dataroot)
		pass
	
	def get_map_positions(self, frame):
		idx = self.argo_loader.get_idx_from_timestamp(frame.timestamp, frame.context)
		self.ego = self.argo_loader.get_pose(idx, frame.context)
		map_pos = self.ego.transform_point_cloud(frame.data[:,self.pos_idx])
		map_ori = self.ego.transform_point_cloud(frame.data[:,self.rot_idx] + frame.data[:,self.pos_idx]) - map_pos
		return map_pos, map_ori

	def filter(self, frame):
		"""
		"""
		map_pos = map_ori = None
		def get_map_pos():
			if map_pos is None:
				return self.get_map_positions(frame)
			else:
				return map_pos, map_ori
			pass
		
		if self.score_filter:
			mask = frame.data.T[self.score_idx] >= self.score_filter
			frame.data = frame.data[mask]
			frame.labels = frame.labels[mask]
			frame.uuids = frame.uuids[mask]
		
		if self.off_ground_filter:
			map_pos, map_ori = get_map_pos()
			city = self.argo_loader.get(frame.context).city_name
			ground = self.map.get_ground_height_at_xy(map_pos, city)
			height = frame.data.T[self.shape_idx[-1]]
			off_ground = map_pos[:,-1] - ground + height * 0.5
			mask = off_ground >= self.off_ground_filter
			frame.data = frame.data[mask]
			frame.labels = frame.labels[mask]
			frame.uuids = frame.uuids[mask]
			map_pos = map_pos[mask]
			map_ori = map_ori[mask]
		
		if self.city_space:
			map_pos, map_ori = get_map_pos()
			frame.data[:,self.pos_idx] = map_pos
			frame.data[:,self.rot_idx] = map_ori
		
		if self.merge:
			m = len(frame)
			pos = frame.data[:,self.pos_idx]
			width = frame.data.T[self.shape_idx[1]]
			mask = spatial.euclidean(pos, pos) - (width * self.merge * 0.5)**2 <= 0
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


class ArgoEgoRecorder(ArgoRecorder):
	"""
	"""
	def __init__(self, dataroot,
		city_space=False,
		argo_loader=None,
		**kwargs
		):
		"""
		"""
		super().__init__(**kwargs)
		self.city_space = city_space
		
		if isinstance(argo_loader, ArgoverseTrackingLoader):
			self.argo_loader = argo_loader
		else:
			self.argo_loader = ArgoverseTrackingLoader(dataroot)
		pass
	
	def get_ego_positions(self, frame, points):
		idx = self.argo_loader.get_idx_from_timestamp(frame.timestamp, frame.context)
		map = self.argo_loader.get_pose(idx, frame.context).inverse()
		return map.transform_point_cloud(points)
	
	def record(self, model, frame, *args, **kwargs):
		"""
		"""
		path = os.path.join(self.outputpath, frame.subset, frame.context, frame.format, frame.filename)
		os.makedirs(os.path.dirname(path), exist_ok=True)
		
		jdata = []
		for trk_id, tracker in model:
			if self.lost_filter and tracker.lost >= self.lost_filter:
				continue
			elif tracker.age < self.age_filter:
				continue
			elif self.dist_filter and self.dist_filter < tracker.dist:
				continue
				
			x = tracker.x.flatten()
			ego = np.vstack((x[self.pos_idx,], x[self.pos_idx,] + x[self.rot_idx,]))
			if self.city_space:
				ego = self.get_ego_positions(frame, ego)
			pos = dict(zip('xyz', ego[0]))
			obj = dict(zip(['length','width','height'], x[self.shape_idx,]))
			uuid = tracker.uuid
			
			rot = spatial.vec_to_yaw(*(ego[1] - ego[0]))
			rot = spatial.R.from_euler('zyx', (rot, 0, 0))
			rot = dict(zip('xyzw', rot.as_quat()))
			
			obj['center'] = pos
			obj['rotation'] = rot
			obj['score'] = tracker.score
			obj['track_label_uuid'] = str(uuid)
			obj['label_class'] = tracker.label
			obj['timestamp'] = frame.timestamp
			obj['tracked'] = True
			obj['occlusion'] = 0
			jdata.append(obj)
		
		with open(path, 'w') as f:
			json.dump(jdata, f)
		pass