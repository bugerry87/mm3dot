"""
"""
# Build-in
import os
import json
from uuid import UUID
from argparse import ArgumentParser

# Installed
import numpy as np

# Local
from . import Frame, Prototype, ifile
from .. import spatial


def init_argoverse_arg_parser(parents=[]):
	parser = ArgumentParser(
		parents=parents,
		description='Arguments for a Agroverse lib'
		)
	parser.add_argument('--inputfiles', metavar='WILDCARD',
		help="A wildcard to pre-computed detection results.")
	parser.add_argument('--groundtruth', metavar='WILDCARD',
		help="A wildcard to the ground truth files")
	parser.add_argument('--dataroot', metavar='PATH',
		help="Path to the argoverse dataset (required for filtering).")
	parser.add_argument('--outputpath', metavar='PATH', default=None,
		help="Filepath to store the tracking results at.")
	parser.add_argument('--pos_idx', type=int, nargs='*', metavar='INT', default=(0,1,2),
		help="Indices for positional information.")
	parser.add_argument('--shape_idx', type=int, nargs='*', metavar='INT', default=(3,4,5),
		help="Indices for shape information.")
	parser.add_argument('--rot_idx', type=int, nargs='*', metavar='INT', default=(6,7,8),
		help="Indices for rotative information.")
	parser.add_argument('--score_idx', type=int, nargs='*', metavar='INT', default=(9,),
		help="Indices for score information.")
	parser.add_argument('--vel_idx', type=int, nargs='*', metavar='INT', default=(10,11,12),
		help="Indices for velocity information.")
	parser.add_argument('--acl_idx', type=int, nargs='*', metavar='INT', default=(),
		help="Indices for accelerative information.")
	parser.add_argument('--score_filter', type=float, metavar='FLOAT', default=0.5,
		help="Consider detections with scores above only.")
	parser.add_argument('--dist_filter', type=float, metavar='FLOAT', default=0.0,
		help="Consider trackers inside a range to the detection only.")
	parser.add_argument('--off_ground_filter', type=float, metavar='FLOAT', default=1.0,
		help="Consider detections on the ground only (fly-tolerance).")
	return parser


def parse_path(path):
	parts = path.replace('\\','/').split('/')
	subset, context, format, filename = parts[-4:]
	root = os.path.join(*parts[:-4])
	return root, subset, context, format, filename


class ArgoDetectionLoader():
	"""
	ArgoLoader loads precomputed detection results.
	
	Expected incoming data:
		"center": {
			"x": -24.69597053527832,
			"y": -3.5076069831848145,
			"z": 0.5611804127693176
		},
		"height": 1.7947044372558594,
		"label_class": "VEHICLE",
		"length": 4.827436923980713,
		"occlusion": 0,
		"rotation": {
			"w": -0.9999523072913704,
			"x": 0.0,
			"y": 0.0,
			"z": 0.009766429371304468
		},
		"score": 0.8911644220352173,
		"timestamp": 315969629019741000,
		"track_label_uuid": null,
		"tracked": true,
		"width": 1.9237309694290161
	"""
	def __init__(self, inputfiles,
		pos_idx=(0,1,2),
		shape_idx=(3,4,5),
		rot_idx=(6,7,8),
		score_idx=(9,),
		vel_idx=(10,11,12),
		acl_idx=(),
		**kwargs
		):
		"""
		"""
		self.inputfiles = inputfiles
		self.pos_idx = pos_idx if isinstance(pos_idx, (tuple, list)) else (pos_idx,)
		self.shape_idx = shape_idx if isinstance(shape_idx, (tuple, list)) else (shape_idx,)
		self.rot_idx = rot_idx if isinstance(rot_idx, (tuple, list)) else (rot_idx,)
		self.score_idx = score_idx if isinstance(score_idx, (tuple, list)) else (score_idx,)
		self.vel_idx = vel_idx if isinstance(vel_idx, (tuple, list)) else (vel_idx,)
		self.acl_idx = acl_idx if isinstance(acl_idx, (tuple, list)) else (acl_idx,)
		self.z_dim = np.max((*self.pos_idx, *self.shape_idx, *self.rot_idx, *self.score_idx))+1
		self.x_dim = np.max((self.z_dim, *self.vel_idx, *self.acl_idx))+1
		self.labels = ["VEHICLE", "PEDESTRIAN", 
			"ON_ROAD_OBSTACLE", "LARGE_VEHICLE", "BICYCLE",
			"BICYCLIST", "BUS", "OTHER_MOVERS", "TRAILER",
			"MOTORCYCLIST", "MOPED", "MOTORCYCLE", "STROLLER",
			"EMERGENCY_VEHICLE", "ANIMAL"
			]
		self.description = {
			'pos_idx':self.pos_idx,
			'shape_idx':self.shape_idx,
			'rot_idx':self.rot_idx,
			'score_idx':self.score_idx,
			'vel_idx':self.vel_idx,
			'acl_idx':self.acl_idx,
			'x_dim':self.x_dim,
			'z_dim':self.z_dim
			}
		pass
	
	def __getitem__(self, file):
		"""
		"""
		with open(file, 'r') as f:
			protos = json.load(f, object_hook=Prototype)
		data = np.empty((len(protos), self.z_dim))
		labels = [proto.label_class for proto in protos]
			
		for i, proto in enumerate(protos):
			data[i, self.pos_idx] = proto.center['x','y','z'][:len(self.pos_idx)]
			data[i, self.shape_idx] = proto['length', 'width', 'height'][:len(self.shape_idx)]
			data[i, self.score_idx] = proto.score if 'score' in proto else 1.0
			data[i, self.rot_idx] = spatial.quat_to_vec(**proto.rotation.__dict__)[:len(self.rot_idx)]
		frame = Frame(labels, data, self.description)
		frame.root, frame.subset, frame.context, frame.format, frame.filename = parse_path(file)
		frame.uuids = np.array([UUID(proto.track_label_uuid) if proto.track_label_uuid else None for proto in protos])
		self.timestamp = frame.timestamp = proto.timestamp
		self.context = frame.context
		return frame
	
	
	def __iter__(self):
		"""
		"""
		idx = 0
		context = None
		for file in ifile(self.inputfiles, sort=True):
			frame = self[file]
			frame.idx = idx
			idx += 1
			
			if context is None:
				context = frame.context
			elif context != frame.context:
				context = frame.context
				idx = 0
				yield None # for reset
			yield frame
		pass
	pass


class ArgoGTLoader(ArgoDetectionLoader):
	"""
	"""
	def __init__(self, groundtruth, **kwargs):
		"""
		"""
		kwargs['inputfiles'] = groundtruth
		self.groundtruth = groundtruth
		super().__init__(**kwargs)
	
	def __getitem__(self, file):
		if isinstance(file, Frame):
			root, subset, _, _, _ = parse_path(self.groundtruth)
			file = os.path.join(
				root,
				subset,
				file.context,
				file.format,
				file.filename
				)
		return super().__getitem__(file)
	pass


class ArgoRecorder():
	"""
	ArgoRecorder records tracking results frame-wise.
	
	Expected outgoing data:
		"center": {
			"x": -24.69597053527832,
			"y": -3.5076069831848145,
			"z": 0.5611804127693176
		},
		"height": 1.7947044372558594,
		"label_class": "VEHICLE",
		"length": 4.827436923980713,
		"occlusion": 0,
		"rotation": {
			"w": -0.9999523072913704,
			"x": 0.0,
			"y": 0.0,
			"z": 0.009766429371304468
		},
		"score": 0.8911644220352173,
		"timestamp": 315969629019741000,
		"track_label_uuid": "6022f892-fdfa-4a32-83e0-274b911e7dbf",
		"tracked": true,
		"width": 1.9237309694290161
	"""
	def __init__(self,
		outputpath='./',
		pos_idx=(0,1,2),
		shape_idx=(3,4,5),
		rot_idx=(6,7,8),
		dist_filter=0.0,
		lost_filter=1,
		age_filter=0,
		**kwargs
		):
		"""
		"""
		self.outputpath = outputpath
		self.pos_idx = pos_idx if isinstance(pos_idx, (tuple, list)) else (pos_idx,)
		self.shape_idx = shape_idx if isinstance(shape_idx, (tuple, list)) else (shape_idx,)
		self.rot_idx = rot_idx if isinstance(rot_idx, (tuple, list)) else (rot_idx,)
		self.dist_filter = dist_filter
		self.lost_filter = lost_filter
		self.age_filter = age_filter
		pass
	
	def __call__(self, model, frame, *args, **kwargs):
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
			elif self.dist_filter and self.dist_filter < tracker.distance:
				continue
				
			x = tracker.x.flatten()
			pos = dict(zip('xyz', x[self.pos_idx,]))
			obj = dict(zip(['length','width','height'], x[self.shape_idx,]))
			uuid = tracker.uuid
			rot = spatial.vec_to_yaw(*x[self.rot_idx,])
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