"""
"""
# Build-in
import os
import json
from uuid import UUID
from argparse import ArgumentParser

# Installed
import numpy as np
import argoverse

# Local
from . import Frame, Prototype, ifile
from .. import spatial


def init_argoverse_arg_parser(parents=[]):
	parser = ArgumentParser(
		parents=parents,
		description='Arguments for a Agroverse lib'
		)
	parser.add_argument('--metafile', metavar='JSON')
	parser.add_argument('--inputfiles', metavar='PATH')
	parser.add_argument('--groundtruth', metavar='PATH')
	parser.add_argument('--outputfile', metavar='PATH', default='')
	parser.add_argument('--pos_idx', type=int, nargs='*', metavar='TUPLE', default=(0,1,2))
	parser.add_argument('--shape_idx', type=int, nargs='*', metavar='TUPLE', default=(3,4,5))
	parser.add_argument('--rot_idx', type=int, nargs='*', metavar='TUPLE', default=(6,7,8))
	parser.add_argument('--score_idx', type=int, nargs='*', metavar='TUPLE', default=(9,))
	parser.add_argument('--vel_idx', type=int, nargs='*', metavar='TUPLE', default=(10,11,12))
	parser.add_argument('--acl_idx', type=int, nargs='*', metavar='TUPLE', default=())
	parser.add_argument('--score_filter', type=float, metavar='FLOAT', default=0.9)
	return parser


def parse_path(path):
	parts = path.replace('\\','/').split('/')
	subset, context, format, filename = parts[-4:]
	root = os.path.join(*parts[:-4])
	return root, subset, context, format, filename


class ArgoLoader():
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
		frame.uuids = [UUID(proto.track_label_uuid) if proto.track_label_uuid else None for proto in protos]
		self.timestamp = frame.timestamp = proto.timestamp
		self.context = frame.context
		return frame
	
	
	def __iter__(self):
		"""
		"""
		context = None
		for file in ifile(self.inputfiles, sort=True):
			frame = self[file]
			
			if context is None:
				context = frame.context
			elif context != frame.context:
				context = frame.context
				yield None # for reset
			yield frame
		pass
	pass


class ArgoGTLoader(ArgoLoader):
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