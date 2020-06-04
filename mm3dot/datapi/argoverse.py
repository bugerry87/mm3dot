"""
"""
# Build-in
import json
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
	parser.add_argument('--inputfile', metavar='PATH')
	parser.add_argument('--outputfile', metavar='PATH', default='')
	parser.add_argument('--pos_idx', type=int, nargs='*', metavar='TUPLE', default=(0,1,2))
	parser.add_argument('--shape_idx', type=int, nargs='*', metavar='TUPLE', default=(3,4,5))
	parser.add_argument('--rot_idx', type=int, nargs='*', metavar='TUPLE', default=(6,7,8))
	parser.add_argument('--score_idx', type=int, nargs='*', metavar='TUPLE', default=(9,))
	parser.add_argument('--vel_idx', type=int, nargs='*', metavar='TUPLE', default=(10,11,12))
	parser.add_argument('--acl_idx', type=int, nargs='*', metavar='TUPLE', default=())
	parser.add_argument('--score_filter', type=float, metavar='FLOAT', default=0.9)
	return parser


class ArgoLoader():
	"""
	"""
	def __init__(self, inputfile,
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
		self.inputfile = inputfile
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
	
	def __iter__(self):
		"""
		Argoverse Detection Example:
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
		context = None
		for file in ifile(self.inputfile, sort=True):
			with open(file, 'r') as f:
				protos = json.load(f, object_hook=Prototype)
			data = np.empty((len(protos), self.z_dim))
			labels = [proto.label_class for proto in protos]
			cont = file.split('/')[-3]
			if context is None:
				self.context = context = cont
			elif context != cont:
				self.context = context = cont
				yield None # for reset
				
			for i, proto in enumerate(protos):
				data[i, self.pos_idx] = proto.center['x','y','z'][:len(self.pos_idx)]
				data[i, self.shape_idx] = proto['length', 'width', 'height'][:len(self.shape_idx)]
				data[i, self.score_idx] = proto.score
				data[i, self.rot_idx] = spatial.quat_to_vec(**proto.rotation.__dict__)[:len(self.rot_idx)]
			frame = Frame(labels, data, self.description)
			self.timestamp = frame.timestamp = proto.timestamp
			frame.context = context
			yield frame
		pass
