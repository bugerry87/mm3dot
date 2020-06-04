"""
"""
# Build In
from argparse import ArgumentParser

# Installed
import numpy as np

# Local
from . import Model, MOTION_MODELS


def init_constant_accumulation_parser(parents=[]):
	parser = ArgumentParser(
		parents=parents,
		description='Arguments for a Constant Accumulation model'
		)
	parser.add_argument('--x_dim', type=int, metavar='INT', default=4)
	parser.add_argument('--z_dim', type=int, metavar='INT', default=2)
	parser.add_argument('--pos_idx', type=int, metavar='TUPLE', nargs='*', default=(0,1))
	parser.add_argument('--vel_idx', type=int, metavar='TUPLE', nargs='*', default=(2,3))
	parser.add_argument('--acl_idx', type=int, metavar='TUPLE', nargs='*', default=None)
	parser.add_argument('--rot_idx', type=int, metavar='TUPLE', nargs='*', default=None)
	parser.add_argument('--template', type=str, metavar='STRING', default='KalmanTracker')
	parser.add_argument('--vel', type=float, metavar='FLOAT', default=1.0)
	parser.add_argument('--acl', type=float, metavar='FLOAT', default=0.1)
	parser.add_argument('--rot', type=float, metavar='FLOAT', default=0.01)
	return parser


class ConstantAccumulation(Model):
	"""
	"""
	def __init__(self, parse=None, **kwargs):
		"""
		"""
		def parse_kwargs(
				x_dim=3,
				z_dim=1,
				pos_idx=(0,),
				vel_idx=(1,),
				rot_idx=None,
				acl_idx=None,
				vel=1.0,
				acl=0.1,
				rot=0.01,
				prediction_model='KalmanTracker',
				label=None,
				**kwargs
				):
			self.label = label
			self.x_dim = x_dim
			self.z_dim = z_dim
			self.u_dim = 0
			self.F = np.eye(x_dim)
			self.H = np.eye(z_dim, x_dim)
			self.P = np.eye(x_dim) * 1000
			self.P[z_dim:] *= 10
			self.Q = np.eye(x_dim)
			self.Q[z_dim:] *= 0.01
			self.prediction_model = prediction_model
			self.motion_model = 'ConstantAccumulation'
			
			if vel_idx:
				self.F[pos_idx, vel_idx] = vel
			if acl_idx:
				self.F[vel_idx, acl_idx] = acl
			if rot_idx:
				self.F[vel_idx, rot_idx] = rot
			pass
		
		if parse is not None and len(parse):
			parser = init_constant_accumulation_parser()
			args, _ = parser.parse_known_args(parse)
			for k,v in kwargs.items():
				args.__setattr__(k,v)
			parse_kwargs(**args.__dict__)
		else:
			parse_kwargs(**kwargs)
	

MOTION_MODELS['ConstantAccumulation'] = ConstantAccumulation