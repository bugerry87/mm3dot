'''
'''
# Build In
from argparse import ArgumentParser

# Installed
import numpy as np

# Local
from . import Model, INITIALIZERS


def init_constant_velocity_parser(parents=[]):
	parser = ArgumentParser(
		parents=parents,
		description='Arguments for a Constant Velocity model'
		)
	parser.add_argument('--x_dim', type=int, metavar='INT', default=4)
	parser.add_argument('--z_dim', type=int, metavar='INT', default=2)
	parser.add_argument('--pos_idx', type=int, metavar='INT', nargs='*', default=(0,1))
	parser.add_argument('--vel_idx', type=int, metavar='INT', nargs='*', default=(2,3))
	parser.add_argument('--template', type=str, metavar='STRING', default='KalmanTracker')
	return parser


def constant_velocity(
	parse=None,
	**kwargs
	):
	'''
	'''
	def init(
		x_dim=2,
		z_dim=1,
		pos_idx=(0),
		vel_idx=(-1),
		template='KalmanTracker',
		label=None,
		**kwargs
		):
		model = Model()
		model.label = label
		model.x_dim = x_dim
		model.z_dim = z_dim
		model.u_dim = 0
		model.F = np.eye(x_dim)
		model.F[pos_idx, vel_idx] = 1
		model.H = np.eye(z_dim, x_dim)
		model.P = np.eye(x_dim) * 1000
		model.P[z_dim:] *= 10
		model.Q = np.eye(x_dim)
		model.Q[z_dim:] *= 0.01
		model.type = template
		return model
	
	if parse is not None:
		parser = init_constant_velocity_parser()
		args, _ = parser.parse_known_args(parse)
		return init(**args.__dict__, **kwargs)
	else:
		return init(**kwargs)

INITIALIZERS['constant_velocity'] = constant_velocity