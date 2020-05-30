
# Local
from mm3dot import MM3DOT
from model import load_models, MOTION_MODELS
from utils import *

# Extensions
import model.constant_velocity
import model.kalman_tracker

try:
	import matplotlib.pyplot as plt
	from matplotlib.colors import hsv_to_rgb
	from math import sin, pi
except:
	print("WARNING: In case of visualization matplotlib is required!")


def init_arg_parser(parents=[]):
	from argparse import ArgumentParser
	parser = ArgumentParser(
		description='Runs the tracking model on pre-computed detection results.',
		parents=parents
		)
	parser.add_argument(
		'--data_root', '-X',
		metavar='PATH',
		help='Path to the dataset root.'
		)
	parser.add_argument(
		'--dataset', '-d',
		metavar='NAME',
		choices=['argoverse', 'nuscenes', 'kitti', 'waymo', 'fake'],
		default='fake',
		help='Name of the dataset.'
		)
	parser.add_argument(
		'--model', '-m',
		metavar='WILDCARD',
		nargs='*',
		help='Wildcard to one or more model files.'
		)
	default = next(iter(MOTION_MODELS.keys()), None)
	parser.add_argument(
		'--initializer', '-i',
		metavar='MOTION_MODEL',
		choices=MOTION_MODELS.keys(),
		default=default,
		help='Initialize a new model based on a motion model. \n default={}'.format(default)
		)
	parser.add_argument(
		'--output', '-Y',
		metavar='PATH',
		help='Output path for tracking results'
		)
	parser.add_argument(
		'--visualize', '-V',
		metavar='FLAG',
		type=bool,
		nargs='?',
		default=False,
		const=True,
		help='Visualize if visualizations are provided'
		)
	parser.add_argument(
		'--verbose', '-v',
		metavar='FLAG',
		type=bool,
		nargs='?',
		default=False,
		const=True,
		help='Text plot if provided'
		)
	parser.add_argument(
		'--hold_lost',
		type=int,
		metavar='INT',
		default=10,
		help="Num of lost frames before a tracker gets dropped."
		)
	return parser


def print_state(model):
	for trk_id, tracker in model:
		print('Trk:', trk_id, tracker.x.flatten(), 'Det:', tracker.feature)
	print()


def plot_state(model, pos_idx=(0,1), vel_idx=(2,3)):
	for trk_id, tracker in model:
		cs = 1.0/len(model)
		c1 = hsv_to_rgb([[sin(cs * trk_id), 1, 1]])
		c2 = hsv_to_rgb([[sin(cs * trk_id), 1, 0.5]])
		det = tracker.feature
		state = tracker.x.flatten()
		plt.scatter(*det[pos_idx,], c=c1)
		plt.quiver(*state[(*pos_idx, *vel_idx),], color=c2)
		print("VIZ:", state[(*pos_idx, *vel_idx),])
	plt.pause(0.1)
	plt.draw()


def plot_show(*args):
	plt.show()


def load_kitti(args, unparsed):
	raise NotImplementedError("KITTI is currently not supported")


def load_nusenes(args, unparsed):
	raise NotImplementedError("NUSCENES is currently not supported")


def load_waymo(args, unparsed):
	from datapi.waymo import WaymoLoader, init_waymo_loader_parser
	parser = init_waymo_loader_parser()
	kwargs, _ = parser.parse_known_args(unparsed)
	waymoloader = WaymoLoader(**kwargs.__dict__)
	pos_idx = waymoloader.pos_idx
	vel_idx = waymoloader.vel_idx
	
	on_update = []
	on_terminate = []
	
	if args.verbose:
		on_update.append(print_state)
	
	if args.visualize:
		on_update.append(plot_state)
		on_terminate.append(lambda x: plot_show(x, pos_idx=pos_idx[:2], vel_idx=vel_idx[:2]))
	
	callbacks = {
		'UPDATE': lambda x: [update(x) for update in on_update],
		'TERMINATE': lambda x: [terminate(x) for terminate in on_terminate]
		}
	return waymoloader, callbacks


def load_argoverse(args, unparsed):
	raise NotImplementedError("ARGOVERSE is currently not supported")


def load_fake(args, unparsed):
	from datapi.fake import FakeLoader, init_fake_loader_parser
	parser = init_fake_loader_parser()
	kwargs, _ = parser.parse_known_args(unparsed)
	fakeloader = FakeLoader(**kwargs.__dict__)
	pos_idx = fakeloader.pos_idx
	vel_idx = fakeloader.vel_idx
	
	on_update = []
	on_terminate = []
	
	if args.verbose:
		on_update.append(print_state)
	
	if args.visualize:
		on_update.append(plot_state)
		on_terminate.append(lambda x: plot_show(x, pos_idx=pos_idx[:2], vel_idx=vel_idx[:2]))
	
	callbacks = {
		'UPDATE': lambda x: [update(x) for update in on_update],
		'TERMINATE': lambda x: [terminate(x) for terminate in on_terminate]
		}
	return fakeloader, callbacks


def main(args, unparsed):
	if args.dataset is None:
		raise ValueError("ERROR: No dataset!")
	elif 'kitti' in args.dataset:
		dataloader, callbacks = load_kitti(args, unparsed)
	elif 'nuscenes' in args.dataset:
		dataloader, callbacks = load_nusenes(args, unparsed)
	elif 'waymo' in args.dataset:
		dataloader, callbacks = load_waymo(args, unparsed)
	elif 'argoverse' in args.dataset:
		dataloader, callbacks = load_argoverse(args, unparsed)
	elif 'fake' in args.dataset:
		dataloader, callbacks = load_fake(args, unparsed)
	else:
		raise ValueError("ERROR: Dataset '{}' unknown.".format(args.dataset))
	
	if args.model:
		models = load_models(ifile(args.model))
	elif args.initializer in MOTION_MODELS:
		initializer = MOTION_MODELS[args.initializer]
		description = dataloader.description
		models = {label:initializer(parse=unparsed, label=label, **description) for label in dataloader.labels}
	else:
		raise ValueError("ERROR: Can't find any model! Please check --initializer or --model.")
	
	mm3dot = MM3DOT(models, **args.__dict__)
	for state in mm3dot.run(dataloader):
		if state in callbacks:
			callbacks[state](mm3dot)
	pass


if __name__ == '__main__':
	parser = init_arg_parser()
	main(*parser.parse_known_args())