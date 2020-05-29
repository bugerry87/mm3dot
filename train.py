# Local
from mm3dot import MM3DOT
from model import load_models, INITIALIZERS
from utils import *

# Extensions
import model.constant_velocity
import model.kalman_tracker


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
		choices=['argoverse', 'nuscenes', 'kitti', 'fake'],
		default='fake',
		help='Name of the dataset.'
		)
	parser.add_argument(
		'--model', '-m',
		metavar='WILDCARD',
		nargs='*',
		help='Wildcard to one or more model files.'
		)
	default = next(iter(INITIALIZERS.keys()), None)
	parser.add_argument(
		'--initializer', '-i',
		metavar='MODEL',
		choices=INITIALIZERS.keys(),
		default=default,
		help='Initialize a new model. \n default={}'.format(default)
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
	return parser


def print_state(model):
	for trk_id, tracker in model:
		print('Trk:', trk_id, tracker.x.flatten(), 'Det:', tracker.feature)
	print()


def load_kitti(args, unparsed):
	raise NotImplementedError("KITTI is currently not supported")


def load_nusenes(args, unparsed):
	raise NotImplementedError("NUSCENES is currently not supported")


def load_waymo(args, unparsed):
	raise NotImplementedError("WAYMO is currently not supported")


def load_argoverse(args, unparsed):
	raise NotImplementedError("ARGOVERSE is currently not supported")


def load_fake(args, unparsed):
	from datapi.fake import FakeLoader, init_fake_loader_parser
	on_update = []
	
	if args.verbose:
		on_update.append(print_state)
	
	if args.visualize:
		import matplotlib.pyplot as plt
		from matplotlib.colors import hsv_to_rgb
		from math import sin, pi
		
		def plot_state(model):
			for trk_id, tracker in model:
				cs = 1.0/len(model)
				c1 = hsv_to_rgb([[sin(cs * trk_id), 1, 1]])
				c2 = hsv_to_rgb([[sin(cs * trk_id), 1, 0.5]])
				det = tracker.feature
				state = tracker.x.flatten()
				plt.scatter(*det[:2], c=c1)
				plt.quiver(*state[:4], color=c2)
			plt.pause(0.1)
			plt.draw()
		on_update.append(plot_state)
		
		def plot_show():
			plt.show()
	
	parser = init_fake_loader_parser()
	kwargs, _ = parser.parse_known_args(unparsed)
	callbacks = {
		'UPDATE': lambda x: [update(x) for update in on_update],
		'TERMINATE': lambda x: plot_show()
		}
	return FakeLoader(**kwargs.__dict__), callbacks


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
	elif args.initializer in INITIALIZERS:
		initializer = INITIALIZERS[args.initializer]
		models = {label:initializer(parse=unparsed, label=label) for label in dataloader.labels}
	else:
		raise ValueError("ERROR: Can't find any model! Please check --initializer or --model.")
	
	mm3dot = MM3DOT(models)
	for state in mm3dot.run(dataloader):
		if state in callbacks:
			callbacks[state](mm3dot)
	pass


if __name__ == '__main__':
	parser = init_arg_parser()
	main(*parser.parse_known_args())