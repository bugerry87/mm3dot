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
	return parser


def print_state(model):
	for trk_id, tracker in model.trackers.items():
		print('Tracker:', trk_id, tracker.x)
	print()


def load_kitti(kwargs):
	raise NotImplementedError("KITTI is currently not supported")


def load_nusenes(kwargs):
	raise NotImplementedError("NUSCENES is currently not supported")


def load_waymo(kwargs):
	raise NotImplementedError("WAYMO is currently not supported")


def load_argoverse(kwargs):
	raise NotImplementedError("ARGOVERSE is currently not supported")


def load_fake(kwargs):
	from datapi.fake import FakeLoader, init_fake_loader_parser
	parser = init_fake_loader_parser()
	args, _ = parser.parse_known_args(kwargs)
	callbacks = {
		'UPDATE': print_state
		}
	return FakeLoader(**args.__dict__), callbacks


def main(args, kwargs):
	if args.dataset is None:
		raise ValueError("ERROR: No dataset!")
	elif 'kitti' in args.dataset:
		dataloader, callbacks = load_kitti(kwargs)
	elif 'nuscenes' in args.dataset:
		dataloader, callbacks = load_nusenes(kwargs)
	elif 'waymo' in args.dataset:
		dataloader, callbacks = load_waymo(kwargs)
	elif 'argoverse' in args.dataset:
		dataloader, callbacks = load_argoverse(kwargs)
	elif 'fake' in args.dataset:
		dataloader, callbacks = load_fake(kwargs)
	else:
		raise ValueError("ERROR: Dataset '{}' unknown.".format(args.dataset))
	
	if args.model:
		models = load_models(ifile(args.model))
	elif args.initializer in INITIALIZERS:
		initializer = INITIALIZERS[args.initializer]
		models = {label:initializer(parse=kwargs, label=label) for label in dataloader.labels}
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