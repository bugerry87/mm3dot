#!/usr/bin/env python


# Local
from mm3dot import MM3DOT
from model import load_models, MOTION_MODELS
from utils import *

# Extensions
import model.constant_accumulation
import model.kalman_tracker

try:
	import matplotlib.pyplot as plt
	from matplotlib.colors import hsv_to_rgb
	from collections import deque
	import numpy as np
except:
	print("WARNING: In case of visualization matplotlib is required!")


def init_arg_parser(parents=[]):
	from argparse import ArgumentParser
	parser = ArgumentParser(
		description='Runs the tracking model on pre-computed detection results.',
		parents=parents
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
	log_likelihood = 0
	for trk_id, tracker in model:
		log_likelihood += tracker.log_likelihood
		print('\nTrk:', trk_id, np.round(tracker.x.flatten(),2), '\nDet:', np.round(tracker.feature,2), '\nLogLikelihood:', tracker.log_likelihood)
	if len(model):
		log_likelihood / len(model)
	print('\nFrame Likelihood:', log_likelihood)


def train_cov(model):
	errors = None
	for i, (trk_id, tracker) in enumerate(model):
		state = tracker.x.flatten()
		feature = tracker.feature
		n = feature.size
		if errors is None:
			errors = np.zeros((state.size, len(model)))
		errors[:n,i] = (state[:n] - feature)**2


def plot_state(model, pos_idx=(0,1), vel_idx=(2,3)):
	pos_idx = pos_idx[:2]
	vel_idx = vel_idx[:2]
	for trk_id, tracker in model:
		if 'history' not in tracker.__dict__:
			tracker.history = deque()
		cs = np.abs(np.sin(0.1 * trk_id))
		c1 = hsv_to_rgb((cs, 1, 1))
		c2 = hsv_to_rgb((cs, 1, 0.5))
		det = tracker.feature[pos_idx,]
		state = tracker.x.flatten()
		pos = state[pos_idx,]
		vel = state[vel_idx,]
		
		if len(tracker.history) > 10:
			tracker.history.rotate(-1)
			scatter, quiver = tracker.history[0]
			scatter.set_offsets(det)
			scatter.set_color(c1)
			if np.any(vel):
				if quiver:
					quiver.set_offsets(pos)
					quiver.set_UVC(*vel)
					quiver.set_color(c2)
				else:
					quiver = plt.quiver(*pos, *vel, color=[c2], angles='xy', scale_units='xy', scale=1)
					tracker.history[0] = (scatter, quiver)
			elif quiver:
				quiver.remove()
				tracker.history[0] = (scatter, None)
		else:
			scatter = plt.scatter(*det, c=[c1])
			if np.any(vel):
				quiver = plt.quiver(*pos, *vel, color=[c2], angles='xy', scale_units='xy', scale=1)
			else:
				quiver = None
			tracker.history.append((scatter, quiver))
	plt.pause(0.1)
	plt.draw()


def plot_show(*args):
	plt.show()


def plot_reset(*args):
	plt.clf()


def save_models(model, filename, ages):
	for trk_id, tracker in model:
		if tracker.lost:
			continue
		label = tracker.label
		if label in ages:
			if ages[label] < tracker.age:
				tracker.save("{}_model_{}.npz".format(filename, label))
				ages[label] = tracker.age
		else:
			ages[label] = tracker.age
	

def reset(model):
	print("\n_______________MODEL_RESET_______________\n")
	model.reset()


def load_kitti(args, unparsed):
	raise NotImplementedError("KITTI is currently not supported")


def load_nusenes(args, unparsed):
	raise NotImplementedError("NUSCENES is currently not supported")


def load_waymo(args, unparsed):
	from datapi.waymo import WaymoMergeLoader, WaymoRecorder, init_waymo_arg_parser
	from waymo_open_dataset.protos import metrics_pb2
	from datapi import xyz_to_yaw
	
	parser = init_waymo_arg_parser()
	kwargs, _ = parser.parse_known_args(unparsed)
	waymoloader = WaymoMergeLoader(**kwargs.__dict__)
	waymorecorder = WaymoRecorder(**kwargs.__dict__)
	
	def waymo_record(model):
		pos_idx = waymoloader.pos_idx
		shape_idx = waymoloader.shape_idx
		rot_idx = waymoloader.rot_idx
		context = waymoloader.context
		timestamp = waymoloader.timestamp
		for trk_id, tracker in model:
			if tracker.lost:
				continue
			object = metrics_pb2.Object()
			object.context_name = context
			object.frame_timestamp_micros = timestamp
			#object.score = tracker.log_likelihood
			object.object.id = str(trk_id)
			object.object.type = tracker.label
			object.object.box.center_x = tracker.x[pos_idx[0]]
			object.object.box.center_y = tracker.x[pos_idx[1]]
			object.object.box.center_z = tracker.x[pos_idx[2]]
			object.object.box.length = tracker.x[shape_idx[0]]
			object.object.box.width = tracker.x[shape_idx[1]]
			object.object.box.height = tracker.x[shape_idx[2]]
			object.object.box.heading = xyz_to_yaw(*tracker.x[rot_idx,])
			waymorecorder.append(object)
	
	on_update = []
	on_terminate = []
	on_nodata = []
	ages = {}
	
	on_nodata.append(reset)
	on_update.append(waymo_record)
	on_update.append(lambda x: save_models(x, kwargs.outputfile + 'waymo2', ages))
	on_nodata.append(lambda x: waymorecorder.save())
	on_terminate.append(lambda x: waymorecorder.save())
	
	if args.verbose:
		on_update.append(print_state)
	
	if args.visualize:
		on_update.append(lambda x: plot_state(x, waymoloader.pos_idx, waymoloader.vel_idx))
		on_terminate.append(plot_show)
		on_nodata.append(plot_reset)
	
	callbacks = {
		'UPDATE': lambda x: [update(x) for update in on_update],
		'TERMINATE': lambda x: [terminate(x) for terminate in on_terminate],
		'NODATA': lambda x: [nodata(x) for nodata in on_nodata]
		}
	return waymoloader, callbacks


def load_argoverse(args, unparsed):
	raise NotImplementedError("ARGOVERSE is currently not supported")


def load_fake(args, unparsed):
	from datapi.fake import FakeLoader, init_fake_loader_parser
	parser = init_fake_loader_parser()
	kwargs, _ = parser.parse_known_args(unparsed)
	fakeloader = FakeLoader(**kwargs.__dict__)
	
	on_update = []
	on_terminate = []
	
	if args.verbose:
		on_update.append(print_state)
		pass
	
	if args.visualize:
		on_update.append(lambda x: plot_state(x, fakeloader.pos_idx, fakeloader.vel_idx))
		on_terminate.append(plot_show)
		pass
	
	callbacks = {
		'UPDATE': lambda x: [update(x) for update in on_update],
		'TERMINATE': lambda x: [terminate(x) for terminate in on_terminate],
		'NODATA': reset
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
	try:
		for state in mm3dot.run(dataloader):
			if state in callbacks:
				callbacks[state](mm3dot)
	except KeyboardInterrupt:
		callbacks['TERMINATE'](mm3dot)
	pass


if __name__ == '__main__':
	parser = init_arg_parser()
	main(*parser.parse_known_args())