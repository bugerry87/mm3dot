#!/usr/bin/env python


# Local
from mm3dot import MM3DOT
from mm3dot.model import load_models, Model, MOTION_MODELS
from mm3dot.spatial import DISTANCES, ASSIGNMENTS
from mm3dot.datapi import ifile

# Extensions
import mm3dot.model.constant_accumulation
import mm3dot.model.kalman_tracker

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
		parents=parents,
		add_help=False
		)
	parser.add_argument(
		'dataset',
		metavar='DATASET',
		nargs='?',
		choices=['argoverse', 'nuscenes', 'kitti', 'waymo', 'fake'],
		default=None,
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
	default = next(iter(DISTANCES.keys()), None)
	parser.add_argument(
		'--dist_func', '-d',
		metavar='DISTANCE_FUNC',
		choices=DISTANCES.keys(),
		default=default,
		help='Choose a distance function. \n default={}'.format(default)
		)
	default = 'hungarian' #next(iter(ASSIGNMENTS.keys()), None)
	parser.add_argument(
		'--assign_func', '-a',
		metavar='ASSIGNMENT_FUNC',
		choices=ASSIGNMENTS.keys(),
		default=default,
		help='Choose an assignment function. \n default={}'.format(default)
		)
	parser.add_argument(
		'--visualize', '-V',
		metavar='FLAG',
		type=bool,
		nargs='?',
		default=False,
		const=True,
		help='Visualize if visualizations are provided.'
		)
	parser.add_argument(
		'--verbose', '-v',
		metavar='FLAG',
		type=bool,
		nargs='?',
		default=False,
		const=True,
		help='Plot text if provided.'
		)
	parser.add_argument(
		'--hold_lost',
		type=int,
		metavar='INT',
		default=10,
		help="Num of lost frames before a tracker gets dropped."
		)
	return parser


timestamp = 0
def print_state(model, frame, *args, **kwargs):
	global timestamp
	likelihood = 0
	for trk_id, tracker in model:
		likelihood += tracker.likelihood
		feature = tracker.feature
		state = tracker.x.flatten()[:feature.size]
		print('\nError:', trk_id, np.round(np.abs(feature - state),2))
		print('Age:', tracker.age, 'Lost:', tracker.lost, 'Score:', tracker.score)
		print('Likelihood:', tracker.likelihood, 'Mahalanobis', tracker.mahalanobis)
	if len(model):
		likelihood / len(model)
	print('\nFrame Likelihood:', likelihood)
	print('Timestamp:', frame.timestamp, 'Delta:', frame.timestamp - timestamp)
	timestamp = frame.timestamp


def train_cov(model, *args, **kwargs):
	errors = None
	for i, (trk_id, tracker) in enumerate(model):
		state = tracker.x.flatten()
		feature = tracker.feature
		n = feature.size
		if errors is None:
			errors = np.zeros((state.size, len(model)))
		errors[:n,i] = (state[:n] - feature)**2
	pass


def plot_state(model, *args,
	history=1,
	pos_idx=(0,1),
	rot_idx=(3,4),
	score_idx=(6,),
	vel_idx=(7,8),
	**kwargs
	):
	"""
	"""
	pos_idx = pos_idx[:2]
	vel_idx = vel_idx[:2]
	rot_idx = rot_idx[:2]
	score_idx = score_idx[0]
	
	def det_quiver(det, rot, color):
		return plt.quiver(
			*det, *rot,
			color=c1,
			angles='xy',
			scale_units='xy',
			scale=0.125,
			pivot='mid'
			)
	
	def pred_quiver(pos, vel, color):
		return plt.quiver(
			*pos, *vel,
			color=color,
			angles='xy',
			scale_units='xy',
			scale=1
			)
	
	for trk_id, tracker in model:
		if 'history' not in tracker.__dict__:
			tracker.history = deque()
		
		state = tracker.x.flatten()
		pos = state[pos_idx,]
		vel = state[vel_idx,]
		rot = state[rot_idx,]
		det = tracker.feature[pos_idx,]
		
		score = tracker.feature[score_idx]
		confi = np.exp(-tracker.lost)
		cs = np.abs(np.sin(0.125 * trk_id))
		#c1 = ((*hsv_to_rgb((cs, 1, 0.5)), score * 0.5),)
		c1 = [(0.5,0.5,0,0.5)]
		c2 = ((*hsv_to_rgb((cs, 1, 1)), confi),)
		
		if len(tracker.history) >= history:
			tracker.history.rotate(-1)
			detection, prediction = tracker.history[0]
			detection.set_offsets(det)
			detection.set_UVC(*rot)
			detection.set_color(c1)
			if np.any(vel):
				if prediction:
					prediction.set_offsets(pos)
					prediction.set_UVC(*vel)
					prediction.set_color(c2)
				else:
					prediction = pred_quiver(pos, vel, c2)
					tracker.history[0] = (detection, prediction)
			elif prediction:
				prediction.remove()
				tracker.history[0] = (detection, None)
		else:
			detection = det_quiver(det, rot, c1)
			if np.any(vel):
				prediction = pred_quiver(pos, vel, c2)
			else:
				prediction = None
			tracker.history.append((detection, prediction))
	plt.draw()
	plt.pause(0.1)
	pass


def plot_gt(frame, gtloader):
	pos_idx = gtloader.pos_idx[:2]
	rot_idx = gtloader.rot_idx[:2]
	gt = gtloader[frame]
	pos = gt.data.T[pos_idx,]
	rot = gt.data.T[rot_idx,]
	#color = [(*hsv_to_rgb((np.abs(np.sin(0.125 * uuid.int)), 0.5, 0.5)), 0.5) for uuid in gt.uuids]
	color=[(0,0.5,0,0.5)]
	
	if 'plt' in gtloader.__dict__:
		gtloader.plt.remove()
	
	gtloader.plt = plt.quiver(
		pos[0], pos[1], rot[0], rot[1],
		color=color,
		angles='xy',
		scale_units='xy',
		scale=0.125,
		pivot='mid'
		)
	plt.draw()
	plt.pause(0.1)
	pass


def plot_show(*args, **kwargs):
	plt.show()
	pass


def plot_reset(*args, **kwargs):
	plt.clf()
	plt.scatter(0,0, marker='x', c='k')
	plt.xlim(-250, 250)
	plt.ylim(-250, 250)
	pass


def save_models(model, filename, ages):
	for trk_id, tracker in model:
		if tracker.lost:
			continue
		label = tracker.label
		if label in ages:
			if ages[label] < tracker.age:
				f = tracker.save("{}_model_{}.npz".format(filename, label))
				ages[label] = tracker.age
				model.models[label] = Model.load(f)
		else:
			ages[label] = tracker.age
	pass
	

def reset(model):
	print("\n_______________MODEL_RESET_______________\n")
	model.reset()
	pass
	

def load_kitti(args, unparsed):
	raise NotImplementedError("KITTI is currently not supported")


def load_nusenes(args, unparsed):
	raise NotImplementedError("NUSCENES is currently not supported")


def load_waymo(args, unparsed):
	from mm3dot.datapi.waymo import WaymoMergeLoader, WaymoRecorder, init_waymo_arg_parser
	from waymo_open_dataset.protos import metrics_pb2
	from mm3dot.spatial import vec_to_yaw
	
	parser = init_waymo_arg_parser()
	kwargs, _ = parser.parse_known_args(unparsed)
	dataloader = WaymoMergeLoader(**kwargs.__dict__)
	datarecorder = WaymoRecorder(**kwargs.__dict__)
	
	def waymo_record(model):
		pos_idx = dataloader.pos_idx
		shape_idx = dataloader.shape_idx
		rot_idx = dataloader.rot_idx
		context = dataloader.context
		timestamp = dataloader.timestamp
		for trk_id, tracker in model:
			if tracker.lost:
				continue
			object = metrics_pb2.Object()
			object.context_name = context
			object.frame_timestamp_micros = timestamp
			object.object.id = str(trk_id)
			object.object.type = tracker.label
			object.object.box.center_x = tracker.x[pos_idx[0]]
			object.object.box.center_y = tracker.x[pos_idx[1]]
			object.object.box.center_z = tracker.x[pos_idx[2]]
			object.object.box.length = tracker.x[shape_idx[0]]
			object.object.box.width = tracker.x[shape_idx[1]]
			object.object.box.height = tracker.x[shape_idx[2]]
			object.object.box.heading = vec_to_yaw(*tracker.x[rot_idx,])
			datarecorder.append(object)
	
	on_update = []
	on_terminate = []
	on_nodata = []
	ages = {}
	
	on_nodata.append(reset)
	on_update.append(waymo_record)
	on_update.append(lambda model, *args: save_models(model, kwargs.outputfile + 'waymo', ages))
	on_nodata.append(lambda *args: datarecorder.save())
	on_terminate.append(lambda *args: datarecorder.save())
	
	if args.verbose:
		on_update.append(print_state)
	
	if args.visualize:
		on_update.append(plot_state)
		on_nodata.append(plot_reset)
		plot_reset()
	
	callbacks = {
		'UPDATE': lambda *args: [update(*args) for update in on_update],
		'TERMINATE': lambda *args: [terminate(*args) for terminate in on_terminate],
		'NODATA': lambda *args: [nodata(*args) for nodata in on_nodata]
		}
	return dataloader, callbacks, kwargs


def load_argoverse(args, unparsed):
	from mm3dot.datapi.argoverse import init_argoverse_arg_parser
	parser = init_argoverse_arg_parser()
	kwargs, _ = parser.parse_known_args(unparsed)
	
	if kwargs.dataroot:
		from mm3dot.datapi.argoverse_filter import ArgoDetectionFilter
		dataloader = ArgoDetectionFilter(**kwargs.__dict__)
	else:
		from mm3dot.datapi.argoverse import ArgoDetectionLoader
		dataloader = ArgoDetectionLoader(**kwargs.__dict__)
		
	
	if kwargs.groundtruth is not None:
		from mm3dot.datapi.argoverse import ArgoGTLoader
		groundtruth = ArgoGTLoader(**kwargs.__dict__)
	else:
		groundtruth = None
	
	on_update = []
	on_terminate = []
	on_nodata = []
	
	on_nodata.append(reset)
	
	if args.verbose:
		on_update.append(print_state)
	
	if args.visualize:
		if groundtruth:
			on_update.append(lambda _, frame, *args, **kwargs: plot_gt(frame, groundtruth))
		on_update.append(plot_state)
		on_nodata.append(plot_reset)
		plot_reset()
	
	if kwargs.outputpath:
		from mm3dot.datapi.argoverse import ArgoRecorder
		recorder = ArgoRecorder(**kwargs.__dict__)
		on_update.append(recorder)
		on_terminate.append(recorder)
	
	callbacks = {
		'UPDATE': lambda *args, **kwargs: [update(*args, **kwargs) for update in on_update],
		'TERMINATE': lambda *args, **kwargs: [terminate(*args, **kwargs) for terminate in on_terminate],
		'NODATA': lambda *args, **kwargs: [nodata(*args, **kwargs) for nodata in on_nodata]
		}
	return dataloader, callbacks, kwargs


def load_fake(args, unparsed):
	from datapi.fake import FakeLoader, init_fake_loader_parser
	parser = init_fake_loader_parser()
	kwargs, _ = parser.parse_known_args(unparsed)
	dataloader = FakeLoader(**kwargs.__dict__)
	
	on_update = []
	on_terminate = []
	
	if args.verbose:
		on_update.append(print_state)
		pass
	
	if args.visualize:
		on_update.append(plot_state)
		plot_reset()
		pass
	
	callbacks = {
		'UPDATE': lambda *args: [update(*args) for update in on_update],
		'TERMINATE': lambda *args: [terminate(*args) for terminate in on_terminate],
		'NODATA': reset
		}
	return dataloader, callbacks, kwargs


def main(args, unparsed):
	if args.dataset is None:
		raise ValueError("ERROR: No dataset!")
	elif 'kitti' in args.dataset:
		dataloader, callbacks, kwargs = load_kitti(args, unparsed)
	elif 'nuscenes' in args.dataset:
		dataloader, callbacks, kwargs = load_nusenes(args, unparsed)
	elif 'waymo' in args.dataset:
		dataloader, callbacks, kwargs = load_waymo(args, unparsed)
	elif 'argoverse' in args.dataset:
		dataloader, callbacks, kwargs = load_argoverse(args, unparsed)
	elif 'fake' in args.dataset:
		dataloader, callbacks, kwargs = load_fake(args, unparsed)
	else:
		raise ValueError("ERROR: Dataset '{}' unknown.".format(args.dataset))
	
	if args.model:
		models = load_models(ifile(args.model))
	elif args.initializer in MOTION_MODELS:
		initializer = MOTION_MODELS[args.initializer]
		desc = dataloader.description
		models = {label:initializer(parse=unparsed, label=label, **desc) for label in dataloader.labels}
	else:
		raise ValueError("ERROR: Can't find any model! Please check --initializer or --model.")
	
	mm3dot = MM3DOT(models, **args.__dict__)
	try:
		for state, *args in mm3dot.run(dataloader):
			if state in callbacks:
				callbacks[state](mm3dot, *args, **kwargs.__dict__)
	except KeyboardInterrupt:
		callbacks['TERMINATE'](mm3dot, *args, **kwargs.__dict__)
	pass


if __name__ == '__main__':
	parser = init_arg_parser()
	args, unparsed = parser.parse_known_args()
	if args.dataset is None:
		parser.print_help()
		exit()
	main(args, unparsed)