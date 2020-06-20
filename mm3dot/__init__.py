
# Build-in
from uuid import uuid1

# Installed
import numpy as np

# Local
from .spatial import S_cov, DISTANCES, ASSIGNMENTS
from .model import PREDICTION_MODELS, MOTION_MODELS

NODATA = 'NODATA'
SPAWN = 'SPAWN'
UPDATE = 'UPDATE'
PREDICT = 'PREDICT'
DROP = 'DROP'
TERMINATE = 'TERMINATE'


def init_mm3dot_arg_parser(parents=[]):
	from argparse import ArgumentParser
	parser = ArgumentParser(
		parents=parents,
		description='Arguments for a MM3DOT lib',
		add_help=len(parents) == 0
		)
	default = next(iter(MOTION_MODELS.keys()), None)
	parser.add_argument(
		'--initializer',
		metavar='MOTION_MODEL',
		choices=MOTION_MODELS.keys(),
		default=default,
		help='Initialize a new model based on a motion model. \n default={}'.format(default)
		)
	default = 'mahalanobis' #next(iter(DISTANCES.keys()), None)
	parser.add_argument(
		'--dist_func',
		metavar='DISTANCE_FUNC',
		choices=DISTANCES.keys(),
		default=default,
		help='Choose a distance function. \n default={}'.format(default)
		)
	default = 'hungarian' #'hungarian' #next(iter(ASSIGNMENTS.keys()), None)
	parser.add_argument(
		'--assign_func',
		metavar='ASSIGNMENT_FUNC',
		choices=ASSIGNMENTS.keys(),
		default=default,
		help='Choose an assignment function. \n default={}'.format(default)
		)
	parser.add_argument('--max_lost', type=int, metavar='INT', default=10,
		help="Num of lost frames before a tracker gets dropped.")
	parser.add_argument('--max_dist', type=float, metavar='FLOAT', default=11.0,
		help="Max accepted distance threshold for assignment.")
	return parser


class MM3DOT():
	def __init__(self, models,
			dist_func='mahalanobis',
			assign_func='hungarian',
			max_lost=10,
			max_dist=0.0,
			**kwargs
		):
		"""
		"""
		self.__trk_id_cntr__ = 0
		self.models = models
		self.dist_func = DISTANCES[dist_func] if isinstance(dist_func, str) else dist_func
		self.assign_func = ASSIGNMENTS[assign_func] if isinstance(assign_func, str) else assign_func
		self.trackers = {}
		self.frame_counter = 0
		self.max_lost = max_lost
		self.max_dist = max_dist
		pass
	
	def __len__(self):
		return len(self.trackers.keys())
	
	def __getitem__(self, idx):
		return self.trackers[idx]
	
	def __contains__(self, idx):
		return idx in self.trackers
	
	def __iter__(self):
		return iter(self.trackers.items())
	
	def spawn_trackers(self, frame, **kwargs):
		for label, detection in frame:
			if label in self.models:
				model = self.models[label]
				if model.prediction_model in PREDICTION_MODELS:
					prediction_model = PREDICTION_MODELS[model.prediction_model]
				else:
					raise RuntimeWarning("Model type '{}' is not registered!".format(model.prediction_model))
				self.__trk_id_cntr__ += 1
				tracker = prediction_model(detection, model, **kwargs)
				tracker.id = self.__trk_id_cntr__
				tracker.uuid = uuid1()
				tracker.age = 0
				tracker.lost = 0
				self.trackers[self.__trk_id_cntr__] = tracker
			else:
				print("WARNING: No model for label '{}'".format(label))
		return self
	
	def predict(self, **kwargs):
		for tracker in self.trackers.values():
			tracker.predict(**kwargs)
		
	def match(self, frame, match_idx=None, max_dist=None, **kwargs):
		if max_dist is not None:
			self.max_dist = max_dist
		if match_idx is None:
			M = frame.shape[-1]
			match_idx = slice(M)
		else:
			M = len(match_idx)
		N = len(self.trackers)
		track_states = np.empty((N,M))
		track_covars = np.empty((N,M,M))
		track_ids = np.empty(N)
		for i, (idx, tracker) in enumerate(self.trackers.items()):
			track_states[i] = tracker.x[match_idx,].flatten()
			track_covars[i] = spatial.S_cov(tracker.H, tracker.P, tracker.R)[match_idx, match_idx] #tracker.SI[match_idx, match_idx]
			track_ids[i] = idx
		cost = self.dist_func(track_states, frame.data[:,match_idx], track_covars, **kwargs)
		(trk_id, det_id), trkm, detm = self.assign_func(cost, **kwargs)
		if self.max_dist:
			mask = cost[(trk_id, det_id)] > self.max_dist
			trkm[trk_id[mask]] = False
			detm[det_id[mask]] = False
		return (track_ids[trkm], *frame[detm]), track_ids[~trkm], frame[~detm]
		
	def update(self, frame, matches, lost, unmatched, **kwargs):
		# update matched trackers
		self.frame_counter += 1
		for trk_id, label, detection in zip(*matches):
			if self.trackers[trk_id].label != label:
				print("WARNING: Label changed!")
			self.trackers[trk_id].lost = 0
			self.trackers[trk_id].age += 1
			self.trackers[trk_id].update(detection, **kwargs)
		# mark unmatched trackers
		for trk_id in lost:
			self.trackers[trk_id].lost += 1
			self.trackers[trk_id].age += 1
			self.trackers[trk_id].update(None, **kwargs)
		# spawn trackers for unmatched frame
		self.spawn_trackers(zip(*unmatched))
		return self
	
	def drop_trackers(self, max_lost=None, **kwargs):
		if max_lost is not None:
			self.max_lost = max_lost
		victims = {}
		for trk_id, tracker in self.trackers.items():
			if tracker.lost >= self.max_lost:
				victims[trk_id] = tracker
		for trk_id in victims.keys():
			del self.trackers[trk_id]
		return victims
	
	def reset(self):
		self.trackers.clear()

	def run(self, dataloader, **kwargs):
		spawn = True
		for frame in dataloader:
			if frame is None:
				yield NODATA
				self.reset()
				spawn = True
				continue
			
			if spawn:
				self.spawn_trackers(frame, **kwargs)
				spawn = False
				yield SPAWN, frame
				continue
			
			match_results = self.match(frame, **kwargs)
			self.update(frame, *match_results, **kwargs)
			yield UPDATE, frame
			
			self.predict(**kwargs)
			yield PREDICT, frame
			
			victims = self.drop_trackers(**kwargs)
			yield DROP, victims
			
		yield TERMINATE


def run_mm3dots(dataloader, models, model_args, **runtime_args):
	"""
	"""
	from .datapi import Frame
	
	mm3dots = { label:MM3DOT(models, **model_args) for label in dataloader.labels }
	spawn=True
	for frame in dataloader:
		for label, mm3dot in mm3dots.items():
			print('\n---', label, '---\n')
			if frame is None:
				yield NODATA, mm3dot
				mm3dot.reset()
				spawn = True
				continue
			
			mask = frame.labels == label
			subframe = Frame(frame.labels[mask], frame.data[mask], frame.description)
			subframe.timestamp = frame.timestamp
			subframe.idx = frame.idx
			if 'uuids' in frame.__dict__:
				subframe.uuids = frame.uuids[mask]
			
			if spawn:
				mm3dot.spawn_trackers(subframe, **runtime_args)
				spawn = False
				yield SPAWN, mm3dot, frame
				continue
				
			match_results = mm3dot.match(subframe, **runtime_args)
			mm3dot.update(subframe, *match_results, **runtime_args)
			yield UPDATE, mm3dot, frame
			
			mm3dot.predict(**runtime_args)
			yield PREDICT, mm3dot, frame
			
			mm3dot.drop_trackers(**runtime_args)
			yield DROP, mm3dot, frame
	yield TERMINATE, mm3dot