
# Build-in
from uuid import uuid1

# Installed
import numpy as np

# Local
from . import spatial
from .model import PREDICTION_MODELS

NODATA = 'NODATA'
SPAWN = 'SPAWN'
UPDATE = 'UPDATE'
PREDICT = 'PREDICT'
DROP = 'DROP'
TERMINATE = 'TERMINATE'


class MM3DOT():
	def __init__(self, models,
			dist_func=spatial.mahalanobis,
			assign_func=spatial.hungarian,
			hold_lost=10,
			**kwargs
		):
		"""
		"""
		self.__trk_id_cntr__ = 0
		self.models = models
		self.dist_func = dist_func
		self.assign_func = assign_func
		self.trackers = {}
		self.frame_counter = 0
		self.hold_lost = hold_lost
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
					raise RuntimeWarning("Model type '{}' is not registered!".format(model.type))
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
		
	def match(self, frame, **kwargs):
		N, M = len(self.trackers), frame.shape[-1]
		track_features = np.empty((N,M))
		track_covars = np.empty((N,M,M))
		track_ids = np.empty(N)
		for i, (idx, tracker) in enumerate(self.trackers.items()):
			track_features[i] = tracker.feature
			track_covars[i] = tracker.SI
			track_ids[i] = idx
		cost = self.dist_func(track_features, frame.data, track_covars, **kwargs)
		(trk_id, det_id), trkm, detm = self.assign_func(cost, **kwargs)
		return (track_ids[trk_id], *frame[det_id]), track_ids[~trkm], frame[~detm]
		
	def update(self, frame, matches, lost, unmatched, **kwargs):
		# update matched trackers
		self.frame_counter += 1
		for trk_id, label, detection in zip(*matches):
			if self.trackers[trk_id].label != label:
				self.trackers[trk_id].label = label
				print("WARNING: Label change!")
			self.trackers[trk_id].lost = 0
			self.trackers[trk_id].age += 1
			self.trackers[trk_id].update(detection, **kwargs)
		# mark unmatched trackers
		for trk_id in lost:
			self.trackers[trk_id].lost += 1
			self.trackers[trk_id].update(None, **kwargs)
		# spawn trackers for unmatched frame
		self.spawn_trackers(zip(*unmatched))
		return self
	
	def drop_trackers(self, hold_lost=None, **kwargs):
		self.hold_lost = hold_lost if hold_lost else self.hold_lost
		victims = []
		for k, tracker in self.trackers.items():
			if tracker.lost > self.hold_lost:
				victims.append(k)
		for k in victims:
			del self.trackers[k]
		return self
	
	def reset(self):
		self.trackers.clear()

	def run(self, dataloader, **kwargs):
		data_iter = iter(dataloader)
		spawn = True
			
		for frame in data_iter:
			if frame is None:
				yield NODATA
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
			
			self.drop_trackers(**kwargs)
			yield DROP, frame
			
		yield TERMINATE
			