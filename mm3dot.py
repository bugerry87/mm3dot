
# Installed
import numpy as np

# Local
import spatial as spatial
from model import TEMPLATES


class MM3DOT():
	def __init__(self, models,
			dist_func=spatial.mahalanobis,
			assign_func=spatial.greedy_threshold,
			**kwargs
		):
		self.__trk_id_cntr__ = 0
		self.models = models
		self.dist_func = dist_func
		self.assign_func = assign_func
		self.trackers = {}
		self.frame_counter = 0
		pass
	
	def __len__(self):
		return len(self.trackers)
	
	def __get__(self, idx):
		return self.trackers[idx]
	
	def __contains__(self, idx):
		return idx in self.trackers
	
	def spawn_trackers(self, detections, **kwargs):
		for label, detection in detections:
			if label in self.models:
				model = self.models[label]
				if model.type in TEMPLATES:
					model_type = TEMPLATES[model.type]
				else:
					raise RuntimeWarning("Model type '{}' is not registered!".format(model.type))
				self.__trk_id_cntr__ += 1
				tracker = model_type(detection, model, **kwargs)
				tracker.id = self.__trk_id_cntr__
				tracker.age = 0
				tracker.lost = 0
				self.trackers[self.__trk_id_cntr__] = tracker
			else:
				print("WARNING: No model for label '{}'".format(label))
		return self
	
	def predict(self, **kwargs):
		for tracker in self.trackers.values():
			tracker.predict(**kwargs)
		
	def match(self, detections, **kwargs):
		N, M = len(self.trackers), detections.shape[-1]
		track_features = np.empty((N,M))
		track_covars = np.empty((N,M,M))
		track_ids = np.empty(N)
		for i, (idx, tracker) in enumerate(self.trackers.items()):
			track_features[i] = tracker.feature
			track_covars[i] = spatial.inv_cov(tracker.H, tracker.P, tracker.R)
			track_ids[i] = idx
		cost = self.dist_func(track_features, detections.data, track_covars, **kwargs)
		print(cost)
		(trk_id, det_id), trkm, detm = self.assign_func(cost, **kwargs)
		return (track_ids[trk_id], detections[det_id]), track_ids[~trkm], detections[~detm]
		
	def update(self, detections, matches, lost, unmatched, **kwargs):
		# update matched trackers
		self.frame_counter += 1
		for trk_id, detection in zip(*matches):
			self.tracker[trk_id].lost = 0
			self.tracker[trk_id].age += 1
			self.tracker[trk_id].update(detection, **kwargs)
		# mark unmatched trackers
		for trk_id in lost:
			self.tracker[trk_id].lost += 1
		# spawn trackers for unmatched detections
		self.spawn_trackers(unmatched)
		return self
	
	def drop_trackers(self, lost_hold = 10, **kwargs):
		for k, tracker in self.trackers.items():
			if hasattr(tracker, 'lost') and tracker.lost > lost_hold:
				del self.trackers[k]
		return self

	def run(self, dataloader, **kwargs):
		data_iter = iter(dataloader)
		detections = next(data_iter, None)
		if detections is None:
			raise RuntimeWarning("WARNING: No data!")
		else:
			self.spawn_trackers(detections, **kwargs)
			yield 'SPAWN'
		for detections in data_iter:
			self.predict(**kwargs)
			yield 'PREDICT'
			match_results = self.match(detections, **kwargs)
			self.update(detections, *match_results, **kwargs)
			yield 'UPDATE'
			self.drop_trackers(**kwargs)
			yield 'DROP'
			
			