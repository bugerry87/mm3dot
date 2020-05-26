
# Installed
import numpy as np

# Local
import .spatial as spatial
from .kalman_tracker import KalmanTracker


class MM3DOT():
    def __init__(self, models,
            dist_func=spatial.mahalanobis,
            assign_func=spatial.greedy_threshold,
            tracker_class=KalmanTracker,
			**kvargs
        ):
		self.__trk_id_cntr__ = 0
        self.models = models
		self.dist_func = dist_func
		self.assign_func = assign_func
		self.trackers = {}
        pass
    
	def __len__(self):
		return len(self.trackers)
	
	def __get__(self, idx):
		return self.trackers[idx]
	
	def __contains__(self, idx):
		return idx in self.trackers
	
    def spawn_trackers(self, detections, **kvargs):
        for detection in detections:
			if detection.label_class in self.models:
				model = self.models[detection.label_class]
				self.__trk_id_cntr__ += 1
				tracker = tracker_class(detection, model, **kvargs)
				tracker.id = self.__trk_id_cntr__
				tracker.lost = 0
				self.trackers[self.__trk_id_cntr__] = tracker
			else:
				print("WARNING: No model for label_class '{}'".format(detection.label_class))
        return self
	
	def predict(self, **kvargs):
		for tracker in self.trackers.values()
			tracker.predict(**kvargs)
		
	def match(self, detections, **kvargs):
		N, M = len(self.trackers), detections.shape[0]
		track_features = np.empty((N,M))
		track_covars = np.empty((N,M,M))
		track_ids = np.empty(N)
		for i, (idx, tracker) in enumerate(self.trackers.items()):
			track_features[i] = tracker.feature
			track_covars[i] = tracker.covar
			track_ids[i] = idx
		cost = self.dist_func(track_features, detections, track_covars, **kvargs)
		(trk_id, det_id), trkm, detm = assign_func(cost, **kvargs)
		return (track_ids[trk_id], detections[det_id]), track_ids[~trkm], detections[~detm]
        
    def update(self, detections, matches, lost, unmatched, **kvargs):
		# update matched trackers
		for trk_id, detection in zip(*matches):
			self.tracker[trk_id].lost = 0
			self.tracker[trk_id].update(detection, **kvargs)
		# mark unmatched trackers
		for trk_id in lost:
			self.tracker[trk_id].lost += 1
		# spawn trackers for unmatched detections
		self.spawn_trackers(unmatched)
        return self
	
	def drop_trackers(self, lost_hold = 10, **kvargs):
		for k, tracker in self.trackers.items():
			if hasattr(tracker, 'lost') and tracker.lost > lost_hold:
				del self.trackers[k]
		return self

	def run(self, data_generator, **kvargs):
		detections = next(data_generator, None)
		if detections is None:
			raise RuntimeWarning("WARNING: No data!")
		else:
			self.spawn_trackers(detections, **kvargs)
			yield 'SPAWN', self
		for detections in data_generator:
			match_results = self.match(detections, **kvargs)
			self.update(detections, *match_results, **kvargs)
			yield 'UPDATE', self
			self.drop_trackers(**kvargs)
			yield 'DROP', self
			self.predict(**kvargs)
			yield 'PREDICT', self
			