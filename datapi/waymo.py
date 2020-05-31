"""
"""
# Build In
from argparse import ArgumentParser
from datetime import datetime
import json

# Installed
import numpy as np
from waymo_open_dataset.protos import metrics_pb2, submission_pb2

# Local
if __name__ != '__main__':
	from . import Features, yaw_to_xyz


def init_waymo_arg_parser(parents=[]):
	parser = ArgumentParser(
		parents=parents,
		description='Arguments for a Waymo lib'
		)
	parser.add_argument('--metafile', metavar='JSON')
	parser.add_argument('--inputfile', metavar='PATH')
	parser.add_argument('--outputfile', metavar='PATH')
	parser.add_argument('--pos_idx', type=int, nargs='*', metavar='TUPLE', default=(0,1,2))
	parser.add_argument('--shape_idx', type=int, nargs='*', metavar='TUPLE', default=(4,3,5))
	parser.add_argument('--rot_idx', type=int, nargs='*', metavar='TUPLE', default=(6,7,8))
	parser.add_argument('--score_idx', type=int, nargs='*', metavar='TUPLE', default=(9,))
	parser.add_argument('--vel_idx', type=int, nargs='*', metavar='TUPLE', default=(10,11,12))
	parser.add_argument('--acl_idx', type=int, nargs='*', metavar='TUPLE', default=())
	parser.add_argument('--score_filter', type=float, metavar='FLOAT', default=0.9)
	return parser


class WaymoLoader():
	"""
	"""
	def __init__(self, inputfile,
		score_filter=0.0,
		limit_frames=None,
		pos_idx=(0,1,2),
		shape_idx=(4,3,5),
		rot_idx=(6,7,8),
		score_idx=(9,),
		vel_idx=(10,11,12),
		acl_idx=(),
		**kwargs
		):
		"""
		"""
		self.metrics = metrics_pb2.Objects()
		with open(inputfile, 'rb') as f:
			self.metrics.ParseFromString(bytearray(f.read()))
		
		self.limit_frames = limit_frames
		self.score_filter = score_filter
		self.pos_idx = pos_idx if isinstance(pos_idx, (tuple, list)) else (pos_idx,)
		self.shape_idx = shape_idx if isinstance(shape_idx, (tuple, list)) else (shape_idx,)
		self.rot_idx = rot_idx if isinstance(rot_idx, (tuple, list)) else (rot_idx,)
		self.score_idx = score_idx if isinstance(score_idx, (tuple, list)) else (score_idx,)
		self.vel_idx = vel_idx if isinstance(vel_idx, (tuple, list)) else (vel_idx,)
		self.acl_idx = acl_idx if isinstance(acl_idx, (tuple, list)) else (acl_idx,)
		self.z_dim = np.max((*self.pos_idx, *self.shape_idx, *self.rot_idx, *self.score_idx))+1
		self.x_dim = np.max((self.z_dim, *self.vel_idx, *self.acl_idx))+1
		self.labels = [0,1,2,3,4]
		self.description = {
			'pos_idx':self.pos_idx,
			'shape_idx':self.shape_idx,
			'rot_idx':self.rot_idx,
			'score_idx':self.score_idx,
			'vel_idx':self.vel_idx,
			'acl_idx':self.acl_idx,
			'x_dim':self.x_dim,
			'z_dim':self.z_dim
			}
		pass
	
	def __len__(self):
		"""
		Returns the number of samples in the dataloader.
		
		Returns: <int>
		"""
		return len(self.metrics.objects)
	
	def __iter__(self):
		timestamp = 0
		frame_samples = []
		frame_num = 0
		context = None
		for object in self.metrics.objects:
			if object.score <= self.score_filter:
				continue
			if timestamp and timestamp != object.frame_timestamp_micros:
				frame_num += 1
				if len(frame_samples):
					features = self.metrics_to_features(frame_samples)
					features.frame_num = frame_num
					if context and context != features.context:
						yield None #for reset!
					context = features.context
					self.context = context
					yield features
				frame_samples.clear()
				if self.limit_frames and frame_num >= self.limit_frames:
					break
			frame_samples.append(object)
			timestamp = object.frame_timestamp_micros
			self.timestamp = timestamp
		if len(frame_samples):
			features = self.metrics_to_features(frame_samples)
			features.frame_num = frame_num
			yield features
		pass
	
	def metrics_to_features(self, metrics:list):
		data = np.empty((len(metrics), self.z_dim))
		labels = [object.object.type for object in metrics]
		for i, object in enumerate(metrics):
			box = object.object.box
			data[i, self.pos_idx] = (box.center_x, box.center_y, box.center_z)[:len(self.pos_idx)]
			data[i, self.shape_idx] = (box.width, box.length, box.height)[:len(self.shape_idx)]
			data[i, self.rot_idx] = yaw_to_xyz(box.heading)
			data[i, self.score_idx] = (object.score,)[:len(self.score_idx)]
			
		features = Features(labels, data, self.description)
		features.timestamp = object.frame_timestamp_micros
		features.context = object.context_name
		return features


class WaymoMergeLoader():
	"""
	"""
	def __init__(self, ifile,
		frame_merge=True,
		**kwargs
		):
		"""
		"""
		self.frame_merge = frame_merge
		self.loaders = [WaymoLoader(file, **kwargs) for file in ifile]
		pass
	
	def __len__(self):
		return np.min([len(L) for L in self.loaders])
	
	def __iter__(self):
		if self.frame_merge:
			timestamp = 0
			context = None
			loaders = [iter(loader) for loader in self.loaders]
			current_frame = Features()
			next_frame = Features()
			has_next = True
			while :
				for loader in loaders:
					features = next(loader, None)
					if features:
						has_next = True
					else:
						has_next = False
						continue
					
					if timestamp and timestamp != features.timestamp:
						yield Features(labels, np.vstack(data),self.description)
						timestamp = features.timestamp
						self.timestamp = timestamp
						labels.clear()
						data.clear()
					if context and context != features.context:
						
					
		else:
			for loader in self.loaders:
				for features in loader:
					yield features
		pass
			


class WaymoRecorder():
	"""
	"""
	def __init__(self,
		outputfile=None,
		metafile=None,
		**kwargs):
		"""
		"""
		self.record = submission_pb2.Submission()
		self.outputfile = outputfile
		if metafile:
			with open(metafile, 'r') as f:
				meta = json.load(f)
			for k,v in meta.items():
				if hasattr(self.record, k):
					if isinstance(v, list):
						for value in v:
							getattr(self.record, k).append(value)
					else:
						setattr(self.record, k, v)
		
		for k,v in kwargs.items():
			if hasattr(self.record, k):
				if isinstance(v, list):
					for value in v:
						getattr(self.record, k).append(value)
				else:
					setattr(self.record, k, v)
		pass
	
	def append(self, object):
		"""
		Records features to a waymo like record file.
		
		Args:
			object: An dict with detection, tracking or prediction information.
		
		Returns:
			record: The record object itself.
		"""
		self.record.inference_results.objects.append(object)
		return self.record


	def save(self, outputfile=None):
		"""
		"""
		if outputfile:
			outputfile = self.outputfile
		if outputfile is None:
			outputfile = datetime.now().strftime("results_%Y-%m-%d_%H-%M-%S.bin")
		with open(self.outputfile, 'wb') as f:
			f.write(self.record.SerializeToString())
		return outputfile
	

# Test WaymoLoader
if __name__ == '__main__':
	from __init__ import Features, yaw_to_xyz
	filename = '/home/gerald/datasets/waymo/results/PointPillars/detection_3d_vehicle_detection_test.bin'
	waymoloader = WaymoLoader(filename, limit_frames=100, score_filter=0.9)
	
	print("WaymoLoader with {} detections!".format(len(waymoloader)))
	for features in waymoloader:
		print("Frame:", features.frame_num)
		print("context:", features.context)
		print("timestamp:", features.timestamp)
		print("detections:", len(features))
		print(features)
		
	