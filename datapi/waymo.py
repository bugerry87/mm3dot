"""
"""
# Build In
from argparse import ArgumentParser
from datetime import datetime
from glob import iglob
import json

# Installed
import numpy as np
from waymo_open_dataset.protos import metrics_pb2, submission_pb2

# Local
if __name__ != '__main__':
	from . import Frame, yaw_to_xyz


def init_waymo_arg_parser(parents=[]):
	parser = ArgumentParser(
		parents=parents,
		description='Arguments for a Waymo lib'
		)
	parser.add_argument('--metafile', metavar='JSON')
	parser.add_argument('--inputfile', metavar='PATH')
	parser.add_argument('--outputfile', metavar='PATH', default='')
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
			elif context is None:
				self.context = context = object.context_name
			elif context != object.context_name:
				self.context = context = object.context_name
				yield None #for reset!
			
			if timestamp is None:
				self.timestamp = timestamp = object.frame_timestamp_micros
			elif timestamp != object.frame_timestamp_micros:
				frame_num += 1
				if len(frame_samples):
					frame = self.metrics_to_frame(frame_samples)
					frame.frame_num = frame_num
					yield frame
				self.timestamp = timestamp = object.frame_timestamp_micros
				frame_samples.clear()
				if self.limit_frames and frame_num >= self.limit_frames:
					break
			frame_samples.append(object)
		if len(frame_samples):
			frame = self.metrics_to_frame(frame_samples)
			frame.frame_num = frame_num
			self.context = frame.context
			self.timestamp = frame.timestamp
			yield frame
		pass
	
	def metrics_to_frame(self, metrics:list):
		data = np.empty((len(metrics), self.z_dim))
		labels = [object.object.type for object in metrics]
		for i, object in enumerate(metrics):
			box = object.object.box
			data[i, self.pos_idx] = (box.center_x, box.center_y, box.center_z)[:len(self.pos_idx)]
			data[i, self.shape_idx] = (box.width, box.length, box.height)[:len(self.shape_idx)]
			data[i, self.rot_idx] = yaw_to_xyz(box.heading)
			data[i, self.score_idx] = (object.score,)[:len(self.score_idx)]
			
		frame = Frame(labels, data, self.description)
		frame.timestamp = object.frame_timestamp_micros
		frame.context = object.context_name
		return frame


class WaymoMergeLoader():
	"""
	"""
	def __init__(self, inputfile,
		frame_merge=False,
		**kwargs
		):
		"""
		"""
		self.frame_merge = frame_merge
		self.loaders = [WaymoLoader(file, **kwargs) for file in iglob(inputfile)]
		self.description = self.loaders[0].description
		self.labels = self.loaders[0].labels
		for k,v in self.description.items():
			self.__setattr__(k, v)
		pass
	
	def __len__(self):
		return np.min([len(L) for L in self.loaders])
	
	def merge_frame(self):
		timestamps = []
		frames = []
		labels = []
		data = []
		timestamp = 0
		context = None
		loaders = [iter(loader) for loader in self.loaders]
		has_next = True
		while has_next:
			has_next = False
			for loader in loaders:
				frame = next(loader, None)
				if frame:
					has_next = True
				else:
					continue
				timestamps.append(frame.timestamp)
				frames.append(frame)
				pass
			if has_next is False:
				continue
			
			indices = np.argsort(timestamps)
			for i in indices:
				frame = frames[i]
				if context is None:
					self.context = context = frame.context
				elif context != frame.context:
					yield None # for reset
				
				if timestamp is None:
					self.timestamp = timestamp = frame.timestamp
				elif timestamp != frame.timestamp:
					yield Frame(labels, np.vstack(data), loader.description)
					self.timestamp = timestamp = frame.timestamp
					labels.clear()
					data.clear()
				labels += frame.labels
				data.append(frame.data)
			if len(labels):
				self.timestamp = frame.timestamp
				self.context = frame.context
				yield Frame(labels, np.vstack(data), loader.description)
			timestamps.clear()
			frames.clear()
		pass
	
	def merge_sequential(self):
		for loader in self.loaders:
			for frame in loader:
				if frame:
					self.timestamp = frame.timestamp
					self.context = frame.context
				yield frame
	
	def __iter__(self):
		if self.frame_merge:
			return self.merge_frame()				
		else:
			return self.merge_sequential()
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
			self.metrics_only = False
		else:
			self.metrics_only = True
		
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
		Records frame to a waymo like record file.
		
		Args:
			object: An dict with detection, tracking or prediction information.
		
		Returns:
			record: The record object itself.
		"""
		self.record.inference_results.objects.append(object)
		return self.record


	def save(self, outputfile=None, metrics_only=False):
		"""
		"""
		if metrics_only or self.metrics_only:
			record = self.record.inference_results
		else:
			record = self.record
		
		if outputfile is None:
			outputfile = self.outputfile
		if outputfile is None:
			outputfile = 'checkpoint'
		outputfile += datetime.now().strftime("_%M.bin")
		with open(outputfile, 'wb') as f:
			f.write(record.SerializeToString())
		return outputfile