"""
"""
# Build In
from argparse import ArgumentParser

# Installed
import numpy as np
from waymo_open_dataset.protos import metrics_pb2

# Local
if __name__ != '__main__':
	from . import Features


def init_waymo_loader_parser(parents=[]):
	parser = ArgumentParser(
		parents=parents,
		description='Arguments for a WaymoLoader'
		)
	parser.add_argument('--filename', metavar='PATH')
	parser.add_argument('--pos_idx', type=int, nargs='*', metavar='TUPLE', default=(0,1,2))
	parser.add_argument('--shape_idx', type=int, nargs='*', metavar='TUPLE', default=(4,3,5))
	parser.add_argument('--rot_idx', type=int, nargs='*', metavar='TUPLE', default=(6,))
	parser.add_argument('--score_idx', type=int, nargs='*', metavar='TUPLE', default=(7,))
	parser.add_argument('--vel_idx', type=int, nargs='*', metavar='TUPLE', default=(8,9,10))
	parser.add_argument('--acl_idx', type=int, nargs='*', metavar='TUPLE', default=())
	parser.add_argument('--score_filter', type=float, metavar='FLOAT', default=0.9)
	return parser


class WaymoLoader():
	"""
	"""
	def __init__(self, filename,
		score_filter=0.0,
		limit_frames=None,
		pos_idx=(0,1,2),
		shape_idx=(4,3,5),
		rot_idx=(6,),
		score_idx=(7,),
		vel_idx=(8,9,10),
		acl_idx=(),
		**kwargs
		):
		"""
		"""
		self.metrics = metrics_pb2.Objects()
		with open(filename, 'rb') as f:
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
						print("Context changed!", context, features.context)
						exit()
					context = features.context
					yield features
				frame_samples.clear()
				if self.limit_frames and frame_num >= self.limit_frames:
					break
			frame_samples.append(object)
			timestamp = object.frame_timestamp_micros
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
			data[i, self.rot_idx] = (box.heading,)[:len(self.rot_idx)]
			data[i, self.score_idx] = (object.score,)[:len(self.score_idx)]
			
		features = Features(labels, data, self.description)
		features.timestamp = object.frame_timestamp_micros
		features.context = object.context_name
		return features


class WaymoMergeLoader():
	pass


def record(predictions, record=None):
	"""
	Records predictions to a waymo like record file.
	
	Args:
		predictions: An object with detection, tracking or prediction information.
		record: An waymo recorder object.
	
	Returns:
		record: The record object itself.
	"""
	if record is None:
		record = metrics_pb2.Objects()
	
	return record


def save_record(record, filename):
	with open(filename, 'wb') as f:
		f.write(record.SerializeToString())
	pass


def create_pd_file():
	"""
	Creates a prediction objects file.
	"""
	objects = metrics_pb2.Objects()

	o = metrics_pb2.Object()
	# The following 3 fields are used to uniquely identify a frame a prediction
	# is predicted at. Make sure you set them to values exactly the same as what
	# we provided in the raw data. Otherwise your prediction is considered as a
	# false negative.
	o.context_name = ('context_name for the prediction. See Frame::context::name '
										'in	dataset.proto.')
	# The frame timestamp for the prediction. See Frame::timestamp_micros in
	# dataset.proto.
	invalid_ts = -1
	o.frame_timestamp_micros = invalid_ts
	# This is only needed for 2D detection or tracking tasks.
	# Set it to the camera name the prediction is for.
	o.camera_name = dataset_pb2.CameraName.UNKNOWN 

	# Populating box and score.
	box = label_pb2.Label.Box()
	box.center_x = 0
	box.center_y = 0
	box.center_z = 0
	box.length = 0
	box.width = 0
	box.height = 0
	box.heading = 0
	o.object.box.CopyFrom(box)
	# This must be within [0.0, 1.0]. It is better to filter those boxes with
	# small scores to speed up metrics computation.
	o.score = 0.5
	# For tracking, this must be set and it must be unique for each tracked
	# sequence.
	o.object.id = 'unique object tracking ID'
	# Use correct type.
	o.object.type = label_pb2.Label.TYPE_PEDESTRIAN

	objects.objects.append(o)

	# Add more objects. Note that a reasonable detector should limit its maximum
	# number of boxes predicted per frame. A reasonable value is around 400. A
	# huge number of boxes can slow down metrics computation.

	# Write objects to a file.
	

# Test WaymoLoader
if __name__ == '__main__':
	from __init__ import Features
	filename = '/home/gerald/datasets/waymo/results/PointPillars/detection_3d_vehicle_detection_test.bin'
	waymoloader = WaymoLoader(filename, limit_frames=100, score_filter=0.9)
	
	print("WaymoLoader with {} detections!".format(len(waymoloader)))
	for features in waymoloader:
		print("Frame:", features.frame_num)
		print("context:", features.context)
		print("timestamp:", features.timestamp)
		print("detections:", len(features))
		print(features)
		
	