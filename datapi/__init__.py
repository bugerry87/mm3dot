
#Installed
import numpy as np


def yaw_to_xyz(yaw):
	x = np.cos(yaw)
	y = np.sin(yaw)
	z = 0.0
	return x,y,z


def xyz_to_yaw(x, y, z=0):
	pi = np.where(x > 0.0, np.pi, -np.pi)
	with np.errstate(divide='ignore', over='ignore'):
		yaw = np.arctan(x / y) + (y < 0) * pi
	return yaw


class Frame():
	def __init__(self, 
		labels:list=[],
		data:np.ndarray=np.empty(0),
		describtion:dict=None
		):
		'''
		Init a frame of labeled features.
		
		Args:
			labels: <list(str)> The label names
			data: <np.ndarray(float)> Numerical data!
			describtion: <dict> Describes what the indices are represent, i.e.
				{'pos_idx':(0,1,2), 'vel_idx':(-3,-2,-1)}
		'''
		assert(len(labels) == len(data))
		self.labels = np.array(labels)
		self.data = data
		self.describtion = describtion
		pass
	
	def __len__(self):
		return len(self.data)
	
	def __str__(self):
		strbuild = ["{}: {}".format(label, data) for label, data in self]
		return "\n".join(strbuild)			
	
	def __iter__(self):
		return zip(self.labels, self.data)
	
	def __contains__(self, labels):
		return labels in self.labels
	
	def __getitem__(self, slc_n, slc_m=None):
		if isinstance(slc_n, str):
			if slc_n in self.labels:
				slc_n = self.labels.tolist().index(slc_n)
			else:
				raise KeyError("ERROR: Label '{}' does not exist!".format(slc_n))
		if slc_m is None:
			return self.labels[slc_n], self.data[slc_n]
		else:
			return self.labels[slc_n], self.data[slc_n, slc_m]
	
	@property
	def shape(self):
		return self.data.shape