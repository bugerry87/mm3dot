
#Installed
import numpy as np


class Features():
	def __init__(self, 
		labels:list,
		data:np.ndarray,
		describtion:dict=None
		):
		'''
		Init a set of labeled features.
		
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