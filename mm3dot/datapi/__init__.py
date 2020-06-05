# Buildin
from glob import glob, iglob

# Installed
import numpy as np


def ifile(wildcards, sort=False, recursive=True):
	def sglob(wc):
		if sort:
			return sorted(glob(wc, recursive=recursive))
		else:
			return iglob(wc, recursive=recursive)

	if isinstance(wildcards, str):
		for wc in sglob(wildcards):
			yield wc
	elif isinstance(wildcards, list):
		if sort:
			wildcards = sorted(wildcards)
		for wc in wildcards:
			if any(('*?[' in c) for c in wc):
				for c in sglob(wc):
					yield c
			else:
				yield wc
	else:
		raise TypeError("wildecards must be string or list.")


class Prototype():
	"""
	"""
	def __init__(self, data):
		"""
		"""
		if isinstance(data, dict):
			self.__dict__ = data
		elif isinstance(data, list):
			self.__list__ = data
		elif isinstance(data, Prototype):
			raise NotImplementedError()
		else:
			self.__value__ = data
	
	def __len__(self):
		if '__list__' in self.__dict__:
			return len(self.__list__)
		elif '__value__' in self.__dict__:
			return len(self.__dict__)
	
	def __str__(self):
		if '__list__' in self.__dict__:
			return 'Prototype: ' + str(self.__list__)
		elif '__value__' in self.__dict__:
			return 'Prototype: ' + str(self.__value__)
		else:
			return 'Prototype: ' + str(self.__dict__)
	
	def __iter__(self):
		if '__list__' in self.__dict__:
			return iter(self.__list__)
		elif '__value__' in self.__dict__:
			return self.__value__
		else:
			return self.__dict__.items()
	
	def __get__(self, *args):
		if '__list__' in self.__dict__:
			return self.__list__
		elif '__value__' in self.__dict__:
			return self.__value__
		else:
			return self
	
	def __getitem__(self, args):
		if '__list__' in self.__dict__:
			return [self.__list__[k] for k in args]
		elif '__value__' in self.__dict__:
			return [self.__value__[k] for k in args]
		else:
			return [self.__dict__[k] for k in args]
	
	def __contains__(self, *args):
		return args in self.__dict__
	pass


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
	
	def __getitem__(self, slc_n, slc_m=None):
		if slc_m is None:
			return self.labels[slc_n], self.data[slc_n]
		else:
			return self.labels[slc_n], self.data[slc_n, slc_m]
	
	@property
	def shape(self):
		return self.data.shape