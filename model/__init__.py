'''
'''
# Build In
from argparse import ArgumentParser

# Installed
import numpy as np


TEMPLATES = {}
INITIALIZERS = {}


class Model():
	@staticmethod
	def load(npz_file, model=None):
		if model is None:
			model = Model()
		
		npz = np.load(npy_file)
		for k,v in npz:
			model.__setattr__(k, v)
		
		return model

	def __init__(self, npz_file=None):
		self.npz_file = npz_file
		if npz_file is not None:
			Model.load(npz_file, self)
		pass
	
	def __contains__(self, attr):
		return attr in self.__dict__


def load_models(ifile, models=None):
	if models is None:
		models = {}
	for filename in ifile:
		model = Model(filename)
		models[model.label]
	return models
