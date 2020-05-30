'''
'''
# Build In
from argparse import ArgumentParser

# Installed
import numpy as np


PREDICTION_MODELS = {}
MOTION_MODELS = {}


class Model():
	@staticmethod
	def load(npz_file, model=None, **kwargs):
		if model is None:
			npz = np.load(npy_file)
			motion_model = npz['motion_model'] if 'motion_model' in npz else False
			if motion_model and motion_model in MOTION_MODELS:
				# Upgrade model
				model = MOTION_MODELS[motion_model](**kwargs)
			else:
				# Use default model
				model = Model()
		
		for k,v in npz:
			model.__setattr__(k, v)
		model.npz_file = npz_file
		return model

	def __init__(self, npz_file=None, **kwargs):
		if npz_file is not None:
			Model.load(npz_file, self)
		pass
	
	def __contains__(self, attr):
		return attr in self.__dict__


def load_models(ifile, models=None, **kwargs):
	if models is None:
		models = {}
	for filename in ifile:
		model = Model.load(filename, **kwargs)
		models[model.label]
	return models
