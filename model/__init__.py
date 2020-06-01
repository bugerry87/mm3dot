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
			npz = np.load(npz_file)
			motion_model = str(npz['motion_model']) if 'motion_model' in npz else False
			if motion_model and motion_model in MOTION_MODELS:
				# Upgrade model
				model = MOTION_MODELS[motion_model](**kwargs)
			else:
				# Use default model
				model = Model()
		
		for k,v in npz.items():
			if '<U' in str(v.dtype):
				v = str(v)
			elif v.size == 1:
				if 'float' in str(v.dtype):
					v = float(v)
				elif 'int' in str(v.dtype):
					v = int(v)
				
			model.__setattr__(k, v)
		model.npz_file = npz_file
		return model

	def __init__(self, npz_file=None, **kwargs):
		if npz_file is not None:
			Model.load(npz_file, self)
		pass
	
	def __contains__(self, attr):
		return attr in self.__dict__
	
	def spawn(self, model, **kwargs):
		pass
	
	def update(self, model, feature, **kwargs):
		return feature, ()
	
	def predict(self, model, **kwargs):
		return ()


def load_models(ifile, models=None, **kwargs):
	if models is None:
		models = {}
	for filename in ifile:
		model = Model.load(filename, **kwargs)
		models[model.label] = model
	return models
