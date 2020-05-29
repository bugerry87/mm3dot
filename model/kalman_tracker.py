
# Installed
import numpy as np

# Local
from . import Model, TEMPLATES
from filterpy.kalman import KalmanFilter


class KalmanTracker(KalmanFilter):
	"""
	This class represents the internel state of individual tracked objects observed as bbox.
	"""
	
	def __init__(self, feature, model:Model,
				update_func=None,
				pedict_func=None,
				**kvargs
				):
		"""
		Initialises a tracker using advanced features.
		"""
		super().__init__(
			model.x_dim,
			model.z_dim,
			model.u_dim
			)
		self.label = model.label
		self.F = model.F.copy() if 'F' in model else np.eye(model.x_dim)
		self.H = model.H.copy() if 'H' in model else np.eye(model.z_dim, model.x_dim)
		self.P = model.P.copy() if 'P' in model else np.eye(model.x_dim)
		self.Q = model.Q.copy() if 'Q' in model else np.eye(model.x_dim)
		self.R = model.R.copy() if 'R' in model else np.eye(model.z_dim)
		self.alpha = model.alpha if 'alpha' in model else 1.0
		self.x[:len(self.z)] = feature[:len(self.z)][:,None]
		self.feature = feature
		self.update_func = update_func
		self.pedict_func = pedict_func
		pass
	
	def predict(self, **kvargs):
		if self.pedict_func is not None:
			self.pedict_func(self, **kvargs)
		u = kvargs['u'] if 'u' in kvargs else None
		B = kvargs['B'] if 'B' in kvargs else None
		F = kvargs['F'] if 'F' in kvargs else None
		Q = kvargs['Q'] if 'Q' in kvargs else None
		super().predict(u, B, F, Q)
		return self

	def update(self, feature, **kvargs):
		if self.update_func is not None:
			self.update_func(self, feature, **kvargs)
		R = kvargs['R'] if 'R' in kvargs else None
		H = kvargs['H'] if 'H' in kvargs else None
		super().update(feature, R, H)
		self.feature = feature
		return self	

	def save(self, filename):
		model = KalmanModel()
		np.savez(
			filename,
			x_dim = len(self.x),
			z_dim = len(self.z),
			u_dim = len(self.u),
			type=self.__class__.__name__,
			label=self.label,
			F=self.F,
			H=self.H,
			P=self.P,
			Q=self.Q,
			R=self.R,
			alpha=self.alpha
			)

TEMPLATES['KalmanTracker'] = KalmanTracker