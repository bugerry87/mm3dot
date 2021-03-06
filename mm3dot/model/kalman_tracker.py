
# Installed
import numpy as np

# Local
from . import Model, PREDICTION_MODELS, MOTION_MODELS
from filterpy.kalman import KalmanFilter


class KalmanTracker(KalmanFilter):
	"""
	This class represents the internel state of individual tracked objects observed as bbox.
	"""
	def __init__(self, feature, model,**kwargs):
		"""
		Initialises a tracker using advanced features.
		"""
		super().__init__(
			model.x_dim,
			model.z_dim,
			model.u_dim
			)
		self.label = model.label
		self.F = model.F if 'F' in model else np.eye(model.x_dim)
		self.H = model.H if 'H' in model else np.eye(model.z_dim, model.x_dim)
		self.P = model.P if 'P' in model else np.eye(model.x_dim)
		self.Q = model.Q if 'Q' in model else np.eye(model.x_dim)
		self.R = model.R if 'R' in model else np.eye(model.z_dim)
		self.alpha = model.alpha if 'alpha' in model else 1.0
		self.x[:len(self.z)] = feature[:len(self.z)][:,None]
		self.feature = feature
		
		if 'motion_model' in model and model.motion_model is not None:
			self.motion_model = MOTION_MODELS[model.motion_model](**kwargs)
		else:
			self.motion_model = None
		
		if self.motion_model is not None:
			args = self.motion_model.spawn(self, **kwargs)
		pass
	
	def predict(self, **kwargs):
		if self.motion_model is not None:
			args = self.motion_model.predict(self, **kwargs)
		else:
			args = (
				kwargs['u'] if 'u' in kwargs else None,
				kwargs['B'] if 'B' in kwargs else None,
				kwargs['F'] if 'F' in kwargs else None,
				kwargs['Q'] if 'Q' in kwargs else None
				)
		n = min(len(args),4)
		super().predict(*args[:n])
		return self

	def update(self, feature, **kwargs):
		if self.motion_model is not None:
			feature, args = self.motion_model.update(self, feature, **kwargs)
		else:
			args = (
				kwargs['R'] if 'R' in kwargs else None,
				kwargs['H'] if 'H' in kwargs else None
				)
		n = min(len(args),2)
		if feature is not None:
			super().update(feature, *args[:n])
			self.feature = feature
		return self
	
	@property
	def score(self):
		return np.exp(-self.mahalanobis)

	def save(self, filename):
		np.savez(
			filename,
			x_dim = self.dim_x,
			z_dim = self.dim_z,
			u_dim = self.dim_u,
			prediction_model=self.__class__.__name__,
			motion_model=self.motion_model.__class__.__name__ if self.motion_model else None,
			label=self.label,
			F=self.F,
			H=self.H,
			P=self.P,
			Q=self.Q,
			R=self.R,
			alpha=self.alpha
			)
		return filename

PREDICTION_MODELS['KalmanTracker'] = KalmanTracker