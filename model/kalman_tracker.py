
# Installed
import numpy as np
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
            model.input_dim,
            model.output_dim,
			model.control_dim
            )
		self.label = model.label
        self.F = model.F.copy()
        self.H = model.H.copy()
        self.P = model.P.copy()
        self.Q = model.Q.copy()
        self.R = model.R.copy()
		self.alpha = model.alpha
        self.x[:] = features[:model.input_dim]
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
            type='KalmanTracker',
			label=self.label,
			F=self.F,
			H=self.H,
			P=self.P,
			Q=self.Q,
			R=self.R,
			alpha=self.alpha
			)