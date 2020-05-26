'''
'''
# Installed
import numpy as np

# Register all supported models here!!!
from .kalman_tracker import KalmanTracker


class Model():
    @staticmethod
    def load(npz_file, model=None)
        if model is None:
            model = Model()
        
        npz = np.load(npy_file)
        for k,v in npz:
            model.__setattr__(k, v)
        
        return model

    def __init__(self, npz_file=None):
        self.npz_file = npz_file
        self.type = None
		self.label = None
        self.input_dim = None
        self.output_dim = None
		self.control_dim = None
        self.F = None
        self.H = None
        self.P = None
        self.Q = None
        self.R = None
		self.alpha=1.0
        
        if npz_file is not None:
            Model.load(npz_file, self)
        pass


def load_models(ifile, models=None):
	if models is None:
		models = {}
	for filename in ifile:
		model = Model(filename)
		models[model.label]
	return models


def ContinuousVelocity()
    def __init__(self,
            pos_idx=(0,1,2),
            vel_idx=(3,4,5),
            stamp_idx=6,
            **kvargs
        ):
        self.pos_idx = pos_idx
        self.vel_idx = vel_idx
        self.stamp_idx = stamp_idx
        pass
    
        
    def update(self, state, feature, **kvargs):
        '''
        '''
        state_time = state[tstamp_idx]
        feat_time = feature[tstamp_idx]
        state_pos = state[pos_idx]
        feat_pos = feature[pos_idx]
        
        if feat_time > state_time:
            detla_time = feat_time - state_time
            state[vel_idx] = (feat_pos - state_pos) * delta_time
        else:
            raise RuntimeWarning("WARNING: Feature time is in the past of state time!")
    
    def predict(state, u, B, **kvargs):
        pass
    
