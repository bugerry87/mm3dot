"""
"""
# Build In
from argparse import ArgumentParser

# Installed
import numpy as np

# Local
if __name__ != '__main__':	
	from . import Features


def init_fake_loader_parser(parents=[]):
	parser = ArgumentParser(
		parents=parents,
		description='Arguments for a FakeLoader'
		)
	parser.add_argument('--x_dim', type=int, metavar='INT', default=6)
	parser.add_argument('--z_dim', type=int, metavar='INT', default=2)
	parser.add_argument('--pos_idx', type=int, nargs='*', metavar='TUPLE', default=(0,1))
	parser.add_argument('--vel_idx', type=int, nargs='*', metavar='TUPLE', default=(2,3))
	parser.add_argument('--acl_idx', type=int, nargs='*', metavar='TUPLE', default=(4,5))
	parser.add_argument('--noise', type=float, nargs='*', metavar='FLOAT', default=1)
	parser.add_argument('--samples', type=int, metavar='INT', default=100)
	parser.add_argument('--framesize', type=int, metavar='INT', default=4)
	parser.add_argument('--seed', type=int, metavar='INT', default=0)
	parser.add_argument('--labels', type=str, metavar='STR', nargs='*', default=None)
	return parser


class FakeLoader():
	"""
	FakeLoader, an iterable object generating fake data for testing.
	It simulates a kalman like prediction but with noise.
	The FakeLoader can simulate upto constant acceleration models.
	(incl. constant velocity model)
	"""
	def __init__(self,
		x_dim=2,
		z_dim=1,
		pos_idx=(0,),
		vel_idx=(1,),
		acl_idx=(),
		noise=0.1,
		samples=10,
		framesize=4,
		seed=0,
		labels=None,
		**kwargs
		):
		"""
		Initialize the FakeLoader
		
		Args:
			x_dim <int>: The assumed state size.
			z_dim <int>: The actuall feature size.
			pos_idx <tuple(int)> Indices of the position.
			vel_idx <tuple(int)> Indices of the velocity.
			acl_idx <tuple(int)> Indices of the acceleration.
			noise <float>: Noise ratio.
			samples <int>: Number of samples to be generated.
				default=100
			framesize <int>: Data points per sample.
			seed <int>: Seed for pseudo random.
			labels <list(str)>: A list of string labels.
				default=['1'...'n']
		"""
		self.z_dim = z_dim
		self.noise = noise
		self.samples = samples
		self.framesize = framesize
		self.seed = seed
		self.pos_idx = pos_idx if isinstance(pos_idx, (tuple, list)) else (pos_idx,)
		self.vel_idx = vel_idx if isinstance(vel_idx, (tuple, list)) else (vel_idx,)
		self.acl_idx = acl_idx if isinstance(acl_idx, (tuple, list)) else (acl_idx,)
		
		self.transition = np.eye(x_dim)
		if vel_idx is not None:
			self.transition[self.pos_idx, self.vel_idx] = 1
			self.transition[self.vel_idx, self.vel_idx] = 1
		if acl_idx is not None:
			self.transition[self.vel_idx, self.acl_idx] = 1
			self.transition[self.acl_idx, self.acl_idx] = 1
		
		self.labels = [str(i) for i in range(framesize)] if labels is None else labels
		self.description = {
			'pos_idx':self.pos_idx,
			'vel_idx':self.vel_idx,
			'acl_idx':self.acl_idx,
			'x_dim':self.x_dim,
			'z_dim':self.z_dim
			}
		pass
	
	def __len__(self):
		"""
		Returns the number of samples in the dataloader.
		
		Returns: <int>
		"""
		return self.samples
	
	def __iter__(self):
		"""
		Returns a generator of fake data.
		
		Yields:
			fakedata <Features>: The generated fake data.
		"""
		N = self.framesize
		M = self.x_dim
		K = len(self.labels)
		np.random.seed(self.seed)
		labels = [self.labels[i] for i in np.random.randint(0, K, K)]
		data = np.zeros((N,M))
		noise = np.random.randn(N,M) * self.noise
		data[:,:self.z_dim] = np.random.randn(N,self.z_dim) * 100
		for sample in range(self.samples):
			np.random.seed(self.seed + sample)
			noise += np.random.randn(N,M) * self.noise
			data += noise @ self.transition
			yield Features(labels, data[:,:self.z_dim])
		pass
	
	@property
	def x_dim(self):
		"""
		Returns the state size (x_dim)
		"""
		return len(self.transition)


# Test
if __name__ == '__main__':
	from __init__ import Features
	fakeloader = FakeLoader()
	print("\nTransition Matrix:")
	print(fakeloader.transition)
	
	print("\nSize of the FakeLoader", len(fakeloader))	
	print("\nIterate the FakeLoader:")
	for i, sample in enumerate(fakeloader):
		print("Sample number", i)
		print(sample)