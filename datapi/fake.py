'''
'''
# Build In
from argparse import ArgumentParser

# Installed
import numpy as np

# Local
from . import Features


def init_fake_loader_parser(parents=[]):
	parser = ArgumentParser(
		parents=parents,
		description='Arguments for a FakeLoader'
		)
	parser.add_argument('--x_dim', type=int, metavar='INT', default=2)
	parser.add_argument('--z_dim', type=int, metavar='INT', default=1)
	parser.add_argument('--pos_idx', type=int, nargs='*', metavar='TUPLE', default=(0))
	parser.add_argument('--vel_idx', type=int, nargs='*', metavar='TUPLE', default=(1))
	parser.add_argument('--acl_idx', type=int, nargs='*', metavar='TUPLE', default=())
	parser.add_argument('--noise', type=float, metavar='FLOAT', default=0.1)
	parser.add_argument('--samples', type=int, metavar='INT', default=10)
	parser.add_argument('--framesize', type=int, metavar='INT', default=4)
	parser.add_argument('--seed', type=int, metavar='INT', default=0)
	parser.add_argument('--labels', type=str, metavar='STR', nargs='*', default=None)
	return parser


class FakeLoader():
	'''
	FakeLoader, an iterable object generating fake data for testing.
	It simulates a kalman like prediction but with noise.
	The FakeLoader can simulate upto constant acceleration models.
	(incl. constant velocity model)
	'''
	def __init__(self,
		x_dim=2,
		z_dim=1,
		pos_idx=(0),
		vel_idx=(1),
		acl_idx=(),
		noise=0.1,
		samples=10,
		framesize=4,
		seed=0,
		labels=None,
		**kwargs
		):
		'''
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
		'''
		self.transition = np.eye(x_dim)
		self.transition[pos_idx, vel_idx] = 0.1
		self.transition[vel_idx, acl_idx] = 0.01
		self.z_dim = z_dim
		self.noise = noise
		self.samples = samples
		self.framesize = framesize
		self.seed = seed
		self.labels = [str(i) for i in range(framesize)] if labels is None else labels
		self.describtion = {'pos_idx':pos_idx, 'vel_idx':vel_idx, 'acl_idx':acl_idx}
		pass
	
	def __len__(self):
		'''
		Returns the number of samples in the dataloader.
		
		Returns: <int>
		'''
		return self.samples
	
	def __iter__(self):
		'''
		Returns a generator of fake data.
		
		Yields:
			fakedata <Features>: The generated fake data.
		'''
		N = self.framesize
		M = self.x_dim
		K = len(self.labels)
		np.random.seed(self.seed)
		labels = [self.labels[i] for i in np.random.randint(0, K, K)]
		data = np.random.randn(N,M) * 100
		for sample in range(self.samples):
			np.random.seed(self.seed + sample)
			noise = np.random.randn(N,M) * self.noise
			data = (data + noise) @ self.transition
			yield Features(labels, data)
		pass
	
	@property
	def x_dim(self):
		'''
		Returns the state size (x_dim)
		'''
		return len(self.transition)


# Test
if __name__ == '__main__':
	np.random.seed(0)
	t = np.eye(3) + np.random.randn(3,3) * (1 - np.eye(3)) / 100
	print("\nTransition Matrix:")
	print(t)
	
	fakeloader = FakeLoader()
	
	print("\nSize of the FakeLoader", len(fakeloader))	
	print("\nIterate the FakeLoader:")
	for i, sample in enumerate(fakeloader):
		print("Sample number", i)
		print(sample)