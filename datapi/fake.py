'''
'''
import numpy as np


class FakeLoader():
	'''
	FakeLoader, an iterable object generating fake data for testing.
	'''

	def __init__(self, transition, samples=100, framesize=10, seed=0):
		'''
		Initialize the FakeLoader
		
		Args:
			transition <np.array>: A transition matrix (N,N)
				describing the random distribution
				of the generated data.
			samples <int>: Number of samples to be generated.
				default=100
			framesize <int>: Data points per sample.
			seed <int>: Seed for pseudo random.
		'''
		self.transition = transition
		self.samples = samples
		self.framesize = framesize
		self.seed = seed
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
			fakedata <np.array>: The generated fake data.
		'''
		N = self.framesize
		M = self.transition.shape[0]
		np.random.seed(self.seed)		
		entities = np.random.randn(N,M) * 100
		for sample in range(self.samples):
			np.random.seed(self.seed + sample)
			noise = np.random.randn(N,M)
			entities = (entities + noise) @ self.transition
			yield entities
		pass


# Test
if __name__ == '__main__':
	np.random.seed(0)
	t = np.eye(3) + np.random.randn(3,3) * (1 - np.eye(3)) / 100
	print("\nTransition Matrix:")
	print(t)
	
	fakeloader = FakeLoader(t, 10, 5)
	
	print("\nSize of the FakeLoader", len(fakeloader))	
	print("\nIterate the FakeLoader:")
	for i, sample in enumerate(fakeloader):
		print("Sample number", i)
		print(sample)