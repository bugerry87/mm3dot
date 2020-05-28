
# Buildin
from glob import glob, iglob


def ifile(wildcards, sort=False, recursive=True):
	def sglob(wc):
		if sort:
			return sorted(glob(wc, recursive=recursive))
		else:
			return iglob(wc, recursive=recursive)

	if isinstance(wildcards, str):
		for wc in sglob(wildcards):
			yield wc
	elif isinstance(wildcards, list):
		if sort:
			wildcards = sorted(wildcards)
		for wc in wildcards:
			if any(('*?[' in c) for c in wc):
				for c in sglob(wc):
					yield c
			else:
				yield wc
	else:
		raise TypeError("wildecards must be string or list.")