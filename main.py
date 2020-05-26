
# Buildin
from glob import glob, iglob

# Local
from . import MM3DOT
from .model import load_models


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


def init_arg_parser(parents=[]):
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Runs the tracking model on pre-computed detection results.',
        parents=parents
        )
    
    parser.add_argument(
        '--data_root', '-X'
        metavar='PATH',
        help='Path to the dataset root.'
        )
    parser.add_argument(
        '--dataset', '-d'
        metavar='NAME',
        choises=['argoverse', 'nuscenes', 'kitti'],
        help='Name of the dataset.'
        )
    parser.add_argument(
        '--subset', '-s',
        metavar='NAME',
        help='Name of the subset'
        )
    parser.add_argument(
        '--model', '-m',
        metarvar='WILDCARD',
        nargs='+',
        help='Wildcard to one or more model files.'
        )
    parser.add_argument(
        '--output', '-Y',
        metavar='PATH',
        help='Output path for tracking results'
        )
    return parser


def load_kitti(args):
    raise NotImplementedError("KITTI is currently not supported")


def load_nusenes(args):
    raise NotImplementedError("NUSCENES is currently not supported")


def load_waymo(args):
    raise NotImplementedError("WAYMO is currently not supported")


def load_argoverse(args):
	raise NotImplementedError("ARGOVERSE is currently not supported")


def load_fake(args):
	from .datapi.fake import FakeLoader
	return FakeLoader()


def main(args):
    if 'kitti' in args.dataset:
        dataloader = load_kitti(args)
    elif 'nuscenes' in args.dataset:
        dataloader = load_nusenes(args)
	elif 'waymo' in args.dataset:
        dataloader = load_waymo(args)
    elif 'argoverse' in args.dataset:
        dataloader = load_argoverse(args)
	elif 'fake' in args.dataset:
        dataloader = load_fake(args)
    else:
        raise ValueError("ERROR: Dataset '{}' unknown.".format(args.dataset))
	
	models = load_models(ifile(args.model))
	mm3dot = MM3DOT(models)
	
	for state in mm3dot.run(dataloader):
		if 'SPAWN' in state:
			pass
		elif 'PREDICT' in state:
			pass
		elif 'UPDATE' in state:
			pass
		elif 'DROP' in state:
			pass
		else:
			pass
    pass


if __name__ == '__main__':
    parser = init_arg_parser()
    args, _ = parser.parse_known_args()
    main(args)