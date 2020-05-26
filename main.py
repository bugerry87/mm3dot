
# Buildin
from glob import glob, iglob

# Local
import .model.MM3DOT


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


def run_kitti(args):
    raise NotImplementedError("KITTI is currently not supported")


def run_nusenes(args):
    raise NotImplementedError("NUSCENES is currently not supported")


def run_waymo(args):
    raise NotImplementedError("WAYMO is currently not supported")


def run_argoverse(args):
	
    pass


def main(args):
    if 'kitti' in args.dataset:
        run_kitti(args)
    elif 'nuscenes' in args.dataset:
        run_nusenes(args)
	elif 'waymo' in args.dataset:
        run_waymo(args)
    elif 'argoverse' in args.dataset:
        run_argoverse(args)
    else:
        raise ValueError("ERROR: Dataset '{}' unknown.".format(args.dataset))
    pass

if __name__ == '__main__':
    parser = init_arg_parser()
    args, _ = parser.parse_known_args()
    main(args)