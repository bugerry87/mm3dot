

# Local
from mm3dot import MM3DOT
from model import load_models
from utils import *


def init_arg_parser(parents=[]):
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description='Runs the tracking model on pre-computed detection results.',
        parents=parents
        )
    
    parser.add_argument(
        '--data_root', '-X',
        metavar='PATH',
        help='Path to the dataset root.'
        )
    parser.add_argument(
        '--dataset', '-d',
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
    '''
    Creates a FakeLoader simulating data of:
    {x, y, z, yaw, l, w, h}
    '''
    from datapi.fake import FakeLoader
    np.random.seed(0)
    t = np.eye(3) + np.random.randn(3,3) * (1 - np.eye(3)) / 100
    return FakeLoader(t, 100, 10)


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