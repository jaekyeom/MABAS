"""Launches routines that train (fewshot) recognition models on MiniImageNet.

Example of usage:
(1) For our proposed approach proposed:

    # 1st training stage: trains a cosine similarity based recognition model.
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifier
    # 2nd training stage: finetunes the classifier of the recognition model and
    # at the same time trains the attention based few-shot classification weight
    # generator:
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN1 # 1-shot case.
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128CosineClassifierGenWeightAttN5 # 5-shot case.

    All the configuration files that are used when launching the above
    training routines (i.e., miniImageNet_Conv128CosineClassifier.py,
    miniImageNet_Conv128CosineClassifierGenWeightAttN1.py, and
    miniImageNet_Conv128CosineClassifierGenWeightAttN5.py) are placed on the
    the directory ./config/

(2) For our implementations of the Matching Networks and Prototypical networks
    approaches:

    # Train the matching networks model for the 1-shot case.
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128MatchingNetworkN1

    # Train the matching networks model for the 5-shot case.
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128MatchingNetworkN5

    # Train the prototypical networks model for the 1-shot case.
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128PrototypicalNetworkN1

    # Train the prototypical networks model for the 5-shot case.
    CUDA_VISIBLE_DEVICES=0 python train.py --config=miniImageNet_Conv128PrototypicalNetworkN5
"""

from __future__ import print_function
import argparse
import os
import pickle
import imp
import algorithms as alg
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, default='',
    help='config file with parameters of the experiment. It is assumed that the'
         ' config file is placed on the directory ./config/.')
parser.add_argument('--parent_exp', type=str, default=None)
parser.add_argument('--checkpoint', type=int, default=0,
    help='checkpoint (epoch id) that will be loaded. If a negative value is '
         'given then the latest existing checkpoint is loaded.')
parser.add_argument('--num_workers', type=int, default=4,
    help='number of data loading workers')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
#parser.add_argument('--disp_step', type=int, default=200,
#    help='display step during training')
parser.add_argument('--disp_step', type=int, default=50,
    help='display step during training')
parser.add_argument('--use_precomputed_features', action='store_true')
parser.add_argument('--dont_load_classifier', action='store_true')
parser.add_argument('--config_overwrite', type=str, default=None)
parser.add_argument('--config_overwrite_data_train_opt', type=str, default=None)
parser.add_argument('--config_overwrite_data_test_opt', type=str, default=None)
parser.add_argument('--train_split', type=str, default='train')
parser.add_argument('--test_split', type=str, default='val')
args_opt = parser.parse_args()

orig_cwd = os.path.abspath(os.getcwd())
os.chdir('experiments')
if args_opt.parent_exp is not None and not os.path.exists(args_opt.parent_exp):
    import glob
    glob_result = glob.glob(args_opt.parent_exp)
    assert len(glob_result) == 1
    args_opt.parent_exp = glob_result[0]
os.chdir(orig_cwd)

if args_opt.config.startswith('miniImageNet_') or args_opt.config.startswith('miniImageNetBase'):
    from dataloader import MiniImageNet, FewShotDataloader
    dataset_cls = MiniImageNet
    dataset_name = 'miniImageNet'
elif args_opt.config.startswith('FC100_') or args_opt.config.startswith('FC100Base'):
    from fc100_dataloader import FC100, FewShotDataloader
    dataset_cls = FC100
    dataset_name = 'FC100'
elif args_opt.config.startswith('CIFARFS_') or args_opt.config.startswith('CIFARFSBase'):
    from cifar_fs_dataloader import CIFAR_FS, FewShotDataloader
    dataset_cls = CIFAR_FS
    dataset_name = 'CIFAR_FS'
else:
    assert False

exp_name = args_opt.config

exp_config_file = os.path.join('.', 'config', args_opt.config + '.py')
if args_opt.parent_exp is not None:
    exp_directory = os.path.join('.', 'experiments', args_opt.parent_exp, exp_name)
else:
    exp_directory = os.path.join('.', 'experiments', exp_name)
os.makedirs(exp_directory)

with open(os.path.join(exp_directory, 'args_opt.pkl'), 'wb') as f:
    pickle.dump(args_opt, f)

if args_opt.use_precomputed_features:
    assert args_opt.parent_exp is not None
    feat_par_dir = os.path.join('.', 'experiments', args_opt.parent_exp, 'features')
    feat_dir = os.path.join(feat_par_dir, dataset_name)
    assert os.path.exists(feat_dir)
    def get_pickle_paths():
        dataset_pickle_paths = dataset_cls.get_pickle_paths()
        result = dict()
        for pkl_key in dataset_pickle_paths.keys():
            feat_pkl_path = os.path.join(feat_dir, pkl_key + '.pickle')
            assert os.path.exists(feat_pkl_path)
            result[pkl_key] = feat_pkl_path
        return result
else:
    get_pickle_paths = None


git_commit = utils.get_git_commit_hash()
print('Git commit: {}'.format(git_commit))
git_diff_file_path = os.path.join(exp_directory, 'git_diff_{}.patch'.format(git_commit))
utils.save_git_diff_to_file(git_diff_file_path)


# Load the configuration params of the experiment
print('Launching experiment: %s' % exp_config_file)
config = imp.load_source("",exp_config_file).config
config['exp_dir'] = exp_directory # the place where logs, models, and other stuff will be stored
for k in config['networks']:
    if config['networks'][k]['pretrained'] is not None:
        if k == 'classifier' and args_opt.dont_load_classifier:
            config['networks'][k]['pretrained'] = None
            continue
        assert args_opt.parent_exp is not None
        parts = config['networks'][k]['pretrained'].split(os.sep, 3)
        assert len(parts) == 4
        assert parts[0] == '.'
        assert parts[1] == 'experiments'
        parts[2] = args_opt.parent_exp
        config['networks'][k]['pretrained'] = os.path.join(*parts)
print("Loading experiment %s from file: %s" % (args_opt.config, exp_config_file))
print("Generated logs, snapshots, and model files will be stored on %s" % (config['exp_dir']))

if args_opt.config_overwrite is not None:
    args_opt.config_overwrite = eval(args_opt.config_overwrite)
    #for k in args_opt.config_overwrite.keys():
    #    assert k in config
    config.update(args_opt.config_overwrite)


# Set train and test datasets and the corresponding data loaders
data_train_opt = config['data_train_opt']
data_test_opt = config['data_test_opt']

if args_opt.config_overwrite_data_train_opt is not None:
    args_opt.config_overwrite_data_train_opt = eval(args_opt.config_overwrite_data_train_opt)
    #for k in args_opt.config_overwrite_data_train_opt.keys():
    #    assert k in data_train_opt
    data_train_opt.update(args_opt.config_overwrite_data_train_opt)
if args_opt.config_overwrite_data_test_opt is not None:
    args_opt.config_overwrite_data_test_opt = eval(args_opt.config_overwrite_data_test_opt)
    #for k in args_opt.config_overwrite_data_test_opt.keys():
    #    assert k in data_test_opt
    data_test_opt.update(args_opt.config_overwrite_data_test_opt)

#train_split, test_split = 'train', 'val'
train_split, test_split = args_opt.train_split, args_opt.test_split
dataset_train = dataset_cls(phase=train_split,
                            get_pickle_paths=get_pickle_paths)
dataset_test = dataset_cls(phase=test_split,
                           get_pickle_paths=get_pickle_paths)

dloader_train = FewShotDataloader(
    dataset=dataset_train,
    nKnovel=data_train_opt['nKnovel'],
    nKbase=data_train_opt['nKbase'],
    nExemplars=data_train_opt['nExemplars'], # num training examples per novel category
    nTestNovel=data_train_opt['nTestNovel'], # num test examples for all the novel categories
    nTestBase=data_train_opt['nTestBase'], # num test examples for all the base categories
    batch_size=data_train_opt['batch_size'],
    num_workers=args_opt.num_workers,
    epoch_size=data_train_opt['epoch_size'], # num of batches per epoch
)

dloader_test = FewShotDataloader(
    dataset=dataset_test,
    nKnovel=data_test_opt['nKnovel'],
    nKbase=data_test_opt['nKbase'],
    nExemplars=data_test_opt['nExemplars'], # num training examples per novel category
    nTestNovel=data_test_opt['nTestNovel'], # num test examples for all the novel categories
    nTestBase=data_test_opt['nTestBase'], # num test examples for all the base categories
    batch_size=data_test_opt['batch_size'],
    num_workers=0,
    epoch_size=data_test_opt['epoch_size'], # num of batches per epoch
)

config['disp_step'] = args_opt.disp_step
algorithm = alg.FewShot(config)
if args_opt.cuda: # enable cuda
    algorithm.load_to_gpu()

if args_opt.checkpoint != 0: # load checkpoint
    algorithm.load_checkpoint(
        epoch=args_opt.checkpoint if (args_opt.checkpoint > 0) else '*',
        train=True)

# train the algorithm
algorithm.solve(dloader_train, dloader_test)
