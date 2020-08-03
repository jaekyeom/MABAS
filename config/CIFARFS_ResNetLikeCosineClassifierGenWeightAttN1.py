config = {}
# set the parameters related to the training and testing set

nKbase = 64 
nKnovel = 5
nExemplars = 1

base_label_ids = [0, 1, 3, 6, 9, 10, 11, 12, 13, 16, 17, 18, 20, 22, 23, 24, 25, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 41, 43, 44, 45, 49, 50, 51, 52, 53, 54, 56, 57, 59, 61, 63, 64, 65, 66, 67, 68, 72, 74, 75, 76, 78, 79, 80, 81, 82, 85, 88, 90, 91, 92, 93, 96, 97]

data_train_opt = {}
data_train_opt['nKnovel'] = nKnovel
data_train_opt['nKbase'] = -1
data_train_opt['nExemplars'] = nExemplars
data_train_opt['nTestNovel'] = nKnovel * 3
data_train_opt['nTestBase'] = nKnovel * 3 
data_train_opt['batch_size'] = 8
data_train_opt['epoch_size'] = data_train_opt['batch_size'] * 1000 

data_test_opt = {}
data_test_opt['nKnovel'] = nKnovel
data_test_opt['nKbase'] = nKbase
data_test_opt['nExemplars'] = nExemplars
data_test_opt['nTestNovel'] = 15 * data_test_opt['nKnovel']
data_test_opt['nTestBase'] = 15 * data_test_opt['nKnovel']
data_test_opt['batch_size'] = 1
data_test_opt['epoch_size'] = 2000

config['data_train_opt'] = data_train_opt
config['data_test_opt'] = data_test_opt

config['max_num_epochs'] = 60

networks = {}
net_optionsF = {'userelu': False, 'in_planes': 3, 'dropout': 0.1}
pretrainedF = './experiments/CIFARFS_ResNetLikeCosineClassifier/feat_model_net_epoch*.best'
net_optim_paramsF = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(20, 0.1),(40, 0.006),(50, 0.0012),(60, 0.00024)]}
networks['feat_model'] = {'def_file': 'architectures/ResNetLike_hybrid.py', 'pretrained': pretrainedF, 'opt': net_optionsF, 'optim_params': None}

net_optim_paramsC = {'optim_type': 'sgd', 'lr': 0.1, 'momentum':0.9, 'weight_decay': 5e-4, 'nesterov': True, 'LUT_lr':[(10*2, 0.1),(20*2, 0.006),(25*2, 0.0012),(30*2, 0.00024)]}
pretrainedC = './experiments/CIFARFS_ResNetLikeCosineClassifier/classifier_net_epoch*.best'
net_optionsC = {'classifier_type': 'cosine', 'weight_generator_type': 'attention_based', 'nKall': nKbase, 'nFeat': 512, 'scale_cls': 10, 'scale_att': 10.0, 'base_label_ids': base_label_ids}
networks['classifier'] = {'def_file': 'architectures/ClassifierWithFewShotGenerationModule.py', 'pretrained': pretrainedC, 'opt': net_optionsC, 'optim_params': net_optim_paramsC}

config['networks'] = networks

criterions = {}
criterions['loss'] = {'ctype':'CrossEntropyLoss', 'opt':None}
config['criterions'] = criterions

config['algorithm_type'] = 'FewShot'
