from __future__ import print_function

import os
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils

import copy
from collections import defaultdict
import functools
import gc
import itertools
import os
from pdb import set_trace as breakpoint
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt

from . import Algorithm


def top1accuracy(output, target):
    _, pred = output.max(dim=1)
    pred = pred.view(-1)
    target = target.view(-1)
    accuracy = 100 * pred.eq(target).float().mean().item()
    return accuracy


def activate_dropout_units(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.training = True

_args_barrier = object()

class FewShotFinetune(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)
        self.nKbase = torch.LongTensor()
        self.activate_dropout = (
            opt['activate_dropout'] if ('activate_dropout' in opt) else False)
        self.keep_best_model_metric_name = 'AccuracyNovel'

    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['images_train'] = torch.FloatTensor()
        self.tensors['labels_train'] = torch.LongTensor()
        self.tensors['labels_train_1hot'] = torch.FloatTensor()
        self.tensors['images_test'] = torch.FloatTensor()
        self.tensors['labels_test'] = torch.LongTensor()
        self.tensors['Kids'] = torch.LongTensor()

    def set_tensors(self, batch):
        self.nKbase = self.dloader.nKbase
        self.nKnovel = self.dloader.nKnovel

        if self.nKnovel > 0:
            train_test_stage = 'fewshot'
            images_train, labels_train, gids_support, glabels_support, images_test, labels_test, gids_query, glabels_query, K, nKbase = batch

            self.nKbase = nKbase.squeeze().item()
            self.tensors['images_train'].resize_(images_train.size()).copy_(images_train)
            self.tensors['labels_train'].resize_(labels_train.size()).copy_(labels_train)
            labels_train = self.tensors['labels_train']

            nKnovel = 1 + labels_train.max() - self.nKbase

            query_novel_filter = (labels_test >= self.nKbase)
            gids_query = gids_query[query_novel_filter].reshape(gids_support.size(0), -1)
            glabels_query = glabels_query[query_novel_filter].reshape(gids_support.size(0), -1)

            labels_train_1hot_size = list(labels_train.size()) + [nKnovel,]
            labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
            self.tensors['labels_train_1hot'].resize_(labels_train_1hot_size).fill_(0).scatter_(
                len(labels_train_1hot_size) - 1, labels_train_unsqueeze - self.nKbase, 1)
            self.tensors['images_test'].resize_(images_test.size()).copy_(images_test)
            self.tensors['labels_test'].resize_(labels_test.size()).copy_(labels_test)
            self.tensors['Kids'].resize_(K.size()).copy_(K)
        else:
            train_test_stage = 'base_classification'
            assert(len(batch) == 4)
            images_test, labels_test, K, nKbase = batch
            self.nKbase = nKbase.squeeze()[0]
            self.tensors['images_test'].resize_(images_test.size()).copy_(images_test)
            self.tensors['labels_test'].resize_(labels_test.size()).copy_(labels_test)
            self.tensors['Kids'].resize_(K.size()).copy_(K)

        return train_test_stage

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch, **finetune_opt):
        return self.process_batch(batch, do_train=False, **finetune_opt)

    def process_batch(self, batch, do_train, do_finetune=False, **finetune_opt):
        process_type = self.set_tensors(batch)

        if process_type=='fewshot':
            record = self.process_batch_fewshot_without_forgetting(
                do_train=do_train, **finetune_opt)
        elif process_type=='base_classification':
            assert not do_finetune
            record = self.process_batch_base_category_classification(
                do_train=do_train)
        else:
            raise ValueError('Unexpected process type {0}'.format(process_type))

        return record

    def process_batch_base_category_classification(self, do_train=True):
        images_test = self.tensors['images_test']
        labels_test = self.tensors['labels_test']
        Kids = self.tensors['Kids']
        nKbase = self.nKbase

        feat_model = self.networks['feat_model']
        classifier = self.networks['classifier']
        criterion  = self.criterions['loss']
        if do_train: # zero the gradients
            self.optimizers['feat_model'].zero_grad()
            self.optimizers['classifier'].zero_grad()
        #********************************************************

        #***********************************************************************
        #*********************** SET TORCH VARIABLES ***************************
        images_test_var = Variable(images_test, volatile=(not do_train))
        labels_test_var = Variable(labels_test, requires_grad=False)
        Kbase_var = (None if (nKbase==0) else Variable(
            Kids[:,:nKbase].contiguous(),requires_grad=False))
        #***********************************************************************

        loss_record  = {}
        with (utils.NoopContext if do_train else torch.no_grad)():
            #***********************************************************************
            #************************* FORWARD PHASE *******************************
            #*********** EXTRACT FEATURES FROM TRAIN & TEST IMAGES *****************
            batch_size, num_test_examples, channels, height, width = images_test.size()
            new_batch_dim = batch_size * num_test_examples
            features_test_var = feat_model(
                images_test_var.view(new_batch_dim, channels, height, width))
            features_test_var = features_test_var.view(
                [batch_size, num_test_examples,] + list(features_test_var.size()[1:]))
            #************************ APPLY CLASSIFIER *****************************
            cls_scores_var = classifier(features_test=features_test_var, Kbase_ids=Kbase_var)
            cls_scores_var = cls_scores_var.view(new_batch_dim,-1)
            labels_test_var = labels_test_var.view(new_batch_dim)
            #***********************************************************************
            #************************** COMPUTE LOSSES *****************************
            loss_cls_all = criterion(cls_scores_var, labels_test_var)
            loss_total = loss_cls_all
            loss_record['loss'] = loss_total.item()
            loss_record['AccuracyBase'] = top1accuracy(
                cls_scores_var.data, labels_test_var.data)
            #***********************************************************************

        #***********************************************************************
        #************************* BACKWARD PHASE ******************************
        if do_train:
            loss_total.backward()
            self.optimizers['feat_model'].step()
            self.optimizers['classifier'].step()
        #***********************************************************************

        return loss_record

    def process_batch_fewshot_without_forgetting(
            self, do_train=True, **finetune_opt
            ):
        assert not do_train

        finetune_result = self.finetune(**finetune_opt)
        for k in finetune_result:
            if k == 'params':
                continue
            self.finetune_data[k].append(finetune_result[k])

        images_train = self.tensors['images_train']
        labels_train = self.tensors['labels_train']
        labels_train_1hot = self.tensors['labels_train_1hot']
        images_test = self.tensors['images_test']
        labels_test = self.tensors['labels_test']
        Kids = self.tensors['Kids']
        nKbase = self.nKbase

        feat_model = self.networks['feat_model']
        classifier = self.networks['classifier']
        criterion = self.criterions['loss']

        do_train_feat_model = do_train and self.optimizers['feat_model'] is not None
        if (not do_train_feat_model):
            feat_model.eval()
            if do_train and self.activate_dropout:
                # Activate the dropout units of the feature extraction model
                # even if the feature extraction model is freezed (i.e., it is
                # in eval mode).
                activate_dropout_units(feat_model)

        if do_train: # zero the gradients
            if do_train_feat_model:
                self.optimizers['feat_model'].zero_grad()
            self.optimizers['classifier'].zero_grad()

        #***********************************************************************
        #*********************** SET TORCH VARIABLES ***************************
        is_volatile = (not do_train or not do_train_feat_model)
        images_test_var = Variable(images_test, volatile=is_volatile)
        labels_test_var = Variable(labels_test, requires_grad=False)
        Kbase_var = (None if (nKbase==0) else
            Variable(Kids[:,:nKbase].contiguous(),requires_grad=False))
        labels_train_1hot_var = Variable(labels_train_1hot, requires_grad=False)
        images_train_var = Variable(images_train, volatile=is_volatile)
        #***********************************************************************

        loss_record = {}
        with (utils.NoopContext if not is_volatile else torch.no_grad)():
            #***********************************************************************
            #************************* FORWARD PHASE: ******************************

            #************ EXTRACT FEATURES FROM TRAIN & TEST IMAGES ****************
            batch_size, num_train_examples, channels, height, width = images_train.size()
            num_test_examples = images_test.size(1)
            features_train_var = feat_model(
                images_train_var.view(batch_size * num_train_examples, channels, height, width),
                params=finetune_result['params'],
            )
            features_test_var = feat_model(
                images_test_var.view(batch_size * num_test_examples, channels, height, width),
                params=finetune_result['params'],
            )
            features_train_var = features_train_var.view(
                [batch_size, num_train_examples,] + list(features_train_var.size()[1:])
            )
            features_test_var = features_test_var.view(
                [batch_size, num_test_examples,] + list(features_test_var.size()[1:])
            )
        if (not do_train_feat_model) and do_train:
            # Make sure that no gradients are backproagated to the feature
            # extractor when the feature extraction model is freezed.
            features_train_var = Variable(features_train_var.data, volatile=False)
            features_test_var = Variable(features_test_var.data, volatile=False)
        #***********************************************************************

        with (utils.NoopContext if do_train else torch.no_grad)():
            #************************ APPLY CLASSIFIER *****************************
            if self.nKbase > 0:
                cls_scores_var = classifier(
                    features_test=features_test_var,
                    Kbase_ids=Kbase_var,
                    features_train=features_train_var,
                    labels_train=labels_train_1hot_var)
            else:
                cls_scores_var = classifier(
                    features_test=features_test_var,
                    features_train=features_train_var,
                    labels_train=labels_train_1hot_var)

            cls_scores_var = cls_scores_var.view(batch_size * num_test_examples, -1)
            labels_test_var = labels_test_var.view(batch_size * num_test_examples)
            #***********************************************************************

            #************************* COMPUTE LOSSES ******************************
            loss_cls_all = criterion(cls_scores_var, labels_test_var)
            loss_total = loss_cls_all
            loss_record['loss'] = loss_total.item()

            if self.nKbase > 0:
                loss_record['AccuracyBoth'] = top1accuracy(
                    cls_scores_var.data, labels_test_var.data)

                preds_data = cls_scores_var.data.cpu()
                labels_test_data = labels_test_var.data.cpu()
                base_ids = torch.nonzero(labels_test_data < self.nKbase).view(-1)
                novel_ids = torch.nonzero(labels_test_data >= self.nKbase).view(-1)
                preds_base = preds_data[base_ids,:]
                preds_novel = preds_data[novel_ids,:]

                loss_record['AccuracyBase'] = top1accuracy(
                    preds_base[:,:nKbase], labels_test_data[base_ids])
                loss_record['AccuracyNovel'] = top1accuracy(
                    preds_novel[:,nKbase:], (labels_test_data[novel_ids]-nKbase))
            else:
                loss_record['AccuracyNovel'] = top1accuracy(
                    cls_scores_var.data, labels_test_var.data)
            #***********************************************************************
        
        #***********************************************************************
        #************************* BACKWARD PHASE ******************************
        if do_train:
            loss_total.backward()
            if do_train_feat_model:
                self.optimizers['feat_model'].step()
            self.optimizers['classifier'].step()
        #***********************************************************************

        if (not do_train):
            if self.biter == 0: self.test_accuracies = {'AccuracyNovel':[]}
            self.test_accuracies['AccuracyNovel'].append(
                loss_record['AccuracyNovel'])
            if self.biter == (self.bnumber - 1):
                # Compute the std and the confidence interval of the accuracy of
                # the novel categories.
                stds = np.std(np.array(self.test_accuracies['AccuracyNovel']), 0)
                ci95 = 1.96*stds/np.sqrt(self.bnumber)
                loss_record['AccuracyNovel_std'] = stds
                loss_record['AccuracyNovel_cnf'] = ci95

        return loss_record

    def evaluate(self, dloader, **finetune_opt):
        self.logger.info('Evaluating (with fine-tuning): %s' % os.path.basename(self.exp_dir))

        self.dloader = dloader
        self.dataset_eval = dloader.dataset
        self.logger.info('==> Dataset: %s [%d batches]' %
                         (dloader.dataset.name, len(dloader)))
        for key, network in self.networks.items():
            network.eval()

        self.finetune_data = defaultdict(list)
        eval_stats = utils.DAverageMeter()
        self.bnumber = len(dloader)
        dloader_iterator = dloader.get_iterator_with_global_ids_and_labels()
        for idx, batch in enumerate(tqdm(dloader_iterator)):
            self.biter = idx
            eval_stats_this = self.evaluation_step(batch, **finetune_opt)
            eval_stats.update(eval_stats_this)

        self.logger.info('==> Results: %s' % eval_stats.average())

        self.on_done_evaluation()

        return eval_stats.average()

    def finetune(self, 
            # Python 2 way of forcing keyword arguments.
            dummy_arg=_args_barrier,
            finetune_objectives=None,
            finetune_n_updates=None,
            finetune_lr=None,
            finetune_optimizer=None,
            finetune_feat_model_param_filter=None,
            finetune_feat_model_no_grad_hint_mask=None):
        if dummy_arg is not _args_barrier:
            raise Exception('No positional arguments are expected')

        feat_model = self.networks['feat_model']
        classifier = self.networks['classifier']

        if finetune_feat_model_param_filter is not None:
            finetune_feat_model_param_filter = [(name in finetune_feat_model_param_filter) for (name, _) in feat_model.named_parameters()]
            print('{} parameters to fine-tune'.format(sum(finetune_feat_model_param_filter)))

        # Input data
        # {{{
        Kids = self.tensors['Kids']
        nKbase = self.nKbase

        images_test_var = Variable(self.tensors['images_test'], volatile=False)
        labels_test_var = Variable(self.tensors['labels_test'], requires_grad=False)
        Kbase_var = (None if (nKbase==0) else
            Variable(Kids[:,:nKbase].contiguous(),requires_grad=False))
        labels_train_1hot_var = Variable(self.tensors['labels_train_1hot'], requires_grad=False)
        images_train_var = Variable(self.tensors['images_train'], volatile=False)

        tasks_per_batch = images_train_var.size(0)
        n_support = images_train_var.size(1)
        n_query = images_test_var.size(1)

        labels_test_flattened = labels_test_var.view(tasks_per_batch * n_query)

        novel_ids = torch.nonzero(labels_test_flattened >= self.nKbase).view(-1)
        # }}}

        # Ensure eval mode
        feat_model_original_training_mode = feat_model.training
        classifier_original_training_mode = classifier.training
        feat_model.eval()
        classifier.eval()

        def _compute_novel_accuracy():
            support = _get_embedding(learnable_params, 'support')
            query = _get_embedding(learnable_params, 'query')
            query_scores = classifier(
                    features_test=query,
                    Kbase_ids=Kbase_var,
                    features_train=support,
                    labels_train=labels_train_1hot_var)
            query_scores = query_scores.view(tasks_per_batch * n_query, -1)
            query_scores = query_scores[novel_ids,:]
            return top1accuracy(
                    query_scores[:, nKbase:], (labels_test_flattened[novel_ids]-nKbase))
        def _compute_novel_accuracy2():
            support = _get_embedding(learnable_params, 'support')
            query = _get_embedding(learnable_params, 'query')
            out_data = dict(
                    unscaled_scores=None,
            )
            query_scores = classifier(
                    features_test=query,
                    Kbase_ids=Kbase_var,
                    features_train=support,
                    labels_train=labels_train_1hot_var,
                    out_data=out_data)
            query_scores = out_data['unscaled_scores']
            query_scores = query_scores.view(tasks_per_batch * n_query, -1)
            query_scores = query_scores[novel_ids,:]
            query_novel_labels = (labels_test_flattened[novel_ids]-nKbase)
            novel_acc = top1accuracy(
                    query_scores[:, nKbase:], query_novel_labels)

            n_query_novel = len(novel_ids)

            query_scores = query_scores[:, nKbase:].view(tasks_per_batch, n_query_novel, self.nKnovel)

            query_novel_labels_one_hot = F.one_hot(query_novel_labels, num_classes=self.nKnovel)
            query_novel_labels_one_hot_u8 = query_novel_labels_one_hot.type(torch.uint8)
            score_diffs = (
                    query_scores.masked_select(query_novel_labels_one_hot_u8).reshape(
                            tasks_per_batch, n_query_novel, 1)
                    - query_scores.masked_select(1 - query_novel_labels_one_hot_u8).reshape(
                            tasks_per_batch, n_query_novel, self.nKnovel - 1))
            qd = score_diffs.cpu().data.numpy()
            return novel_acc, qd


        # Parameters to fine-tune
        learnable_params = [p for p in feat_model.parameters()]
        if finetune_optimizer is not None:
            # {{{
            learnable_params = utils.duplicate_parameters(learnable_params)

            params_to_optimize = learnable_params
            if finetune_feat_model_param_filter is not None:
                params_to_optimize = [p for p, f in zip(params_to_optimize, finetune_feat_model_param_filter) if f]
            params_to_optimize = [
                {'params': params_to_optimize},
            ]

            optimizer_kwargs = dict(
                    lr=finetune_lr,
            )
            if finetune_optimizer.startswith('SGD'):
                if 'Nesterov' in finetune_optimizer:
                    optimizer_kwargs.update(dict(nesterov=True))
                optimizer = torch.optim.SGD(params_to_optimize, **optimizer_kwargs)
            elif finetune_optimizer.startswith('Adam'):
                optimizer = torch.optim.Adam(params_to_optimize, **optimizer_kwargs)
            else:
                assert False
            lr_scheduler = None
            if 'cosinedecay' in finetune_optimizer:
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_n_updates)
            elif 'steplr/' in finetune_optimizer:
                lr_configs = [n for n in finetune_optimizer.split('steplr/')[1].split(',')]
                step_size = int(lr_configs[0])
                gamma = float(lr_configs[1])
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            self.logger.info('finetune optimizer: {}, lr scheduler: {}'.format(optimizer, lr_scheduler))
            # }}}

        # Feature extraction and classification
        def _compute_embedding(learnable_params, data):
            return feat_model(data, params=learnable_params, no_grad_hint_mask=finetune_feat_model_no_grad_hint_mask)
        cached_emb = dict()
        cached_weight_gen_result = [None]
        def _invalidate_embedding_cache():
            cached_emb.clear()
            cached_weight_gen_result[0] = None
        def _get_embedding(learnable_params, data_name):
            # {{{
            if data_name not in cached_emb:
                with torch.enable_grad():
                    if data_name == 'support':
                        support = _compute_embedding(
                                learnable_params,
                                images_train_var.view([tasks_per_batch * n_support] + list(images_train_var.size()[-3:])))
                        support = support.view(
                                [tasks_per_batch, n_support] +
                                list(support.size()[1:]))
                        cached_emb[data_name] = support
                    elif data_name == 'query':
                        query = _compute_embedding(
                                learnable_params,
                                images_test_var.view([tasks_per_batch * n_query] + list(images_test_var.size()[-3:])))
                        query = query.view(
                                [tasks_per_batch, n_query] +
                                list(query.size()[1:]))
                        cached_emb[data_name] = query
                    else:
                        assert False
            return cached_emb[data_name]
            # }}}
        def _run_weight_generator_and_classifier(learnable_params):
            # {{{
            if cached_weight_gen_result[0] is None:
                support = _get_embedding(learnable_params, 'support')
                out_data = dict(
                        #unnormalized_cls_weights=None,  # Not needed
                        normalized_cls_weights=None,
                        unscaled_scores=None,
                )
                support_scores = classifier(
                        features_test=support,
                        Kbase_ids=Kbase_var,
                        features_train=support,
                        labels_train=labels_train_1hot_var,
                        out_data=out_data)
                cached_weight_gen_result[0] = (support_scores, out_data)
            return cached_weight_gen_result[0]
            # }}}
        def _compute_scaled_scores(learnable_params, features):
            # {{{
            _, out_data = _run_weight_generator_and_classifier(learnable_params)
            features = F.normalize(
                    features, p=2, dim=features.dim()-1, eps=1e-12)
            cls_weights = out_data['normalized_cls_weights']
            cls_scores = classifier.scale_cls * torch.baddbmm(1.0,
                    classifier.bias.view(1, 1, 1), 1.0, features, cls_weights.transpose(1,2))
            return cls_scores
            # }}}
        def _compute_unscaled_scores(learnable_params, features):
            # {{{
            _, out_data = _run_weight_generator_and_classifier(learnable_params)
            features = F.normalize(
                    features, p=2, dim=features.dim()-1, eps=1e-12)
            cls_weights = out_data['normalized_cls_weights']
            cls_scores = torch.baddbmm(1.0,
                    classifier.bias.view(1, 1, 1), 1.0, features, cls_weights.transpose(1,2))
            return cls_scores
            # }}}

        inner_losses = []
        all_inner_losses = []
        orig_correct_scores = []
        orig_score_diff = []
        query_score_diff = []
        adv_score_diff = []
        anchors = []
        attack_grad_norms = []
        attack_grad_global_norms = []

        # Fine-tuning methods
        # The standard option is:
        # AdvPushing/10
        inner_loss_funcs = dict()
        for fo_spec in finetune_objectives.split('#'):
            finetune_objective = fo_spec.split('^')[0]
            if finetune_objective.startswith('AdvPushing'):
                def _compute_inner_loss(learnable_params, finetune_objective, update_epoch):
                    # {{{
                    _, attack_step_size = finetune_objective.split('/')[:2]
                    options = finetune_objective.split('/')[2:]
                    attack_step_size = float(attack_step_size)
                    assert attack_step_size >= 0.0
                    use_scale_factor = ('UseScaleFactor' in options)

                    support = _get_embedding(learnable_params, 'support')
                    support_scores, out_data = _run_weight_generator_and_classifier(learnable_params)
                    if not use_scale_factor:
                        support_scores = out_data['unscaled_scores']
                    nKnovel = support_scores.size(-1) - nKbase

                    support_labels_one_hot = labels_train_1hot_var
                    n_target_cls = nKnovel
                    support_scores = support_scores[:, :, nKbase:]

                    assert support_scores.size(-1) == n_target_cls
                    assert support_labels_one_hot.size(-1) == n_target_cls
                    assert tuple(support_labels_one_hot.size()) == tuple(support_scores.size())
                    support_labels_one_hot_u8 = support_labels_one_hot.type(torch.uint8)
                    support_labels = support_labels_one_hot.argmax(dim=-1)

                    with torch.no_grad():
                        orig_correct_scores.append(support_scores.masked_select(support_labels_one_hot_u8).reshape(
                                tasks_per_batch, n_support).cpu().data.numpy())

                    # {{{
                    n_adv_sample = n_support * (n_target_cls - 1)
                    normalized_cls_weights = out_data['normalized_cls_weights']
                    assert tuple(normalized_cls_weights.size()) == (tasks_per_batch, nKbase + nKnovel, support.size(-1))
                    normalized_cls_weights = normalized_cls_weights[:, nKbase:]
                    assert tuple(normalized_cls_weights.size()) == (tasks_per_batch, n_target_cls, support.size(-1))
                    normalized_cls_weights = normalized_cls_weights.unsqueeze(1).expand(
                            tasks_per_batch, n_support, n_target_cls, support.size(-1))
                    normalized_weight_diffs = (
                            normalized_cls_weights.masked_select(support_labels_one_hot_u8.unsqueeze(-1)).reshape(
                                    tasks_per_batch, n_support, 1, support.size(-1))
                            - normalized_cls_weights.masked_select(1 - support_labels_one_hot_u8.unsqueeze(-1)).reshape(
                                    tasks_per_batch, n_support, n_target_cls - 1, support.size(-1)))
                    assert tuple(normalized_weight_diffs.size()) == (tasks_per_batch, n_support, n_target_cls - 1, support.size(-1))

                    # {{{
                    normalized_weight_diffs = normalized_weight_diffs.view(
                            tasks_per_batch * n_support, n_target_cls - 1, support.size(-1))

                    support_all = support.view(tasks_per_batch * n_support, -1)
                    support_norms = support_all.norm(p=2, dim=-1, keepdim=True).unsqueeze(2)
                    assert tuple(support_norms.size()) == (tasks_per_batch * n_support, 1, 1)
                    mat = (torch.eye(support_all.size(-1)).cuda().unsqueeze(0).expand(
                            tasks_per_batch * n_support, support_all.size(-1), support_all.size(-1)) / support_norms)
                    mat = mat - torch.bmm(support_all.unsqueeze(2), support_all.unsqueeze(1)) / (support_norms ** 3)
                    assert tuple(mat.size()) == (tasks_per_batch * n_support, support_all.size(-1), support_all.size(-1))

                    attack_grad = torch.bmm(normalized_weight_diffs, mat)
                    if use_scale_factor:
                        attack_grad = classifier.scale_cls * attack_grad
                    assert tuple(attack_grad.size()) == (tasks_per_batch * n_support, n_target_cls - 1, support.size(-1))
                    attack_grad = attack_grad.view(
                            tasks_per_batch, n_support, n_target_cls - 1, support.size(-1))
                    with torch.no_grad():
                        curr_attack_grad_norms_squared = (attack_grad ** 2).sum(-1)
                        curr_attack_grad_norms = torch.sqrt(curr_attack_grad_norms_squared)
                        attack_grad_norms.append(curr_attack_grad_norms.cpu().data.numpy())
                        curr_attack_grad_global_norms = torch.sqrt(curr_attack_grad_norms_squared.sum(dim=(1,2)))
                        attack_grad_global_norms.append(curr_attack_grad_global_norms.cpu().data.numpy())
                        del curr_attack_grad_norms_squared, curr_attack_grad_norms, curr_attack_grad_global_norms

                    if isinstance(attack_step_size, float):
                        adv_sample = support.unsqueeze(2) - attack_step_size * attack_grad
                    else:
                        assert False

                    if update_epoch == 0:
                        self.logger.info('attack_grad size: {}, adv_sample size: {}'.format(attack_grad.size(), adv_sample.size()))
                    # }}}

                    if isinstance(attack_step_size, float):
                        assert tuple(adv_sample.size()) == (tasks_per_batch, n_support, n_target_cls - 1, support.size(-1))

                    adv_sample = adv_sample.view(
                            tasks_per_batch, n_adv_sample, support.size(-1))

                    if update_epoch == 0:
                        self.logger.info('adv_sample.size(): {}'.format(adv_sample.size()))

                    if use_scale_factor:
                        adv_sample_scores = _compute_scaled_scores(
                                learnable_params, adv_sample)
                    else:
                        adv_sample_scores = _compute_unscaled_scores(
                                learnable_params, adv_sample)
                    assert tuple(adv_sample_scores.size()) == (tasks_per_batch, n_adv_sample, nKbase + nKnovel)
                    adv_sample_scores = adv_sample_scores[:, :, nKbase:]
                    assert tuple(adv_sample_scores.size()) == (tasks_per_batch, n_adv_sample, n_target_cls)

                    adv_sample_labels_one_hot = support_labels_one_hot.unsqueeze(2).expand(
                            tasks_per_batch, n_support, n_target_cls - 1, n_target_cls).reshape(
                                    tasks_per_batch, n_support * (n_target_cls - 1), n_target_cls)
                    assert tuple(adv_sample_labels_one_hot.size()) == (tasks_per_batch, n_adv_sample, n_target_cls)
                    adv_sample_labels_one_hot_u8 = adv_sample_labels_one_hot.type(torch.uint8)
                    adv_sample_score_diffs = (
                            adv_sample_scores.masked_select(adv_sample_labels_one_hot_u8).reshape(
                                    tasks_per_batch, n_adv_sample, 1)
                            - adv_sample_scores.masked_select(1 - adv_sample_labels_one_hot_u8).reshape(
                                    tasks_per_batch, n_adv_sample, n_target_cls - 1))

                    score_diffs = (
                            support_scores.masked_select(support_labels_one_hot_u8).reshape(
                                    tasks_per_batch, n_support, 1)
                            - support_scores.masked_select(1 - support_labels_one_hot_u8).reshape(
                                    tasks_per_batch, n_support, n_target_cls - 1))
                    orig_score_diff.append(score_diffs.cpu().data.numpy())

                    adv_sample_score_diff_argmin_mask = torch.zeros_like(
                            adv_sample_score_diffs, dtype=torch.uint8).cuda()
                    adv_sample_score_diff_argmin_mask.scatter_(
                            -1, adv_sample_score_diffs.argmin(-1, keepdim=True), 1)
                    adv_sample_score_diffs = adv_sample_score_diffs.masked_select(
                            adv_sample_score_diff_argmin_mask).view(
                                    tasks_per_batch, n_adv_sample)

                    adv_score_diff.append(adv_sample_score_diffs.cpu().data.numpy())

                    with torch.no_grad():
                        score_diffs_matched = score_diffs.unsqueeze(2).expand(
                                tasks_per_batch, n_support, n_target_cls - 1, n_target_cls - 1).reshape(
                                        tasks_per_batch, n_support * (n_target_cls - 1), n_target_cls - 1)
                        assert tuple(score_diffs_matched.size()) == (tasks_per_batch, n_adv_sample, n_target_cls - 1)

                        score_diffs_matched = score_diffs_matched.masked_select(
                                adv_sample_score_diff_argmin_mask).view(
                                        tasks_per_batch, n_adv_sample)

                    loss = - adv_sample_score_diffs

                    score_diff_means = score_diffs.mean(-1, keepdim=True)
                    curr_anchor = score_diff_means
                    curr_anchor = curr_anchor.detach()
                    curr_anchor = curr_anchor.expand(
                            tasks_per_batch, n_support, n_target_cls - 1).reshape(
                                    tasks_per_batch, n_support * (n_target_cls - 1))
                    assert tuple(curr_anchor.size()) == tuple(adv_sample_score_diffs.size())
                    curr_anchor = curr_anchor.detach()

                    if isinstance(curr_anchor, float):
                        anchors.append(curr_anchor)
                    else:
                        anchors.append(curr_anchor.cpu().data.numpy())
                    loss = F.relu(loss + curr_anchor)
                    # }}}

                    loss = loss.mean()

                    return loss
                    # }}}
                inner_loss_funcs[finetune_objective] = functools.partial(_compute_inner_loss, finetune_objective=finetune_objective)

        try:
            del _compute_inner_loss
        except NameError:
            pass

        def _compute_final_inner_loss(learnable_params, update_epoch, obj_index):
            # {{{
            all_losses = []
            loss = torch.tensor(0.0).cuda()
            splitted = finetune_objectives.split('#')
            if obj_index is not None:
                splitted = splitted[obj_index:obj_index+1]
            for fo_spec in splitted:
                fo_spec_splitted = fo_spec.split('^')
                finetune_objective = fo_spec_splitted[0]
                if len(fo_spec_splitted) >= 2:
                    fo_weight = float(fo_spec_splitted[1])
                else:
                    fo_weight = 1.0
                curr_loss = inner_loss_funcs[finetune_objective](learnable_params, update_epoch=update_epoch)
                all_losses.append(curr_loss)
                loss = loss + fo_weight * curr_loss
            return loss, all_losses
            # }}}

        novel_accuracies = []

        # Fine-tuning.
        for i in range(finetune_n_updates):
            # {{{

            if True:
                novel_accuracies.append(_compute_novel_accuracy())
            else:
                novel_acc, qd = _compute_novel_accuracy2()
                novel_accuracies.append(novel_acc)
                query_score_diff.append(qd)

            current_lr = finetune_lr
            if finetune_optimizer is not None:
                if lr_scheduler is not None:
                    lr_scheduler.step()
                for param_group in optimizer.param_groups:
                    current_lr = param_group['lr']

            obj_indices = [None]
            for obj_index in obj_indices:
                # {{{
                loss, all_losses = _compute_final_inner_loss(
                        learnable_params, update_epoch=i, obj_index=obj_index)

                inner_losses.append(loss.item())
                all_inner_losses.append([l.item() for l in all_losses])
                del all_losses

                if finetune_optimizer is not None:
                    # {{{
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    # }}}
                else:
                    # {{{
                    grad_kwargs = dict()

                    all_learnable_params = learnable_params
                    if finetune_feat_model_param_filter is not None:
                        learnable_params = [p for p, f in zip(learnable_params, finetune_feat_model_param_filter) if f]

                    grad = torch.autograd.grad(loss, learnable_params, **grad_kwargs)

                    learnable_params = list(map(lambda p: p[1] - finetune_lr * p[0], zip(grad, learnable_params)))

                    if finetune_feat_model_param_filter is not None:
                        rec_learnable_params = []
                        c = 0
                        for op, f in zip(all_learnable_params, finetune_feat_model_param_filter):
                            if f:
                                rec_learnable_params.append(learnable_params[c])
                                c += 1
                            else:
                                rec_learnable_params.append(op)
                        assert c == len(learnable_params)
                        learnable_params = rec_learnable_params
                    assert len(learnable_params) == len(all_learnable_params)
                    del all_learnable_params
                    # }}}

                _invalidate_embedding_cache()
                # }}}
            # }}}

        if True:
            novel_accuracies.append(_compute_novel_accuracy())
        else:
            novel_acc, qd = _compute_novel_accuracy2()
            novel_accuracies.append(novel_acc)
            query_score_diff.append(qd)

        with torch.no_grad():
            loss, all_losses = _compute_final_inner_loss(
                    learnable_params, update_epoch=finetune_n_updates, obj_index=None)
            inner_losses.append(loss.item())
            all_inner_losses.append([l.item() for l in all_losses])
            del all_losses

        feat_model.train(feat_model_original_training_mode)
        classifier.train(classifier_original_training_mode)

        result = dict(
                params=learnable_params,
                novel_accuracies=novel_accuracies,
                orig_correct_scores=orig_correct_scores,
                orig_score_diff=orig_score_diff,
                query_score_diff=query_score_diff,
                adv_score_diff=adv_score_diff,
                inner_losses=inner_losses,
                anchors=anchors,
                attack_grad_norms=attack_grad_norms,
                attack_grad_global_norms=attack_grad_global_norms,
        )
        all_inner_losses = np.asarray(all_inner_losses)
        for obj_index in range(all_inner_losses.shape[1]):
            result['inner_loss_{}'.format(obj_index)] = all_inner_losses[:, obj_index]
        return result

    def log_finetune_data(self, data, idx_update_dim, name):
        s = tuple([slice(None) for _ in range(idx_update_dim)])
        last_data = None
        for idx_update in range(data.shape[idx_update_dim]):
            curr_data = data[(s + (idx_update,))].reshape(-1)
            self.tb_writer.add_scalar('finetune/avg_{}'.format(name),
                                      curr_data.mean(),
                                      idx_update)
            last_data = curr_data

    def on_done_evaluation(self):
        finetune_data = copy.copy(self.finetune_data)
        novel_accuracies = np.asarray(finetune_data['novel_accuracies'])
        dims = list(range(novel_accuracies.ndim))
        dims.remove(1)
        dims = tuple(dims)
        novel_stds = np.std(novel_accuracies, axis=dims)
        novel_ci95s = 1.96 * novel_stds / np.sqrt(self.biter + 1)
        novel_ci95s = np.expand_dims(novel_ci95s, axis=0)
        finetune_data['novel_ci95s'] = novel_ci95s
        for k in finetune_data.keys():
            data = np.asarray(finetune_data[k])
            self.log_finetune_data(data, 1, k)

        self.tb_writer.flush()

        pickle_path = os.path.join(self.exp_dir, 'finetune_data_{}.pkl'.format(self.biter))
        import cPickle as pickle
        with open(pickle_path, 'wb') as f:
            pickle.dump(finetune_data, f)

        self.logger.info('finetune_data keys: {}'.format(finetune_data.keys()))



