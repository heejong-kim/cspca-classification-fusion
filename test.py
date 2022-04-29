#!/usr/bin/env python
import os

import SimpleITK
import nibabel as nib
import numpy as np
import glob

import importlib

import pandas as pd
import torch
import yaml
from torch.autograd import Variable

from utils.loader import _load_config_yaml
from utils.trainer import get_loader, TrainerBuilder, GradCAM, FullGrad
import random
import torch.nn as nn
from sklearn.metrics import roc_auc_score, f1_score, roc_curve, confusion_matrix, accuracy_score

import matplotlib.pyplot as plt
import h5py
from scipy.stats import bootstrap

sigmoid = nn.Sigmoid()

def set_random_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def load_trained_model(config, server=False, local_model_dir=None):
    '''
    change config for analysis
    '''
    # loader no shuffle
    config['loaders']['val']['weighted_sampler'] = False
    config['loaders']['train']['weighted_sampler'] = False

    # create test config
    config['loaders']['test'] = config['loaders']['val']
    config['loaders']['dataset'] = 'prostatex_dataloader_cnn3d'
    config['loaders']['test']['channels'] = config['loaders']['val']['channels']

    config['trainer']['resume'] = None

    if server:
        model_dir = local_model_dir
    else:
        model_dir = config['trainer']['checkpoint_dir']

    # resultname = os.path.join(output_dir, f'top5-prediction-test.npy')
    # if not os.path.exists(resultname):
    if True:
        manual_seed = config.get('manual_seed', None)
        if manual_seed is not None:
            torch.manual_seed(manual_seed)
            np.random.seed(manual_seed)
            random.seed(manual_seed)
            os.environ['PYTHONHASHSEED'] = str(manual_seed)
            torch.backends.cudnn.deterministic = True  # this might make your code slow

        device_str = 'cuda:0'
        device = torch.device(device_str)
        config['device'] = device

        # best_eval = np.load(os.path.join(model_dir, 'best_evaluation.npy'))
        # best_checkpoint = np.load(os.path.join(model_dir, 'best_evaluation_iteration.npy'))

        trainer_eval = TrainerBuilder.build(config)
        trainer_eval.model.to(device)
        trainer_eval.model.eval()


        #### get testloader
        batch_size = config['loaders']['batch_size']
        demof = trainer_eval.loaders['train'].dataset.csvfname
        demo = pd.read_csv(demof)
        if 'test' in np.unique(demo['trainvaltest']):
            testloader = get_loader(config['loaders'])(csvfname=trainer_eval.loaders['train'].dataset.csvfname,
                                                       mode='test', config_mode=config['loaders']['test'])

            manual_seed = config.get('manual_seed', None)
            if manual_seed is not None:
                g = torch.Generator()
                g.manual_seed(manual_seed)
                set_random_seed(manual_seed)
                trainer_eval.loaders['test'] = torch.utils.data.DataLoader(testloader, batch_size=batch_size, shuffle=False,
                                                                      drop_last=False,
                                                                      worker_init_fn=seed_worker, generator=g)

        #### parallel
        # original saved file with DataParallel
        best_checkpoint_fname = os.path.join(model_dir,'best_checkpoint_all.pytorch')
        state = torch.load(best_checkpoint_fname, map_location='cpu')
        try:
            trainer_eval.model.load_state_dict(state['model_state_dict'])

        except:
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state['model_state_dict'].items():
                # print(k)
                name = 'module.'+k  # add `module.`
                new_state_dict[name] = v
            # load params
            trainer_eval.model.load_state_dict(new_state_dict)

        else:
            print(f'Loaded without nn.Parallel')
        finally:
            trainer_eval.model.eval()


    return trainer_eval

def get_prediction_each_modality(config_dir, confignamebase=None, num_of_seeds=5,
                                 server=False, local_model_dir=None,test_saliency=False):
    seeds_pred_val = []
    seeds_pred_test = []

    for seed in range(num_of_seeds):
        if seed == 0:
            if confignamebase == None:
                config_name = os.path.join(config_dir,
                                           f'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml')
            else:
                config_name = os.path.join(config_dir,
                                       f'{confignamebase}.yaml')
        else:
            if confignamebase == None:
                config_name = os.path.join(config_dir,
                                           f'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed{seed}.yaml')
            else:
                config_name = os.path.join(config_dir,
                                       f'{confignamebase}-seed{seed}.yaml')

        # get config
        config = _load_config_yaml(config_name)
        trainer_eval = load_trained_model(config, server=server, local_model_dir=local_model_dir)
        device = trainer_eval.device

        with torch.no_grad():
            for i, t in enumerate(trainer_eval.loaders['val']):
                print(f'Validation iteration {i}')
                # t = next(iter(trainer_eval.loaders[test_val_mode]))
                input, target = t
                input = input.to(device).float()
                target_val = target.float()
                pred = trainer_eval.model.forward(input).cpu()  # , target
                seeds_pred_val.append(sigmoid(pred).numpy())

            if 'test' in trainer_eval.loaders.keys():
                for i, t in enumerate(trainer_eval.loaders['test']):
                    print(f'Validation iteration {i}')
                    # t = next(iter(trainer_eval.loaders[test_val_mode]))
                    input, target = t
                    input = input.to(device).float()
                    target_test = target.float()
                    pred = trainer_eval.model.forward(input).cpu()  # , target
                    seeds_pred_test.append(sigmoid(pred).numpy())

        if test_saliency:
            input_grad = Variable(input.data, requires_grad=True)
            dir_prediction = os.path.join(config['trainer']['checkpoint_dir'], 'prediction')
            if not os.path.exists(dir_prediction):
                os.makedirs(dir_prediction)

            if len(glob.glob(dir_prediction+'/*')) != trainer_eval.loaders['test'].dataset:
                os.system(f'rm {dir_prediction}/*')
                pid_fid = trainer_eval.loaders['test'].dataset.csvfmode['ProxID'] + '-' + \
                          trainer_eval.loaders['test'].dataset.csvfmode['fid'].astype('str')

                try:
                    target_layers = [trainer_eval.model.convs[24]]  # last convolution
                except:
                    target_layers = [trainer_eval.model.module.convs[24]] # last convolution

                # input_tensor, target_tensor = next(iter(trainer_eval.loaders['test']))
                cam = GradCAM(model=trainer_eval.model, target_layers=target_layers, use_cuda=True)
                # cam.batch_size = config['loaders']['batch_size']

                # calculate and save images
                for ld in range(len(trainer_eval.loaders['test'].dataset)):
                    grayscale_cam = cam(input_tensor=input_grad[ld, None, :])
                    h5_saliency = os.path.join(dir_prediction, pid_fid[ld] + '.h5')
                    hfnew = h5py.File(h5_saliency,'w')
                    hfnew.create_dataset('input', data= input_grad[ld].detach().cpu().numpy().squeeze())
                    hfnew.create_dataset('gradcam', data= grayscale_cam[0])
                    hfnew.close()

            del cam
            del grayscale_cam
        del trainer_eval

    if len(seeds_pred_test) == 0:
        return seeds_pred_val, target_val
    else:
        return seeds_pred_val, seeds_pred_test, target_val, target_test

def get_prediction_file(seeds_pred, target, fnamebase, testdemo, test_n_epochs):

    seeds_pred = np.array(seeds_pred).squeeze()
    # n_epochs = trainer_eval.loaders['test'].__len__()
    if test_n_epochs > 1:
        seeds_pred_reshape = []
        for ne in range(int(len(seeds_pred)/test_n_epochs)):
            print((ne*4),(ne*4)+(test_n_epochs-1))
            seeds_pred_split = seeds_pred[(ne*4):(ne*4)+(test_n_epochs-1)+1]
            seeds_pred_concat = np.concatenate(seeds_pred_split, 0)
            seeds_pred_reshape.append(seeds_pred_concat)

        seeds_pred_reshape = np.array(seeds_pred_reshape).squeeze()
    else:
        seeds_pred_reshape = seeds_pred

    # save for challenge
    tmpname = testdemo['h5-fname'].str.split('ProstateX', expand=True)
    tmpname2 = tmpname[1].str.split('_', expand=True)
    tmpname3 = tmpname2[1].str.split('.h5', expand=True)

    prostatex_prediction = pd.DataFrame()
    # prostatex_prediction['h5-fname'] = testdemo['h5-fname']
    prostatex_prediction['ProxID'] = 'ProstateX' + tmpname2[0]
    prostatex_prediction['fid'] = tmpname3[0].str[-1]
    prostatex_prediction['ClinSig-target'] = target
    prostatex_prediction['ClinSig-ensemble'] = seeds_pred_reshape.mean(0).T
    for s in range(seeds_pred_reshape.shape[0]): # number of seeds
        prostatex_prediction[f'CilnSig-seed{s}'] = seeds_pred_reshape[s].T

    prostatex_prediction.to_csv(fnamebase + '.csv', index=False)
    return seeds_pred_reshape.mean(0).T

def load_trained_model_setouttest(config, testsetcsv):
    '''
    change config for analysis
    '''
    # loader no shuffle
    config['loaders']['val']['weighted_sampler'] = False
    config['loaders']['train']['weighted_sampler'] = False

    # create test config
    config['loaders']['test'] = config['loaders']['val']
    config['loaders']['dataset'] = 'prostatex_dataloader_cnn3d'
    config['loaders']['test']['channels'] = config['loaders']['val']['channels']

    config['trainer']['resume'] = None

    output_dir = os.path.join(config['trainer']['checkpoint_dir'], 'prediction')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_dir = config['trainer']['checkpoint_dir']
    # resultname = os.path.join(output_dir, f'top5-prediction-test.npy')
    # if not os.path.exists(resultname):
    if True:
        manual_seed = config.get('manual_seed', None)
        if manual_seed is not None:
            torch.manual_seed(manual_seed)
            np.random.seed(manual_seed)
            random.seed(manual_seed)
            os.environ['PYTHONHASHSEED'] = str(manual_seed)
            torch.backends.cudnn.deterministic = True  # this might make your code slow


        device_str = 'cuda:0'
        device = torch.device(device_str)
        config['device'] = device

        trainer_eval = TrainerBuilder.build(config)
        trainer_eval.model.to(device)
        trainer_eval.model.eval()

        #### get testloader
        batch_size = config['loaders']['batch_size']
        testloader = get_loader(config['loaders'])(csvfname=testsetcsv,
                                                   mode='test', config_mode=config['loaders']['test'])

        manual_seed = config.get('manual_seed', None)
        if manual_seed is not None:
            g = torch.Generator()
            g.manual_seed(manual_seed)
            set_random_seed(manual_seed)
            trainer_eval.loaders['test'] = torch.utils.data.DataLoader(testloader, batch_size=batch_size, shuffle=False,
                                                                  drop_last=False,
                                                                  worker_init_fn=seed_worker, generator=g)

        #### parallel
        # original saved file with DataParallel
        best_checkpoint_fname = os.path.join(model_dir,'best_checkpoint_all.pytorch')
        state = torch.load(best_checkpoint_fname, map_location='cpu')
        try:
            trainer_eval.model.load_state_dict(state['model_state_dict'])

        except:
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state['model_state_dict'].items():
                # print(k)
                name = 'module.'+k  # add `module.`
                new_state_dict[name] = v
            # load params
            trainer_eval.model.load_state_dict(new_state_dict)

        else:
            print(f'Loaded without nn.Parallel')
        finally:
            trainer_eval.model.eval()

    return trainer_eval

def get_prediction_each_modality_setouttest(config_dir, testsetcsv, confignamebase=None, num_of_seeds=5):
    seeds_pred_val = []
    seeds_pred_test = []

    for seed in range(num_of_seeds):
        if seed == 0:
            if confignamebase == None:
                config_name = os.path.join(config_dir,
                                           f'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml')
            else:
                config_name = os.path.join(config_dir,
                                       f'{confignamebase}.yaml')
        else:
            if confignamebase == None:
                config_name = os.path.join(config_dir,
                                           f'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed{seed}.yaml')
            else:
                config_name = os.path.join(config_dir,
                                       f'{confignamebase}-seed{seed}.yaml')

        # get config
        config = _load_config_yaml(config_name)
        trainer_eval = load_trained_model_setouttest(config, testsetcsv)
        device = trainer_eval.device

        with torch.no_grad():
            for i, t in enumerate(trainer_eval.loaders['val']):
                print(f'Validation iteration {i}')
                # t = next(iter(trainer_eval.loaders[test_val_mode]))
                input, target = t
                input = input.to(device).float()
                target_val = target.float()
                pred = trainer_eval.model.forward(input).cpu()  # , target
                seeds_pred_val.append(sigmoid(pred).numpy())

            for i, t in enumerate(trainer_eval.loaders['test']):
                print(f'Validation iteration {i}')
                # t = next(iter(trainer_eval.loaders[test_val_mode]))
                input, target = t
                input = input.to(device).float()
                target_test = target.float()
                pred = trainer_eval.model.forward(input).cpu()  # , target
                seeds_pred_test.append(sigmoid(pred).numpy())

    return seeds_pred_val, seeds_pred_test, target_val, target_test

def get_submission_file(seeds_pred, fnamebase, testdemo, test_n_epochs):
    if not os.path.exists(fnamebase+ '.csv'):

        seeds_pred = np.array(seeds_pred).squeeze()
        # n_epochs = trainer_eval.loaders['test'].__len__()
        seeds_pred_reshape = []
        for ne in range(int(len(seeds_pred)/test_n_epochs)):
            print((ne*4),(ne*4)+(test_n_epochs-1))
            seeds_pred_split = seeds_pred[(ne*4):(ne*4)+(test_n_epochs-1)+1]
            seeds_pred_concat = np.concatenate(seeds_pred_split, 0)
            seeds_pred_reshape.append(seeds_pred_concat)

        seeds_pred_reshape = np.array(seeds_pred_reshape).squeeze()

        # save for challenge
        tmpname = testdemo['h5-fname'].str.split('ProstateX', expand=True)
        tmpname2 = tmpname[1].str.split('_', expand=True)
        tmpname3 = tmpname2[1].str.split('.h5', expand=True)

        prostatex_submission = pd.DataFrame()
        prostatex_submission['ProxID'] = 'ProstateX' + tmpname2[0]
        prostatex_submission['fid'] = tmpname3[0].str[-1]
        prostatex_submission['ClinSig'] = seeds_pred_reshape.mean(0).T
        prostatex_submission.to_csv(fnamebase + '.csv', index=False)
        return seeds_pred_reshape.mean(0).T

    else:
        return np.array(pd.read_csv(fnamebase + '.csv')['ClinSig'])


def get_separate_submission_file(seeds_pred, fnamebase, testdemo):

    # save for challenge
    tmpname = testdemo['h5-fname'].str.split('ProstateX', expand=True)
    tmpname2 = tmpname[1].str.split('_', expand=True)
    tmpname3 = tmpname2[1].str.split('.h5', expand=True)

    prostatex_submission = pd.DataFrame()
    prostatex_submission['ProxID'] = 'ProstateX' + tmpname2[0]
    prostatex_submission['fid'] = tmpname3[0].str[-1]
    prostatex_submission['ClinSig'] = seeds_pred
    prostatex_submission.to_csv(fnamebase + '.csv', index=False)
    return

def get_auc(target, pred):
    return roc_auc_score(target, pred)

def get_specificity(target, pred):
    tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
    sensitivity = tp / (tp + fn)  # overall positives
    specificity = tn / (tn + fp)  # overall negatives
    return specificity

def get_sensitivity(target, pred):
    tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
    sensitivity = tp / (tp + fn)  # overall positives
    specificity = tn / (tn + fp)  # overall negatives
    return sensitivity

def get_accuracy(target, pred):
    return accuracy_score(target, pred)

def accuracy_with_threshold(target, prediction):
    fpr, tpr, thresholds = roc_curve(target, prediction)
    idx = np.argmax(tpr - fpr)
    return accuracy_score(target, prediction > thresholds[idx]), thresholds[idx]

def specificity_sensitivity(target, prediction):
    tn, fp, fn, tp = confusion_matrix(target, prediction).ravel()
    sensitivity = tp / (tp + fn)  # overall positives
    specificity = tn / (tn + fp)  # overall negatives
    return specificity, sensitivity

def print_all_results(target, prediction, n_trials, rng, bootstrap_subsample=None):
    acc, accthr = accuracy_with_threshold(target, prediction)
    auc = get_auc(target, prediction)
    prediction_binary = prediction > accthr
    specificity, sensitivity = specificity_sensitivity(target, prediction_binary) # requires binary

    if bootstrap_subsample == None:
        resauc = bootstrap((target, prediction), get_auc, vectorized=False, paired=True,
                        n_resamples=n_trials, random_state=rng) # requires binary

        resacc = bootstrap((target, prediction_binary), get_accuracy, vectorized=False, paired=True,
                        n_resamples=n_trials, random_state=rng) # requires binary

        resspe = bootstrap((target, prediction_binary), get_specificity, vectorized=False, paired=True,
                        n_resamples=n_trials, random_state=rng) # requires binary

        ressen = bootstrap((target, prediction_binary), get_sensitivity, vectorized=False, paired=True,
                        n_resamples=n_trials, random_state=rng) # requires binary

        print(f'AUC: {auc:.2f} CI: {resauc.confidence_interval.low:.2f} / {resauc.confidence_interval.high:.2f} \n '
              f'Acc: {acc:.2f} CI: {resacc.confidence_interval.low:.2f} / {resacc.confidence_interval.high:.2f} \n '
              f'Sen: {sensitivity:.2f} CI: {ressen.confidence_interval.low:.2f} / {ressen.confidence_interval.high:.2f} \n '
              f'Spe: {specificity:.2f} CI: {resspe.confidence_interval.low:.2f} / {resspe.confidence_interval.high:.2f} \n ')

    else:

        n_bootstraps = n_trials
        bootstrapped_scores = []

        for i in range(n_bootstraps):
            # bootstrap by sampling with replacement on the prediction indices
            indices = rng.randint(0, len(target), bootstrap_subsample) # low high size
            if len(np.unique(target[indices])) < 2:
                # We need at least one positive and one negative sample for ROC AUC
                # to be defined: reject the sample
                continue

            score = roc_auc_score(target[indices], prediction[indices])
            bootstrapped_scores.append(score)
            # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))

        # import matplotlib.pyplot as plt
        # plt.hist(bootstrapped_scores, bins=50)
        # plt.title('Histogram of the bootstrapped ROC AUC scores')
        # plt.show()

        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()

        # Computing the lower and upper bound of the 90% confidence interval
        # You can change the bounds percentiles to 0.025 and 0.975 to get
        # a 95% confidence interval instead.
        confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
        confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
        print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(
            confidence_lower, confidence_upper))

    return accthr

def ensemble_different_modality(dir_prediction, model_basename, list_of_modality):
    multimodal_ensemble = []

    for lm in list_of_modality:
        predictionfname = os.path.join(dir_prediction, model_basename + lm + '.csv')
        prediction = pd.read_csv(predictionfname)
        target = np.array(prediction['ClinSig-target'])
        multimodal_ensemble.append(np.array(prediction['ClinSig-ensemble']))

    return np.array(multimodal_ensemble).mean(0), target


def ensemble_average_saliency_map(config_dirbase, list_ensemble_saliency, confignamebase, num_of_seeds, testdemo,
                                  save_ensemble_saliency_dir, saveslice= False, save_format='.svg'):
    ensemble_name = ''
    for en in list_ensemble_saliency:
        ensemble_name += '-' + en

    ensemble_name = ensemble_name[1:]
    dir_ensemble_saliency = os.path.join(save_ensemble_saliency_dir, ensemble_name)
    if not os.path.exists(dir_ensemble_saliency):
        os.makedirs(dir_ensemble_saliency)

    # number of subjects in test demo
    pid_fid = testdemo['ProxID'] + '-' +  testdemo['fid'].astype('str')
    for pf in pid_fid:
        savefname = os.path.join(dir_ensemble_saliency, pf + '.nii.gz')
        count = 0
        if not os.path.exists(savefname):
            for modality in list_ensemble_saliency:
                if modality == 't2':
                    config_dir = f'{config_dirbase}/'  # local
                else:
                    config_dir = f'{config_dirbase}/{modality}'  # local

                for seed in range(num_of_seeds):
                    if seed == 0:
                        if confignamebase == None:
                            config_name = os.path.join(config_dir,
                                                       f'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml')
                        else:
                            config_name = os.path.join(config_dir,
                                                       f'{confignamebase}.yaml')


                    else:
                        if confignamebase == None:
                            config_name = os.path.join(config_dir,
                                                       f'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed{seed}.yaml')
                        else:
                            config_name = os.path.join(config_dir,
                                                       f'{confignamebase}-seed{seed}.yaml')

                    config = _load_config_yaml(config_name)
                    dir_prediction = os.path.join(config['trainer']['checkpoint_dir'], 'prediction')
                    saliency_fname = os.path.join(dir_prediction, pf + '.h5')

                    if count == 0:
                        hfread = h5py.File(saliency_fname, 'r')
                        input_vis = np.array(hfread['input'])
                        saliency_ensemble = np.array(hfread['gradcam'])
                        hfread.close()
                    else:
                        hfread = h5py.File(saliency_fname, 'r')
                        saliency_ensemble += np.array(hfread['gradcam'])
                        hfread.close()

                    count += 1

            assert count == (len(list_ensemble_saliency)*num_of_seeds), "Check ensemble calculation, something is missing"
            print(f'saliency ensemble of {list_ensemble_saliency}')
            saliency_ensemble = saliency_ensemble / count
            nib.save(nib.Nifti1Image(saliency_ensemble, np.eye(4)), savefname)

        else:
            # base image t2
            config_dir = f'{config_dirbase}/'  # local                 if modality == 't2':
            if confignamebase == None:
                config_name = os.path.join(config_dir,
                                           f'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml')
            else:
                config_name = os.path.join(config_dir,
                                           f'{confignamebase}.yaml')
            config = _load_config_yaml(config_name)
            dir_prediction = os.path.join(config['trainer']['checkpoint_dir'], 'prediction')
            saliency_ensemble = np.asanyarray(nib.load(savefname).dataobj)
            saliency_fname = os.path.join(dir_prediction, pf + '.h5')
            hfread = h5py.File(saliency_fname, 'r')
            input_vis = np.array(hfread['input'])
            hfread.close()

        if saveslice:
            dir_ensemble_saliency_figure = os.path.join(dir_ensemble_saliency, 'figure')
            # save figure
            if not os.path.exists(dir_ensemble_saliency_figure):
                os.makedirs(dir_ensemble_saliency_figure)

            savefigname = os.path.join(dir_ensemble_saliency_figure, pf+save_format)
            if not os.path.exists(savefigname):
                xmid,ymid,zmid = (np.array(saliency_ensemble.shape) / 2).astype('int')
                fig, arr = plt.subplots(3,3)
                if len(input_vis.shape)>3: # max channel number used for experiments are three. .transpose((1, 2, 0))
                    arr[0, 0].imshow(input_vis[:,xmid, :, :].transpose((1, 2, 0)))
                    arr[0, 1].imshow(input_vis[:,:, ymid, :].transpose((1, 2, 0)))
                    arr[0, 2].imshow(input_vis[:,:, :, zmid].transpose((1, 2, 0)))
                    #
                    arr[1, 0].imshow(saliency_ensemble[xmid, :, :], cmap=plt.get_cmap('jet'))
                    arr[1, 1].imshow(saliency_ensemble[:, ymid, :], cmap=plt.get_cmap('jet'))
                    arr[1, 2].imshow(saliency_ensemble[:, :, zmid], cmap=plt.get_cmap('jet'))
                    # overlay
                    arr[2, 0].imshow(input_vis[:,xmid, :, :].transpose((1, 2, 0)))
                    arr[2, 1].imshow(input_vis[:,:, ymid, :].transpose((1, 2, 0)))
                    arr[2, 2].imshow(input_vis[:,:, :, zmid].transpose((1, 2, 0)))
                    arr[2, 0].imshow(saliency_ensemble[xmid, :, :], cmap=plt.get_cmap('jet'), alpha = 0.5)
                    arr[2, 1].imshow(saliency_ensemble[:, ymid, :], cmap=plt.get_cmap('jet'), alpha = 0.5)
                    im = arr[2, 2].imshow(saliency_ensemble[:, :, zmid], cmap=plt.get_cmap('jet'), alpha = 0.5)

                    plt.colorbar(im, ax=arr[2,2])
                    fig.savefig(savefigname)
                    plt.close(fig)
                    plt.clf()

                else:
                    arr[0, 0].imshow(input_vis[xmid, :, :], cmap=plt.get_cmap('gist_gray'))
                    arr[0, 1].imshow(input_vis[:, ymid, :], cmap=plt.get_cmap('gist_gray'))
                    arr[0, 2].imshow(input_vis[:, :, zmid], cmap=plt.get_cmap('gist_gray'))
                    #
                    arr[1, 0].imshow(saliency_ensemble[xmid, :, :], cmap=plt.get_cmap('jet'))
                    arr[1, 1].imshow(saliency_ensemble[:, ymid, :], cmap=plt.get_cmap('jet'))
                    arr[1, 2].imshow(saliency_ensemble[:, :, zmid], cmap=plt.get_cmap('jet'))
                    # overlay
                    arr[2, 0].imshow(input_vis[xmid, :, :], cmap=plt.get_cmap('gist_gray'))
                    arr[2, 1].imshow(input_vis[:, ymid, :], cmap=plt.get_cmap('gist_gray'))
                    arr[2, 2].imshow(input_vis[:, :, zmid], cmap=plt.get_cmap('gist_gray'))
                    arr[2, 0].imshow(saliency_ensemble[xmid, :, :], cmap=plt.get_cmap('jet'), alpha = 0.5)
                    arr[2, 1].imshow(saliency_ensemble[:, ymid, :], cmap=plt.get_cmap('jet'), alpha = 0.5)
                    im = arr[2, 2].imshow(saliency_ensemble[:, :, zmid], cmap=plt.get_cmap('jet'), alpha = 0.5)

                    plt.colorbar(im, ax=arr[2,2])
                    fig.savefig(savefigname)
                    plt.close(fig)
                    plt.clf()


def ensemble_average_saliency_map_with_prediction(config_dirbase, list_ensemble_saliency, confignamebase, num_of_seeds, testdemo,
                                                  prediction_scores, save_ensemble_saliency_dir, saveslice= False, save_format='.svg'):
    ensemble_name = ''
    for en in list_ensemble_saliency:
        ensemble_name += '-' + en

    ensemble_name = ensemble_name[1:]
    dir_ensemble_saliency = os.path.join(save_ensemble_saliency_dir, ensemble_name)
    if not os.path.exists(dir_ensemble_saliency):
        os.makedirs(dir_ensemble_saliency)

    # number of subjects in test demo
    pid_fid = testdemo['ProxID'] + '-' +  testdemo['fid'].astype('str')
    for ii in range(len(pid_fid)):
        pf = pid_fid[ii]
        prediction_score = prediction_scores[ii]
        savefname = os.path.join(dir_ensemble_saliency, pf + '.nii.gz')
        count = 0
        if not os.path.exists(savefname):
            for modality in list_ensemble_saliency:
                if modality == 't2':
                    config_dir = f'{config_dirbase}/'  # local
                else:
                    config_dir = f'{config_dirbase}/{modality}'  # local

                for seed in range(num_of_seeds):
                    if seed == 0:
                        if confignamebase == None:
                            config_name = os.path.join(config_dir,
                                                       f'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml')
                        else:
                            config_name = os.path.join(config_dir,
                                                       f'{confignamebase}.yaml')


                    else:
                        if confignamebase == None:
                            config_name = os.path.join(config_dir,
                                                       f'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc-seed{seed}.yaml')
                        else:
                            config_name = os.path.join(config_dir,
                                                       f'{confignamebase}-seed{seed}.yaml')

                    config = _load_config_yaml(config_name)
                    dir_prediction = os.path.join(config['trainer']['checkpoint_dir'], 'prediction')
                    saliency_fname = os.path.join(dir_prediction, pf + '.h5')

                    if count == 0:
                        hfread = h5py.File(saliency_fname, 'r')
                        input_vis = np.array(hfread['input'])
                        saliency_ensemble = np.array(hfread['gradcam'])
                        hfread.close()
                    else:
                        hfread = h5py.File(saliency_fname, 'r')
                        saliency_ensemble += np.array(hfread['gradcam'])
                        hfread.close()

                    count += 1

            assert count == (len(list_ensemble_saliency)*num_of_seeds), "Check ensemble calculation, something is missing"
            print(f'saliency ensemble of {list_ensemble_saliency}')
            saliency_ensemble = saliency_ensemble / count
            nib.save(nib.Nifti1Image(saliency_ensemble, np.eye(4)), savefname)

        else:
            if saveslice:
                # base image t2
                config_dir = f'{config_dirbase}/'  # local                 if modality == 't2':
                if confignamebase == None:
                    config_name = os.path.join(config_dir,
                                               f'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc.yaml')
                else:
                    config_name = os.path.join(config_dir,
                                               f'{confignamebase}.yaml')
                config = _load_config_yaml(config_name)
                dir_prediction = os.path.join(config['trainer']['checkpoint_dir'], 'prediction')
                saliency_ensemble = np.asanyarray(nib.load(savefname).dataobj)
                saliency_fname = os.path.join(dir_prediction, pf + '.h5')
                hfread = h5py.File(saliency_fname, 'r')
                input_vis = np.array(hfread['input'])
                hfread.close()

        if saveslice:
            dir_ensemble_saliency_figure = os.path.join(dir_ensemble_saliency, 'figure')
            # save figure
            if not os.path.exists(dir_ensemble_saliency_figure):
                os.makedirs(dir_ensemble_saliency_figure)

            savefigname = os.path.join(dir_ensemble_saliency_figure, pf+save_format)
            if not os.path.exists(savefigname):
                xmid,ymid,zmid = (np.array(saliency_ensemble.shape) / 2).astype('int')
                fig, arr = plt.subplots(3,3)
                if len(input_vis.shape)>3: # max channel number used for experiments are three. .transpose((1, 2, 0))
                    arr[0, 0].imshow(input_vis[:,xmid, :, :].transpose((1, 2, 0)))
                    arr[0, 1].imshow(input_vis[:,:, ymid, :].transpose((1, 2, 0)))
                    arr[0, 2].imshow(input_vis[:,:, :, zmid].transpose((1, 2, 0)))
                    #
                    arr[1, 0].imshow(saliency_ensemble[xmid, :, :], cmap=plt.get_cmap('jet'))
                    arr[1, 1].imshow(saliency_ensemble[:, ymid, :], cmap=plt.get_cmap('jet'))
                    arr[1, 2].imshow(saliency_ensemble[:, :, zmid], cmap=plt.get_cmap('jet'))
                    # overlay
                    arr[2, 0].imshow(input_vis[:,xmid, :, :].transpose((1, 2, 0)))
                    arr[2, 1].imshow(input_vis[:,:, ymid, :].transpose((1, 2, 0)))
                    arr[2, 2].imshow(input_vis[:,:, :, zmid].transpose((1, 2, 0)))
                    arr[2, 0].imshow(saliency_ensemble[xmid, :, :], cmap=plt.get_cmap('jet'), alpha = 0.5)
                    arr[2, 1].imshow(saliency_ensemble[:, ymid, :], cmap=plt.get_cmap('jet'), alpha = 0.5)
                    im = arr[2, 2].imshow(saliency_ensemble[:, :, zmid], cmap=plt.get_cmap('jet'), alpha = 0.5)

                    plt.colorbar(im, ax=arr[2,2])
                    fig.suptitle(f'{pf}: {prediction_score}')
                    fig.savefig(savefigname)
                    plt.close(fig)
                    plt.clf()

                else:
                    arr[0, 0].imshow(input_vis[xmid, :, :], cmap=plt.get_cmap('gist_gray'))
                    arr[0, 1].imshow(input_vis[:, ymid, :], cmap=plt.get_cmap('gist_gray'))
                    arr[0, 2].imshow(input_vis[:, :, zmid], cmap=plt.get_cmap('gist_gray'))
                    #
                    arr[1, 0].imshow(saliency_ensemble[xmid, :, :], cmap=plt.get_cmap('jet'))
                    arr[1, 1].imshow(saliency_ensemble[:, ymid, :], cmap=plt.get_cmap('jet'))
                    arr[1, 2].imshow(saliency_ensemble[:, :, zmid], cmap=plt.get_cmap('jet'))
                    # overlay
                    arr[2, 0].imshow(input_vis[xmid, :, :], cmap=plt.get_cmap('gist_gray'))
                    arr[2, 1].imshow(input_vis[:, ymid, :], cmap=plt.get_cmap('gist_gray'))
                    arr[2, 2].imshow(input_vis[:, :, zmid], cmap=plt.get_cmap('gist_gray'))
                    arr[2, 0].imshow(saliency_ensemble[xmid, :, :], cmap=plt.get_cmap('jet'), alpha = 0.5)
                    arr[2, 1].imshow(saliency_ensemble[:, ymid, :], cmap=plt.get_cmap('jet'), alpha = 0.5)
                    im = arr[2, 2].imshow(saliency_ensemble[:, :, zmid], cmap=plt.get_cmap('jet'), alpha = 0.5)

                    plt.colorbar(im, ax=arr[2,2])
                    fig.suptitle(f'{pf}: {prediction_score}')
                    fig.savefig(savefigname)
                    plt.close(fig)
                    plt.clf()




def main():

    def save_results_test():
        '''
        1. Save results for validation / testing set
        - Single modality (t2 / ADC / DWIb800 / Ktrans)
        - Late fusion: Ensemble single modality results
        - Early fusion: 3 channel inputs
        '''

        dir_final_result = './test-results'
        dir_final_prediction = os.path.join(dir_final_result, 'prediction')
        if not os.path.exists(dir_final_prediction):
            os.makedirs(dir_final_prediction)

        # local
        testdemof = '/home/heejong/data/prostatex/demo-h5-sitk/prostateX-demo-csPCa-prediction-multimodal-sitk-newidx.csv'
        # server
        testdemof = '/share/sablab/nfs04/data/PROSTATEx/preprocessed/PROSTATEx-new/prostateX-demo-csPCa-prediction-multimodal-sitk-newidx-scratch.csv'


        testdemo = pd.read_csv(testdemof) # config['loaders']['csvfname']
        testdemo = testdemo[testdemo.trainvaltest == 'test'].reset_index()
        test_n_epochs = 1

        #### T2
        config_dirbase = './config/' # local
        confignamebase = None
        config_dir = f'{config_dirbase}/t2/' # local
        # 15 init ensemble
        seeds_pred_val_t2_15init, seeds_pred_test_t2_15init, target_val, target_test = get_prediction_each_modality(config_dir, confignamebase,num_of_seeds=15)
        _ = get_prediction_file(seeds_pred_test_t2_15init, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-t2-15init'), testdemo,
                            test_n_epochs)

        # 5 init ensemble
        seeds_pred_val_t2, seeds_pred_test_t2, target_val, target_test = get_prediction_each_modality(config_dir, confignamebase,
                                                                                  server=False, local_model_dir=None,test_saliency=True)
        _ = get_prediction_file(seeds_pred_test_t2, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-t2'), testdemo,
                            test_n_epochs)


        #### ADC
        config_dir = f'{config_dirbase}/adc/'  # local
        # 15 init ensemble
        seeds_pred_val_adc_15init, seeds_pred_test_adc_15init, target_val, target_test = get_prediction_each_modality(config_dir, confignamebase,num_of_seeds=15)
        _ = get_prediction_file(seeds_pred_test_adc_15init, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-adc-15init'), testdemo,
                            test_n_epochs)

        # 5 init ensemble
        seeds_pred_val_adc, seeds_pred_test_adc, _, _ = get_prediction_each_modality(config_dir, confignamebase,
                                                                                  server=False, local_model_dir=None,test_saliency=True)
        _ = get_prediction_file(seeds_pred_test_adc, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-adc'), testdemo,
                            test_n_epochs)

        ## Compare
        seeds_pred = np.array(seeds_pred_test_adc_15init).squeeze()
        # n_epochs = trainer_eval.loaders['test'].__len__()
        if test_n_epochs > 1:
            seeds_pred_reshape = []
            for ne in range(int(len(seeds_pred) / test_n_epochs)):
                print((ne * 4), (ne * 4) + (test_n_epochs - 1))
                seeds_pred_split = seeds_pred[(ne * 4):(ne * 4) + (test_n_epochs - 1) + 1]
                seeds_pred_concat = np.concatenate(seeds_pred_split, 0)
                seeds_pred_reshape.append(seeds_pred_concat)

            seeds_pred_reshape = np.array(seeds_pred_reshape).squeeze()
        else:
            seeds_pred_reshape = seeds_pred

        rng_seed = 42  # control reproducibility
        rng = np.random.RandomState(rng_seed)
        # rng = np.random.default_rng()
        n_trials = 10000

        dir_prediction = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/'
        model_basename = '8conv4mpfcf4-earlystopping-eval-'

        dir_prediction = './test-results/prediction/'
        model_basename = '8conv4mpfcf4-earlystopping-eval-'

        chan1_adc, target = ensemble_different_modality(dir_prediction, model_basename, ['adc-15init'])
        print_all_results(target, chan1_adc, n_trials, rng)
        print_all_results(target, seeds_pred_reshape.mean(0), n_trials, rng)


    config_dir = f'{config_dirbase}/ktrans/'  # local
        # 15 init ensemble
        seeds_pred_val_ktrans_15init, seeds_pred_test_ktrans_15init, target_val, target_test = get_prediction_each_modality(config_dir, confignamebase,num_of_seeds=15)
        _ = get_prediction_file(seeds_pred_test_ktrans_15init, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-ktrans-15init'), testdemo,
                            test_n_epochs)
        # 5 init ensemble
        seeds_pred_val_ktrans, seeds_pred_test_ktrans, _, _ = get_prediction_each_modality(config_dir, confignamebase,
                                                                                  server=False, local_model_dir=None,test_saliency=True)
        _ = get_prediction_file(seeds_pred_test_ktrans, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-ktrans'), testdemo,
                            test_n_epochs)



        config_dir = f'{config_dirbase}/dwib800/'  # local
        # 15 init ensemble
        seeds_pred_val_dwib800_15init, seeds_pred_test_dwib800_15init, target_val, target_test = get_prediction_each_modality(config_dir, confignamebase,num_of_seeds=15)
        _ = get_prediction_file(seeds_pred_test_dwib800_15init, np.array(target_test),
                                os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-dwib800-15init'), testdemo,
                                test_n_epochs)
        # 5 init ensemble
        seeds_pred_val_dwib800, seeds_pred_test_dwib800, _, _ = get_prediction_each_modality(config_dir, confignamebase,
                                                                                  server=False, local_model_dir=None,test_saliency=True)

        _ = get_prediction_file(seeds_pred_test_dwib800, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-dwib800'), testdemo,
                            test_n_epochs)



        config_dir = f'{config_dirbase}/adc-unregistered/'  # local
        # 15 init ensemble
        seeds_pred_val_adc_unregistered_15init, seeds_pred_test_adc_unregistered_15init, target_val, target_test = get_prediction_each_modality(config_dir, confignamebase,num_of_seeds=15)
        _ = get_prediction_file(seeds_pred_test_adc_unregistered_15init, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-adc-unregistered-15init'), testdemo,
                            test_n_epochs)

        # 5 init ensemble
        seeds_pred_val_adc_unregistered, seeds_pred_test_adc_unregistered, _, _ = get_prediction_each_modality(config_dir, confignamebase,
                                                                                  server=False, local_model_dir=None,test_saliency=True)
        _ = get_prediction_file(seeds_pred_test_adc_unregistered, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-adc-unregistered'), testdemo,
                            test_n_epochs)


        config_dir = f'{config_dirbase}/dwib800-unregistered/'  # local
        # 15 init ensemble
        seeds_pred_val_dwib800_unregistered_15init, seeds_pred_test_dwib800_unregistered_15init, target_val, target_test = get_prediction_each_modality(config_dir, confignamebase,num_of_seeds=15)
        _ = get_prediction_file(seeds_pred_test_dwib800_unregistered_15init, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-dwib800-unregistered-15init'), testdemo,
                            test_n_epochs)

        # 5 init ensemble
        seeds_pred_val_dwib800_unregistered, seeds_pred_test_dwib800_unregistered, _, _ = get_prediction_each_modality(config_dir, confignamebase,
                                                                                  server=False, local_model_dir=None,test_saliency=True)
        _ = get_prediction_file(seeds_pred_test_dwib800_unregistered, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-dwib800-unregistered'), testdemo,
                            test_n_epochs)



        # 3 channel input with registration
        config_dir = f'{config_dirbase}/t2-adc-ktrans/comparison-f7/' # local
        confignamebase = 'augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        seeds_pred_val_3channel_ktrans, seeds_pred_test_3channel_ktrans, _, _ = get_prediction_each_modality(
                                                                                    config_dir,confignamebase, num_of_seeds=15,
                                                                                    server=False, local_model_dir=None,test_saliency=True)
        _ = get_prediction_file(seeds_pred_test_3channel_ktrans, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-3channel-t2-adc-ktrans'), testdemo,
                            test_n_epochs)

        config_dir = f'{config_dirbase}/t2-adc-dwib800/comparison-f7/' # local
        confignamebase = 'augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc'

        seeds_pred_val_3channel_dwib800, seeds_pred_test_3channel_dwib800, _, _ = get_prediction_each_modality(
                                                                                    config_dir,confignamebase, num_of_seeds=15,
                                                                                    server=False, local_model_dir=None,test_saliency=True)

        _ = get_prediction_file(seeds_pred_test_3channel_dwib800, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-3channel-t2-adc-dwib800'), testdemo,
                            test_n_epochs)


        # 3 channel input without registration
        config_dir = f'{config_dirbase}/t2-adc-ktrans-unregistered/comparison-f7/' # local
        confignamebase = 'augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        seeds_pred_val_3channel_ktrans_unregistered, seeds_pred_test_3channel_ktrans_unregistered, _, _ = get_prediction_each_modality(
                                                                                    config_dir,confignamebase, num_of_seeds=15,
                                                                                    server=False, local_model_dir=None,test_saliency=True)
        _ = get_prediction_file(seeds_pred_test_3channel_ktrans_unregistered, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-3channel-t2-adc-ktrans-unregistered'), testdemo,
                            test_n_epochs)

        config_dir = f'{config_dirbase}/t2-adc-dwib800-unregistered/comparison-f7/' # local
        confignamebase = 'augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc'

        seeds_pred_val_3channel_dwib800_unregistered, seeds_pred_test_3channel_dwib800_unregistered, _, _ = get_prediction_each_modality(
                                                                                    config_dir,confignamebase, num_of_seeds=15,
                                                                                    server=False, local_model_dir=None,test_saliency=True)

        _ = get_prediction_file(seeds_pred_test_3channel_dwib800_unregistered, np.array(target_test),
                            os.path.join(dir_final_prediction, '8conv4mpfcf4-earlystopping-eval-3channel-t2-adc-dwib800-unregistered'), testdemo,
                            test_n_epochs)


        ################
        ## Saliency map
        ################
        ## Saliency map ensemble
        dir_final_saliency = os.path.join(dir_final_result, 'saliency')
        if not os.path.exists(dir_final_saliency):
            os.makedirs(dir_final_saliency)

        num_of_seeds = 5
        testdemof = '/home/heejong/data/prostatex/demo-h5-sitk/prostateX-demo-csPCa-prediction-multimodal-sitk-newidx.csv'
        testdemo = pd.read_csv(testdemof) # config['loaders']['csvfname']
        testdemo = testdemo[testdemo.trainvaltest == 'test'].reset_index()

        save_ensemble_saliency_dir = dir_final_saliency + '-1channel'
        config_dirbase = './config-prostatex-local-8conv4mpfcf4-earlystopping-eval' # local
        confignamebase = None
        list_ensemble_saliency = ['t2', 'adc-unregistered', 'dwib800-unregistered']
        ensemble_average_saliency_map(config_dirbase, list_ensemble_saliency, confignamebase,num_of_seeds, testdemo,
                                      save_ensemble_saliency_dir, saveslice=False, save_format='.png')
        # ensemble_average_saliency_map_with_prediction(config_dirbase, list_ensemble_saliency, confignamebase,
        #                                               num_of_seeds, testdemo,
        #                                               chan1_t2_adc_dwib800, save_ensemble_saliency_dir,saveslice=False,
        #                                               save_format='.png')

        list_ensemble_saliency = ['t2', 'adc-unregistered', 'ktrans']
        ensemble_average_saliency_map(config_dirbase, list_ensemble_saliency, confignamebase,num_of_seeds, testdemo,
                                      save_ensemble_saliency_dir)

        list_ensemble_saliency = ['t2']
        ensemble_average_saliency_map(config_dirbase, list_ensemble_saliency, confignamebase,num_of_seeds, testdemo,
                                      save_ensemble_saliency_dir)

        list_ensemble_saliency = ['adc-unregistered']
        ensemble_average_saliency_map(config_dirbase, list_ensemble_saliency, confignamebase,num_of_seeds, testdemo,
                                      save_ensemble_saliency_dir)

        list_ensemble_saliency = ['dwib800-unregistered']
        ensemble_average_saliency_map(config_dirbase, list_ensemble_saliency, confignamebase,num_of_seeds, testdemo,
                                      save_ensemble_saliency_dir)

        list_ensemble_saliency = ['ktrans']
        ensemble_average_saliency_map(config_dirbase, list_ensemble_saliency, confignamebase,num_of_seeds, testdemo,
                                      save_ensemble_saliency_dir)


        num_of_seeds = 15
        list_ensemble_saliency = ['t2-adc-dwib800']
        save_ensemble_saliency_dir = dir_final_saliency + '-3channel'
        confignamebase = 'comparison-f7/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        ensemble_average_saliency_map(config_dirbase, list_ensemble_saliency, confignamebase, num_of_seeds, testdemo,
                                      save_ensemble_saliency_dir)

        list_ensemble_saliency = ['t2-adc-ktrans']
        save_ensemble_saliency_dir = dir_final_saliency + '-3channel'
        confignamebase = 'comparison-f7/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        ensemble_average_saliency_map(config_dirbase, list_ensemble_saliency, confignamebase, num_of_seeds, testdemo,
                                      save_ensemble_saliency_dir)

        list_ensemble_saliency = ['t2-adc-dwib800-unregistered']
        save_ensemble_saliency_dir = dir_final_saliency + '-3channel'
        confignamebase = 'comparison-f7/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        ensemble_average_saliency_map(config_dirbase, list_ensemble_saliency, confignamebase, num_of_seeds, testdemo,
                                      save_ensemble_saliency_dir)

        list_ensemble_saliency = ['t2-adc-ktrans-unregistered']
        save_ensemble_saliency_dir = dir_final_saliency + '-3channel'
        confignamebase = 'comparison-f7/augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        ensemble_average_saliency_map(config_dirbase, list_ensemble_saliency, confignamebase, num_of_seeds, testdemo,
                                      save_ensemble_saliency_dir)



    def print_results_test_statistics():
        '''
        1-1. print results
        save_results_test output & statistics
        visualize saliency map
        '''

        rng_seed = 42  # control reproducibility
        rng = np.random.RandomState(rng_seed)
        # rng = np.random.default_rng()
        n_trials = 10000

        dir_prediction = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/'
        model_basename = '8conv4mpfcf4-earlystopping-eval-'

        chan1_t2_adc_dwib800, target = ensemble_different_modality(dir_prediction, model_basename,
                                                                   ['t2', 'adc', 'dwib800'])
        print_all_results(target, chan1_t2_adc_dwib800, n_trials, rng)
        print_all_results(target, chan1_t2_adc_dwib800, n_trials, rng, 30)

        chan1_t2_adc_ktrans, target = ensemble_different_modality(dir_prediction, model_basename,
                                                                  ['t2', 'adc', 'ktrans'])
        print_all_results(target, chan1_t2_adc_ktrans, n_trials, rng)
        print_all_results(target, chan1_t2_adc_ktrans, n_trials, rng, 30)

        chan1_t2_adc_ktrans_unregistered, target = ensemble_different_modality(dir_prediction, model_basename,
                                                                  ['t2', 'adc-unregistered', 'ktrans'])
        print_all_results(target, chan1_t2_adc_ktrans_unregistered, n_trials, rng)


        chan1_t2_adc_dwib800_unregistered, target = ensemble_different_modality(dir_prediction, model_basename,
                                                                  ['t2', 'adc-unregistered', 'dwib800-unregistered'])
        print_all_results(target, chan1_t2_adc_dwib800_unregistered, n_trials, rng)

        chan3_t2_adc_dwib800, target = ensemble_different_modality(dir_prediction, model_basename,
                                                                   ['3channel-t2-adc-dwib800'])
        print_all_results(target, chan3_t2_adc_dwib800, n_trials, rng)
        print_all_results(target, chan3_t2_adc_dwib800, n_trials, rng, 30)

        chan3_t2_adc_ktrans, target = ensemble_different_modality(dir_prediction, model_basename,
                                                                  ['3channel-t2-adc-ktrans'])
        print_all_results(target, chan3_t2_adc_ktrans, n_trials, rng)
        print_all_results(target, chan3_t2_adc_ktrans, n_trials, rng, 30)


        chan1_t2, target = ensemble_different_modality(dir_prediction, model_basename, ['t2-15init'])
        print_all_results(target, chan1_t2, n_trials, rng)

        chan1_adc, target = ensemble_different_modality(dir_prediction, model_basename, ['adc-15init'])
        print_all_results(target, chan1_adc, n_trials, rng)

        chan1_dwib800, target = ensemble_different_modality(dir_prediction, model_basename, ['dwib800-15init'])
        print_all_results(target, chan1_dwib800, n_trials, rng)

        chan1_ktrans, target = ensemble_different_modality(dir_prediction, model_basename, ['ktrans-15init'])
        print_all_results(target, chan1_ktrans, n_trials, rng)

        chan1_adc_unregistered, target = ensemble_different_modality(dir_prediction, model_basename, ['adc-unregistered-15init'])
        print_all_results(target, chan1_adc_unregistered, n_trials, rng)

        chan1_dwib800_unregistered, target = ensemble_different_modality(dir_prediction, model_basename, ['dwib800-unregistered-15init'])
        print_all_results(target, chan1_dwib800_unregistered, n_trials, rng)

        chan3_t2_adc_dwib800_unregistered, target = ensemble_different_modality(dir_prediction, model_basename,
                                                                   ['3channel-t2-adc-dwib800-unregistered'])
        print_all_results(target, chan3_t2_adc_dwib800_unregistered, n_trials, rng)


        chan3_t2_adc_ktrans_unregistered, target = ensemble_different_modality(dir_prediction, model_basename,
                                                                  ['3channel-t2-adc-ktrans-unregistered'])
        print_all_results(target, chan3_t2_adc_ktrans_unregistered, n_trials, rng)


        ## AUC comparison pvalue DeLong

        from roccomparison.compare_auc_delong_xu import delong_roc_variance, delong_roc_test
        import sklearn.linear_model
        import scipy.stats

        def pvalue_delong_roc_test(target, pred1, pred2):
            return 10**(delong_roc_test(target, pred1, pred2))

        pvalue_delong_roc_test(target, chan1_adc, chan3_t2_adc_dwib800)
        pvalue_delong_roc_test(target, chan1_adc, chan3_t2_adc_ktrans)
        pvalue_delong_roc_test(target, chan3_t2_adc_ktrans_unregistered, chan3_t2_adc_ktrans)
        pvalue_delong_roc_test(target, chan3_t2_adc_dwib800_unregistered, chan3_t2_adc_dwib800)
        pvalue_delong_roc_test(target, chan1_t2, chan3_t2_adc_dwib800)
        pvalue_delong_roc_test(target, chan1_t2, chan3_t2_adc_ktrans)







    '''
    Get original slice / Saliency
    '''
    def getPatchOnOriginalSpace_returnSlice(RefImage, Patch, MRvox_middle, xthr, ythr, zthr):
        # zero padding
        if (RefImage.shape[0] < xthr) or (RefImage.shape[1] < ythr) or (RefImage.shape[2] < zthr):
            pad = np.array([xthr - RefImage.shape[0], ythr - RefImage.shape[1], zthr - RefImage.shape[2]]) / 2
            pad[pad < 0] = 0
            RefImage = np.pad(RefImage, ((np.floor(pad[0]).astype('int'), np.ceil(pad[0]).astype('int')),
                                         (np.floor(pad[1]).astype('int'), np.ceil(pad[1]).astype('int')),
                                         (np.floor(pad[2]).astype('int'), np.ceil(pad[2]).astype('int'))),
                              'constant')

        # Image: 3D
        bbx_start, bby_start, bbz_start = np.floor(
            MRvox_middle - np.array([xthr / 2, ythr / 2, zthr / 2])).astype(
            'int')

        if bbx_start < 0: bbx_start = 0
        if bby_start < 0: bby_start = 0
        if bbz_start < 0: bbz_start = 0
        if bbx_start + xthr > RefImage.shape[0]: bbx_start = RefImage.shape[0] - xthr
        if bby_start + ythr > RefImage.shape[1]: bby_start = RefImage.shape[1] - ythr
        if bbz_start + zthr > RefImage.shape[2]: bbz_start = RefImage.shape[2] - zthr
        # return Image[:, bbx_start: (bbx_start + xthr), bby_start: (bby_start + ythr), bbz_start:(bbz_start + zthr)]
        PatchMask = np.zeros(np.shape(RefImage))
        PatchMask[bbx_start: (bbx_start + xthr), bby_start: (bby_start + ythr),
        bbz_start:(bbz_start + zthr)] = Patch
        return PatchMask

    def getBoundingBox(Image, MRvox_middle, xthr, ythr, zthr):
        # zero padding
        if (Image.shape[0] < xthr) or (Image.shape[1] < ythr) or (Image.shape[2] < zthr):
            pad = np.array([xthr - Image.shape[0], ythr - Image.shape[1], zthr - Image.shape[2]]) / 2
            pad[pad < 0] = 0
            Image = np.pad(Image, ((np.floor(pad[0]).astype('int'), np.ceil(pad[0]).astype('int')),
                                         (np.floor(pad[1]).astype('int'), np.ceil(pad[1]).astype('int')),
                                         (np.floor(pad[2]).astype('int'), np.ceil(pad[2]).astype('int'))),
                              'constant')

        # Image: 3D
        bbx_start, bby_start, bbz_start = np.floor(
            MRvox_middle - np.array([xthr / 2, ythr / 2, zthr / 2])).astype(
            'int')

        if bbx_start < 0: bbx_start = 0
        if bby_start < 0: bby_start = 0
        if bbz_start < 0: bbz_start = 0
        if bbx_start + xthr > Image.shape[0]: bbx_start = Image.shape[0] - xthr
        if bby_start + ythr > Image.shape[1]: bby_start = Image.shape[1] - ythr
        if bbz_start + zthr > Image.shape[2]: bbz_start = Image.shape[2] - zthr
        return Image[bbx_start: (bbx_start + xthr), bby_start: (bby_start + ythr), bbz_start:(bbz_start + zthr)]

    ## visualize saliency map total
    # t2 / adc / dwi / ktrans
    # saliency of each of them
    # ensemble (1c w/ dwi) (1c w/ ktrans) (3c w/ dwi) (3c w/ ktrans)
    # tumor overlay on

    # nifti name
    # subject name
    # h5 name (original)
    # get patches # look at loaders
    # get patch location & zero mask & Locate the saliency map
    # high resolution (original size)
    # prediction score

    ##
    # separate image (orig) t2 / adc / dwib800 / ktrans
    # separate saliency for t2 / adc / dwib800 / ktrans
    # average saliency t2+adc+dwib800 / +overlay / t2+adc+ktrans + overlay
    # name based on tn/fn/fp/tp

    testdemof = '/home/heejong/data/prostatex/demo-h5-sitk/prostateX-demo-csPCa-prediction-multimodal-sitk-newidx.csv'
    testdemo = pd.read_csv(testdemof) # config['loaders']['csvfname']
    testdemo = testdemo[testdemo.trainvaltest == 'test'].reset_index()

    names = np.array(testdemo['ProxID'] + '-' + testdemo['fid'].astype('str'))
    h5names = np.array(testdemo['h5-fname'])

    dir_saliency_t2 = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-1channel/' \
                      't2'
    dir_saliency_adc = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-1channel/' \
                       'adc'
    dir_saliency_dwib800 = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-1channel/' \
                           'dwib800'
    dir_saliency_ktrans = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-1channel/' \
                          'ktrans'
    dir_saliency_1c_t2_adc_dwib800 = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-1channel/' \
                                     't2-adc-dwib800'
    dir_saliency_1c_t2_adc_ktrans = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-1channel/' \
                                    't2-adc-ktrans'

    dir_saliency_3c_t2_adc_dwib800 = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-3channel/' \
                                     't2-adc-dwib800'
    dir_saliency_3c_t2_adc_ktrans = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-3channel/' \
                                    't2-adc-ktrans'
    target = np.array(pd.read_csv('/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-t2-15init.csv')['ClinSig-target'])
    prediction_t2 = np.array(pd.read_csv('/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-t2-15init.csv')['ClinSig-ensemble'])
    prediction_adc = np.array(pd.read_csv('/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-adc-15init.csv')['ClinSig-ensemble'])
    prediction_dwib800 = np.array(pd.read_csv('/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-dwib800-15init.csv')['ClinSig-ensemble'])
    prediction_ktrans = np.array(pd.read_csv('/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-ktrans-15init.csv')['ClinSig-ensemble'])
    prediction_1c_t2_adc_dwib800 = (prediction_t2 + prediction_adc + prediction_dwib800)/3
    prediction_1c_t2_adc_ktrans = (prediction_t2 + prediction_adc + prediction_ktrans)/3
    prediction_3c_t2_adc_dwib800 = np.array(pd.read_csv('/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-3channel-t2-adc-dwib800.csv')['ClinSig-ensemble'])
    prediction_3c_t2_adc_ktrans = np.array(pd.read_csv('/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-3channel-t2-adc-ktrans.csv')['ClinSig-ensemble'])
    dir_figure_slice = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/slice-visualization/summary'

    #############################
    # dir_saliency_adc = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-1channel/' \
    #                    'adc-unregistered'
    # dir_saliency_dwib800 = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-1channel/' \
    #                        'dwib800-unregistered'
    # dir_saliency_1c_t2_adc_dwib800 = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-1channel/' \
    #                                  't2-adc-unregistered-dwib800-unregistered'
    # dir_saliency_1c_t2_adc_ktrans = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-1channel/' \
    #                                 't2-adc-unregistered-ktrans'
    #
    # dir_saliency_3c_t2_adc_dwib800 = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-3channel/' \
    #                                  't2-adc-dwib800-unregistered'
    # dir_saliency_3c_t2_adc_ktrans = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/saliency-3channel/' \
    #                                 't2-adc-ktrans-unregistered'
    #
    # prediction_adc = np.array(pd.read_csv('/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-adc-unregistered-15init.csv')['ClinSig-ensemble'])
    # prediction_dwib800 = np.array(pd.read_csv('/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-dwib800-unregistered-15init.csv')['ClinSig-ensemble'])
    # prediction_1c_t2_adc_dwib800 = (prediction_t2 + prediction_adc + prediction_dwib800)/3
    # prediction_1c_t2_adc_ktrans = (prediction_t2 + prediction_adc + prediction_ktrans)/3
    # prediction_3c_t2_adc_dwib800 = np.array(pd.read_csv('/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-3channel-t2-adc-dwib800-unregistered.csv')['ClinSig-ensemble'])
    # prediction_3c_t2_adc_ktrans = np.array(pd.read_csv('/home/heejong/projects/biopsy-prediction/prostatex_final_result/prediction/8conv4mpfcf4-earlystopping-eval-3channel-t2-adc-ktrans-unregistered.csv')['ClinSig-ensemble'])
    # dir_figure_slice = '/home/heejong/projects/biopsy-prediction/prostatex_final_result/slice-visualization/summary-unregistered/'

    #############################

    acc, thr_t2 = accuracy_with_threshold(target, prediction_t2)
    acc, thr_adc = accuracy_with_threshold(target, prediction_adc)
    acc, thr_dwib800 = accuracy_with_threshold(target, prediction_dwib800)
    acc, thr_ktrans = accuracy_with_threshold(target, prediction_ktrans)
    acc, thr_1c_t2_adc_dwib800 = accuracy_with_threshold(target, prediction_1c_t2_adc_dwib800)
    acc, thr_1c_t2_adc_ktrans = accuracy_with_threshold(target, prediction_1c_t2_adc_ktrans)
    acc, thr_3c_t2_adc_dwib800 = accuracy_with_threshold(target, prediction_3c_t2_adc_dwib800)
    acc, thr_3c_t2_adc_ktrans = accuracy_with_threshold(target, prediction_3c_t2_adc_ktrans)


    # without minmax
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 22})


    def get_padded_saliency(dir_saliency, name, referenceImage, patch_size, lesion_middle_point):
        saliency = np.asanyarray(nib.load(os.path.join(dir_saliency, name + '.nii.gz')).dataobj)
        padded_saliency = getPatchOnOriginalSpace_returnSlice(referenceImage, saliency,
                                                              lesion_middle_point, patch_size[0], patch_size[1], patch_size[2])
        return padded_saliency


    segdemo = pd.read_csv(
        '/home/heejong/data/prostatex/demo-h5-sitk/prostateX-demo-csPCa-prediction-multimodal-sitk-newidx-segmentation.csv')
    segdemo = segdemo[segdemo.trainvaltest == 'test'].reset_index()

    '/nfs04/data/PROSTATEx/PROSTATEx_masks-master/'
    '/home/heejong/data/prostatex/PROSTATEx_masks-master/'
    segfnames = np.array(segdemo['lesion-segmentation-fname'].str.replace('/nfs04/data/PROSTATEx/PROSTATEx_masks-master/','/home/heejong/data/prostatex/PROSTATEx_masks-master/')).astype('str')

    for si in range(len(names)):
        name = names[si]
        h5open = h5py.File(h5names[si], 'r')
        t2 = np.array(h5open['t2'])
        adc = np.array(h5open['adcregistered'])
        ktrans = np.array(h5open['ktrans'])
        dwi = np.array(h5open['dwib800registered'])
        lesion = np.array(h5open['lesion'])
        patch_size = (np.array(h5open['patch-size'])/2).astype('int')
        h5open.close()

        lesion_middle_point = np.median(np.array(np.where(lesion)),1).astype('int')

        fig, arr = plt.subplots(4, 4, figsize=(24,24))# , figsize=(50,100))
        arr[0, 0].imshow(t2[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
        arr[0, 0].title.set_text('T2')
        arr[0, 1].imshow(adc[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
        arr[0, 1].title.set_text('ADC')
        arr[0, 2].imshow(dwi[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
        arr[0, 2].title.set_text('DWIb800')
        arr[0, 3].imshow(ktrans[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
        arr[0, 3].title.set_text('Ktrans')
        arr[0, 0].axis('off'); arr[0, 1].axis('off'); arr[0, 2].axis('off'); arr[0, 3].axis('off');
        #
        arr[1, 0].imshow(t2[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
        arr[1, 1].imshow(t2[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
        arr[1, 2].imshow(t2[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
        arr[1, 3].imshow(t2[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
        arr[1, 0].imshow(get_padded_saliency(dir_saliency_t2, name, t2, patch_size, lesion_middle_point)[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('jet'), alpha=0.5)
        arr[1, 1].imshow(get_padded_saliency(dir_saliency_adc, name, t2, patch_size, lesion_middle_point)[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('jet'), alpha=0.5)
        arr[1, 2].imshow(get_padded_saliency(dir_saliency_dwib800, name, t2, patch_size, lesion_middle_point)[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('jet'),alpha=0.5)
        arr[1, 3].imshow(get_padded_saliency(dir_saliency_ktrans, name, t2, patch_size, lesion_middle_point)[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('jet'),alpha=0.5)
        # im = arr[1, 3].imshow(padded_saliency[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('jet'), alpha=0.5)
        arr[1, 0].title.set_text(f'T2 only \n Pred:{prediction_t2[si]:.2f}/Label:{prediction_t2[si]>thr_t2:.1f}')
        arr[1, 1].title.set_text(f'ADC only \n Pred:{prediction_adc[si]:.2f}/Label:{prediction_adc[si]>thr_adc:.1f}')
        arr[1, 2].title.set_text(f'DWIb800 only \n Pred:{prediction_dwib800[si]:.2f}/Label:{prediction_dwib800[si]>thr_dwib800:.1f}')
        arr[1, 3].title.set_text(f'Ktrans only \n Pred:{prediction_ktrans[si]:.2f}/Label:{prediction_ktrans[si]>thr_ktrans:.1f}')
        arr[1, 0].axis('off'); arr[1, 1].axis('off'); arr[1, 2].axis('off'); arr[1, 3].axis('off');


        arr[2, 0].imshow(t2[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
        arr[2, 1].imshow(t2[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
        arr[2, 2].imshow(t2[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
        arr[2, 3].imshow(t2[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
        arr[2, 0].imshow(get_padded_saliency(dir_saliency_1c_t2_adc_dwib800, name, t2, patch_size, lesion_middle_point)[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('jet'), alpha=0.5)
        arr[2, 1].imshow(get_padded_saliency(dir_saliency_1c_t2_adc_ktrans, name, t2, patch_size, lesion_middle_point)[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('jet'), alpha=0.5)
        arr[2, 2].imshow(get_padded_saliency(dir_saliency_3c_t2_adc_dwib800, name, t2, patch_size, lesion_middle_point)[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('jet'),alpha=0.5)
        arr[2, 3].imshow(get_padded_saliency(dir_saliency_3c_t2_adc_ktrans, name, t2, patch_size, lesion_middle_point)[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('jet'),alpha=0.5)
        # im = arr[1, 3].imshow(padded_saliency[:, :, lesion_middle_point[-1]].transpose(), cmap=plt.get_cmap('jet'), alpha=0.5)
        arr[2, 0].title.set_text(f'T2-ADC-DWIb800 1 chan\n Pred:{prediction_1c_t2_adc_dwib800[si]:.2f}/Label:{prediction_1c_t2_adc_dwib800[si]>thr_1c_t2_adc_dwib800:.1f}')
        arr[2, 1].title.set_text(f'T2-ADC-Ktrans 1 chan\n Pred:{prediction_1c_t2_adc_ktrans[si]:.2f}/Label:{prediction_1c_t2_adc_ktrans[si]>thr_1c_t2_adc_ktrans:.1f}')
        arr[2, 2].title.set_text(f'T2-ADC-DWIb800 3 chan\n Pred:{prediction_3c_t2_adc_dwib800[si]:.2f}/Label:{prediction_1c_t2_adc_dwib800[si]>thr_3c_t2_adc_dwib800:.1f}')
        arr[2, 3].title.set_text(f'T2-ADC-Ktrans 3 chan\n Pred:{prediction_3c_t2_adc_ktrans[si]:.2f}/Label:{prediction_3c_t2_adc_ktrans[si]>thr_3c_t2_adc_ktrans:.1f}')
        arr[2, 0].axis('off'); arr[2, 1].axis('off'); arr[2, 2].axis('off'); arr[2, 3].axis('off');

        # tumor overlay
        if False:
        # if segfnames[si] != 'nan':
            import torchio as tio
            from oscarpreprocessing.preprocessing_lib import get_lesion_mask_id_seed
            import SimpleITK as sitk
            # get t2 size
            # get segimage size # t2 = 0.5mm
            segsize = nib.load(segfnames[si]).header.get_zooms()
            t2size = (0.5, 0.5, 0.5)
            transform = tio.Resample((1, 1, t2size[2]/segsize[2]))
            t2_shape = t2.shape
            dir_segt2 = '/home/heejong/data/prostatex/PROSTATEx_masks-master/Files/lesions/Images/T2'
            t2itk = sitk.ReadImage(glob.glob(os.path.join(dir_segt2, segdemo.ProxID.iloc[si] + '*'))[0])
            t2reshaped = sitk.GetArrayFromImage(t2itk).transpose(2, 1, 0)[::-1, ::-1]
            t2resampled = transform((t2reshaped[None, :]).copy())[0]
            segitk = sitk.ReadImage(segfnames[si])
            positions = np.array(testdemo['pos'].iloc[si].split()).astype('float')
            positions_img = np.array([segitk.TransformPhysicalPointToContinuousIndex(positions.astype(np.float64))])
            lesion_mask_id_seed_update = get_lesion_mask_id_seed(positions_img, segitk)
            positionI = sitk.GetArrayFromImage(lesion_mask_id_seed_update).transpose(2, 1, 0)[::-1, ::-1]
            positionresampled = transform((positionI * 10)[None, :].copy())[0]
            segitk_reshaped = sitk.GetArrayFromImage(segitk).transpose(2, 1, 0)[::-1, ::-1]
            segitkresampled = transform((segitk_reshaped)[None, :].copy())[0]
            seg_middle = np.median(np.array(np.where(positionresampled)), 1).astype('int')
            positionresampled_patch = getBoundingBox(positionresampled, seg_middle, t2_shape[0], t2_shape[1], t2_shape[2])
            seg_middle_patch = np.median(np.array(np.where(positionresampled_patch)), 1).astype('int')

            # lesion_middle_point_orig =
            arr[3, 0].title.set_text('T2-lesion:  \n'
                                     'Cuocolo selected T2 & manual segmentation')
            arr[3, 0].imshow(getBoundingBox(t2resampled, seg_middle, t2_shape[0], t2_shape[1], t2_shape[2])[:, :,
                          seg_middle_patch[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))

            arr[3, 0].imshow(getBoundingBox(segitkresampled, seg_middle, t2_shape[0], t2_shape[1], t2_shape[2])[:, :,
                          seg_middle_patch[-1]].transpose(), cmap=plt.get_cmap('Reds'), alpha=0.5)

        arr[3, 0].axis('off'); arr[3, 1].axis('off'); arr[3, 2].axis('off'); arr[3, 3].axis('off');

        # plt.colorbar(im, ax=arr[1, 2])
        fig.suptitle(f'{names[si]}: / True Label {target[si]}')
        savefigname = os.path.join(dir_figure_slice, f'label{target[si]:.0f}_{names[si]}')
        fig.savefig(savefigname)
        plt.close(fig)
        plt.clf()

    from oscarpreprocessing.preprocessing_lib import get_lesion_mask_id_seed
    dir_segt2 ='/home/heejong/data/prostatex/PROSTATEx_masks-master/Files/lesions/Images/T2'
    t2itk = sitk.ReadImage(glob.glob(os.path.join(dir_segt2, segdemo.ProxID.iloc[si] + '*'))[0])
    t2reshaped = sitk.GetArrayFromImage(t2itk).transpose(2, 1, 0)[ ::-1, ::-1]
    t2resampled = transform((t2reshaped[None, :]).copy())[0]

    segitk = sitk.ReadImage(segfnames[si])
    positions = np.array(testdemo['pos'].iloc[si].split()).astype('float')
    positions_img = np.array([segitk.TransformPhysicalPointToContinuousIndex(positions.astype(np.float64))])
    lesion_mask_id_seed_update = get_lesion_mask_id_seed(positions_img, segitk)
    positionI = sitk.GetArrayFromImage(lesion_mask_id_seed_update).transpose(2, 1, 0)[::-1, ::-1]
    positionresampled = transform((positionI*10)[None, :].copy())[0]
    segitk_reshaped = sitk.GetArrayFromImage(segitk).transpose(2, 1, 0)[ ::-1, ::-1]
    segitkresampled = transform((segitk_reshaped)[None, :].copy())[0]

    seg_middle = np.median(np.array(np.where(positionresampled)),1).astype('int')
    # seg_middle = np.median(np.array(np.where(segitkresampled)), 1).astype('int')

    fig, arr = plt.subplots(1, 3, figsize=(12, 4))  # , figsize=(50,100))
    arr[0].imshow(getBoundingBox(t2resampled, seg_middle, t2_shape[0], t2_shape[1], t2_shape[2])[:, :, seg_middle[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
    arr[1].imshow(getBoundingBox(segitkresampled, seg_middle, t2_shape[0], t2_shape[1], t2_shape[2])[:, :, seg_middle[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
    arr[2].imshow(getBoundingBox(positionresampled, seg_middle, t2_shape[0], t2_shape[1], t2_shape[2])[:, :, seg_middle[-1]].transpose(), cmap=plt.get_cmap('gist_gray'))
    arr[0].axis('off');arr[1].axis('off');arr[2].axis('off');
    fig.savefig('./tmp2.png')



    def save_results_challenge_test():
        '''
        2. Get ensemble result for independent testing set (challenge)
        '''

        testdemof = '/home/heejong/data/prostatex/demo-h5-sitk/prostateX-demo-challenge-testset.csv'
        # make test demo
        if os.path.exists(testdemof):
            testdemo = pd.read_csv(testdemof)
            testdemo = testdemo.drop(columns=['Unnamed: 0'])

        else:
            dir_testset = '/home/heejong/data/prostatex/test/'
            testlist = glob.glob(os.path.join(dir_testset, '*.h5'))
            testlist.sort()

            testdemo = pd.DataFrame()
            testdemo['h5-fname'] = testlist
            testdemo['ClinSig'] = np.nan
            testdemo['trainvaltest'] = 'test'
            testdemo.to_csv('/home/heejong/data/prostatex/demo-h5-sitk/prostateX-demo-challenge-testset.csv')

        config_dir_base = './config-prostatex-local-8conv4mpfcf4-earlystopping-eval/'
        test_n_epochs = 4

        ############ t2
        subname = 't2'
        config_dir = config_dir_base  # local
        confignamebase = 'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        # seeds_pred_val, seeds_pred_test, target_val, target_test = get_prediction_each_modality_setouttest(config_dir,
        #                                                                                                    testdemof,
        #                                                                                                    confignamebase=confignamebase,
        #                                                                                                    num_of_seeds=5)
        # fnamebase = f'/home/heejong/projects/biopsy-prediction/prostatex_submission/prostatex_submission_{subname}-earlystoppingAUC'
        # seeds_pred_test_t2 = get_submission_file(seeds_pred_test, fnamebase, testdemo, test_n_epochs)

        seeds_pred_val, seeds_pred_test, target_val, target_test = get_prediction_each_modality_setouttest(config_dir,
                                                                                                           testdemof,
                                                                                                           confignamebase=confignamebase,
                                                                                                           num_of_seeds=15)
        fnamebase = f'/home/heejong/projects/biopsy-prediction/prostatex_submission/prostatex_submission_{subname}-earlystoppingAUC-15init'
        seeds_pred_test_t2 = get_submission_file(seeds_pred_test, fnamebase, testdemo, test_n_epochs)

        del seeds_pred_val
        del seeds_pred_test
        del target_val
        del target_test

        ############ adc
        subname = 'adc'
        config_dir = f'{config_dir_base}/{subname}/'  # local
        confignamebase = 'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        seeds_pred_val, seeds_pred_test, target_val, target_test = get_prediction_each_modality_setouttest(config_dir,
                                                                                                           testdemof,
                                                                                                           confignamebase=confignamebase,
                                                                                                           num_of_seeds=15)
        fnamebase = f'/home/heejong/projects/biopsy-prediction/prostatex_submission/prostatex_submission_{subname}-earlystoppingAUC-15init-rerun'
        seeds_pred_test_adc = get_submission_file(seeds_pred_test, fnamebase, testdemo, test_n_epochs)

        del seeds_pred_val
        del seeds_pred_test
        del target_val
        del target_test

        ############ dwib800
        subname = 'dwib800'
        config_dir = f'{config_dir_base}/{subname}/'  # local
        confignamebase = 'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        seeds_pred_val, seeds_pred_test, target_val, target_test = get_prediction_each_modality_setouttest(config_dir,
                                                                                                           testdemof,
                                                                                                           confignamebase=confignamebase,
                                                                                                           num_of_seeds=15)
        fnamebase = f'/home/heejong/projects/biopsy-prediction/prostatex_submission/prostatex_submission_{subname}-earlystoppingAUC-15init'
        seeds_pred_test_dwib800 = get_submission_file(seeds_pred_test, fnamebase, testdemo, test_n_epochs)

        del seeds_pred_val
        del seeds_pred_test
        del target_val
        del target_test

        ############ ktrans
        subname = 'ktrans'
        config_dir = f'{config_dir_base}/{subname}/'  # local
        confignamebase = 'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        seeds_pred_val, seeds_pred_test, target_val, target_test = get_prediction_each_modality_setouttest(config_dir,
                                                                                                           testdemof,
                                                                                                           confignamebase=confignamebase,
                                                                                                           num_of_seeds=15)
        fnamebase = f'/home/heejong/projects/biopsy-prediction/prostatex_submission/prostatex_submission_{subname}-earlystoppingAUC-15init'
        seeds_pred_test_ktrans = get_submission_file(seeds_pred_test, fnamebase, testdemo, test_n_epochs)

        del seeds_pred_val
        del seeds_pred_test
        del target_val
        del target_test

        ############ adcunregistered
        subname = 'adc-unregistered'
        config_dir = f'{config_dir_base}/{subname}/'  # local
        confignamebase = 'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        seeds_pred_val, seeds_pred_test, target_val, target_test = get_prediction_each_modality_setouttest(config_dir,
                                                                                                           testdemof,
                                                                                                           confignamebase=confignamebase,
                                                                                                           num_of_seeds=15)
        fnamebase = f'/home/heejong/projects/biopsy-prediction/prostatex_submission/prostatex_submission_{subname}-earlystoppingAUC-15init'
        seeds_pred_test_adc_unregistered = get_submission_file(seeds_pred_test, fnamebase, testdemo, test_n_epochs)

        del seeds_pred_val
        del seeds_pred_test
        del target_val
        del target_test

        ############ dwib800 unregistered
        subname = 'dwib800-unregistered'
        config_dir = f'{config_dir_base}/{subname}/'  # local
        confignamebase = 'augment-all-cnn3d8conv4mp3fcf4-1channel-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        seeds_pred_val, seeds_pred_test, target_val, target_test = get_prediction_each_modality_setouttest(config_dir,
                                                                                                           testdemof,
                                                                                                           confignamebase=confignamebase,
                                                                                                           num_of_seeds=15)
        fnamebase = f'/home/heejong/projects/biopsy-prediction/prostatex_submission/prostatex_submission_{subname}-earlystoppingAUC-15init'
        seeds_pred_test_dwib800_unregistered = get_submission_file(seeds_pred_test, fnamebase, testdemo, test_n_epochs)

        ############  1 channel multimodality ensemble (t2-adc-dwib800-unregistered)
        subname = 't2-adc-dwib800-unregistered-earlystoppingAUC'
        fnamebase = f'/home/heejong/projects/biopsy-prediction/prostatex_submission/prostatex_submission_{subname}'
        seeds_pred_t2_adc_dwib800_unregistered = (seeds_pred_test_t2 + seeds_pred_test_adc_unregistered + seeds_pred_test_dwib800_unregistered) / 3
        get_separate_submission_file(seeds_pred_t2_adc_dwib800_unregistered, fnamebase, testdemo)

        ############  1 channel multimodality ensemble (t2-adc-ktrans-unregistered)
        subname = 't2-adc-ktrans-unregistered-earlystoppingAUC'
        fnamebase = f'/home/heejong/projects/biopsy-prediction/prostatex_submission/prostatex_submission_{subname}'
        seeds_pred_t2_adc_ktrans_unregistered = (seeds_pred_test_t2 + seeds_pred_test_adc_unregistered + seeds_pred_test_ktrans) / 3
        get_separate_submission_file(seeds_pred_t2_adc_ktrans_unregistered, fnamebase, testdemo)




        ############ 3 channel input (t2-adc-dwib800)
        # config_dir = './config-prostatex-local-8conv4mpfcf4-earlystopping-eval/t2-adc-dwib800/comparison-f7/'  # local
        config_dir = './config-prostatex-local-8conv4mpfcf4-earlystopping-eval/t2-adc-dwib800-unregistered/comparison-f7/'  # local
        confignamebase = 'augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        seeds_pred_val_3c, seeds_pred_test_3c, target_val_3c, target_test_3c = get_prediction_each_modality_setouttest(
            config_dir, testdemof,
            confignamebase, 15)
        subname = '3channel-t2-adc-dwib800-f7-unregistered-earlystoppingAUC'
        fnamebase = f'/home/heejong/projects/biopsy-prediction/prostatex_submission/prostatex_submission_{subname}.csv'
        get_submission_file(seeds_pred_test_3c, fnamebase, testdemo, test_n_epochs)

        del seeds_pred_val_3c
        del seeds_pred_test_3c
        del target_val_3c
        del target_test_3c

        ############ 3 channel input (t2-adc-ktrans)
        # config_dir = './config-prostatex-local-8conv4mpfcf4-earlystopping-eval/t2-adc-ktrans/comparison-f7/'  # local
        config_dir = './config-prostatex-local-8conv4mpfcf4-earlystopping-eval/t2-adc-ktrans-unregistered/comparison-f7/'  # local
        confignamebase = 'augment-all-cnn3d8conv4mp3fcf7-batch64-adam-lre-0001-bcelogit-auc-nobfc'
        seeds_pred_val_3c, seeds_pred_test_3c, target_val_3c, target_test_3c = get_prediction_each_modality_setouttest(
            config_dir, testdemof,
            confignamebase, 15)
        subname = '3channel-t2-adc-ktrans-f7-unregistered-earlystoppingAUC'
        fnamebase = f'/home/heejong/projects/biopsy-prediction/prostatex_submission/prostatex_submission_{subname}.csv'
        get_submission_file(seeds_pred_test_3c, fnamebase, testdemo, test_n_epochs)

        del seeds_pred_val_3c
        del seeds_pred_test_3c
        del target_val_3c
        del target_test_3c



    def draw_bar_graph():
        '''
        Visualization
        '''
        # draw bar graph

        # width of the bars
        barWidth = 0.3

        # t2 adc dwib800 ktrans
        bars_single = [0.82, 0.84, 0.84, 0.72]
        bars_single_challenge = [0.74, 0.83, 0.79, 0.76]
        single_ci = np.array([[0.68, 0.91], [0.71, 0.92], [0.68, 0.92],[0.54, 0.85]]).T
        single_err = np.abs(np.array([bars_single, bars_single]) - single_ci)


        # late fusion (dwib800 ktrans)
        bars_late = [0.90, 0.89]
        bars_late_challenge = [0.84, 0.84]
        late_ci = np.array([[0.78, 0.96], [0.76, 0.96]]).T
        late_err = np.abs(np.array([bars_late, bars_late]) - late_ci)

        # early fusion (dwib800 ktrans)
        bars_early = [0.88, 0.80]
        bars_early_challenge = [0.84, 0.82]
        early_ci = np.array([[0.75, 0.95],[0.62, 0.91]]).T
        early_err = np.abs(np.array([bars_early, bars_early]) - early_ci)


        # position
        rsingle = np.arange(len(bars_single))
        rlate = np.arange(len(bars_single), len(bars_single)+len(bars_late))
        rearly =np.arange( len(bars_single)+len(bars_late), len(bars_single)+len(bars_late)+len(bars_early))

        rsinglec =  [x + barWidth for x in rsingle]
        rlatec =  [x + barWidth for x in rlate]
        rearlyc =  [x + barWidth for x in rearly]


        # single
        plt.bar(rsingle, bars_single, width=barWidth, color='blue', edgecolor='black', yerr=single_err, capsize=7, label='1c')
        # single challenge
        plt.bar(rsinglec, bars_single_challenge, width=barWidth, color='cyan', edgecolor='black',capsize=7, label='1c challenge')

        # late fusion
        plt.bar(rlate, bars_late, width=barWidth, color='red', edgecolor='black', yerr=late_err, capsize=7, label='1c')
        # late fusion challenge
        plt.bar(rlatec, bars_late_challenge, width=barWidth, color='pink', edgecolor='black', capsize=7, label='1c')

        # early fusion
        plt.bar(rearly, bars_early, width=barWidth, color='red', edgecolor='blue', yerr=early_err, capsize=7, label='1c')
        # early fusion challenge
        plt.bar(rearlyc, bars_early_challenge, width=barWidth, color='pink', edgecolor='blue', capsize=7, label='1c')


        # general layout
        # tickpos = np.sort(np.concatenate((rsingle, rsinglec, rlate, rlatec, rearly, rearlyc), 0))
        tickpos = np.sort(np.concatenate((rsingle, rlate, rearly), 0))
        plt.xticks(list(tickpos), ['t2', 'adc', 'dwib800','ktrans','late-dwib800', 'late-ktrans','early-dwib800', 'early-ktrans'])
        # plt.ylim([0.5,1])
        plt.ylim([0,1])
        plt.ylabel('ROC-AUC')
        plt.legend()

        # Show graphic
        plt.show()






# if __name__ == '__main__':
    # random_seed()
    # random_seed_setout_test_biopsy()
    