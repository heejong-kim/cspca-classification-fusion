'''
Many functions are based on "pytorch-3dunet" and modified.
https://github.com/wolny/pytorch-3dunet
'''

# from .utils import get_logger, get_tensorboard_formatter, create_sample_plotter, create_optimizer, \
#     create_lr_scheduler, get_number_of_learnable_parameters
from .utils import *
import os
import numpy as np

import importlib
import random

import torch
import torch.nn as nn

from tensorboardX import SummaryWriter
from sklearn.metrics import precision_score, recall_score

logger = get_logger('trainer')

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def get_model(model_config):
    def _model_class(class_name):
        modules = ['utils.model']
        for module in modules:
            m = importlib.import_module(module)
            clazz = getattr(m, class_name, None)
            if clazz is not None:
                return clazz

    model_class = _model_class(model_config['name'])
    return model_class

def get_loader(loader_config):
    def _loader_class(class_name):
        modules = ['utils.loader']
        for module in modules:
            m = importlib.import_module(module)
            clazz = getattr(m, class_name, None)
            if clazz is not None:
                return clazz

    loader_class = _loader_class(loader_config['dataset'])
    return loader_class



def get_loss(loss_config):
    def _loader_class(class_name):
        modules = ['utils.losses']
        for module in modules:
            m = importlib.import_module(module)
            clazz = getattr(m, class_name)
            if clazz is not None:
                return clazz

    loader_class = _loader_class(loss_config['name'])
    return loader_class



def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)

    # get tensorboard formatter
    tensorboard_formatter = get_tensorboard_formatter(trainer_config.pop('tensorboard_formatter', None))
    # get sample plotter
    sample_plotter = create_sample_plotter(trainer_config.pop('sample_plotter', None))

    if resume is not None:
        # continue training from a given checkpoint
        trainer = Trainer.from_checkpoint(model=model,
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler,
                                loss_criterion=loss_criterion,
                                eval_criterion=eval_criterion,
                                loaders=loaders,
                                tensorboard_formatter=tensorboard_formatter,
                                sample_plotter=sample_plotter,
                                **trainer_config)
        # check if config is changed
        if trainer.max_num_epochs != trainer_config['max_num_epochs']:
            trainer.max_num_epochs = trainer_config['max_num_epochs']

        if trainer.max_num_iterations != trainer_config['max_num_iterations']:
            trainer.max_num_iterations = trainer_config['max_num_iterations']

        return trainer

    elif pre_trained is not None:
        # fine-tune a given pre-trained model
        trainer = Trainer.from_pretrained(model=model,
                                             optimizer=optimizer,
                                             lr_scheduler=lr_scheduler,
                                             loss_criterion=loss_criterion,
                                             eval_criterion=eval_criterion,
                                             tensorboard_formatter=tensorboard_formatter,
                                             sample_plotter=sample_plotter,
                                             device=config['device'],
                                             loaders=loaders,
                                             **trainer_config)


        return trainer
    else:
        # start training from scratch
        return Trainer(model=model,
                             optimizer=optimizer,
                             lr_scheduler=lr_scheduler,
                             loss_criterion=loss_criterion,
                             eval_criterion=eval_criterion,
                             device=config['device'],
                             loaders=loaders,
                             tensorboard_formatter=tensorboard_formatter,
                             sample_plotter=sample_plotter,
                             **trainer_config)



class Trainer:
    """ trainer.

    Args:
        model: model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        device (torch.device): device to train on
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        sample_plotter (callable): saves sample inputs, network outputs and targets to a given directory
            during validation phase
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used mostly when
            evaluation is expensive)
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion,
                 eval_criterion, device, loaders, checkpoint_dir,
                 max_num_epochs=100, max_num_iterations=int(1e5),
                 validate_after_iters=100, log_after_iters=100,
                 validate_iters=None, num_iterations=1, num_epoch=0,
                 eval_score_higher_is_better=True, best_eval_score=None,
                 tensorboard_formatter=None, sample_plotter=None,
                 skip_train_validation=False, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.device = device
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.eval_score_higher_is_better = eval_score_higher_is_better

        trainer_config = kwargs

        logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        if best_eval_score is not None:
            self.best_eval_score = best_eval_score
        else:
            # initialize the best_eval_score
            if eval_score_higher_is_better:
                self.best_eval_score = float('-inf')
                self.current_eval_score = float('-inf')
            else:
                self.best_eval_score = float('+inf')
                self.current_eval_score = float('+inf')


        self.best_eval_loss = float('+inf')
        self.current_eval_loss = float('+inf')
        self.early_stopping = trainer_config['early_stopping']
        if self.early_stopping:
            self.early_stopping_count = 0

        if 'early_stopping_criterion' in trainer_config:
            self.early_stopping_criterion = trainer_config['early_stopping_criterion']

        self.visualize_image = trainer_config['visualize_image']
        if 'additional_eval' in trainer_config.keys():
            self.additional_eval = trainer_config['additional_eval']
        else:
            self.additional_eval = False

        if 'save_top5' in trainer_config.keys():
            self.save_top5 = trainer_config['save_top5']
        else:
            self.save_top5 = False

        self.writer = SummaryWriter(log_dir=checkpoint_dir)

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter
        self.sample_plotter = sample_plotter

        self.num_iterations = num_iterations
        self.num_epoch = num_epoch
        self.skip_train_validation = skip_train_validation

    @classmethod
    def from_checkpoint(cls, resume, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders,
                        tensorboard_formatter=None, sample_plotter=None, **kwargs):
        logger.info(f"Loading checkpoint '{resume}'...")
        state = load_checkpoint(resume, model, optimizer)
        logger.info(
            f"Checkpoint loaded. Epoch: {state['epoch']}. Best val score: {state['best_eval_score']}. Num_iterations: {state['num_iterations']}")
        checkpoint_dir = os.path.split(resume)[0]

        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   torch.device(state['device']),
                   loaders, checkpoint_dir,
                   eval_score_higher_is_better=state['eval_score_higher_is_better'],
                   best_eval_score=state['best_eval_score'],
                   num_iterations=state['num_iterations'],
                   num_epoch=state['epoch'],
                   max_num_epochs=state['max_num_epochs'],
                   max_num_iterations=state['max_num_iterations'],
                   validate_after_iters=state['validate_after_iters'],
                   log_after_iters=state['log_after_iters'],
                   validate_iters=state['validate_iters'],
                   skip_train_validation=state.get('skip_train_validation', False),
                   tensorboard_formatter=tensorboard_formatter,
                   sample_plotter=sample_plotter, **kwargs)

    @classmethod
    def from_pretrained(cls, pre_trained, model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                        device, loaders,
                        max_num_epochs=100, max_num_iterations=int(1e5),
                        validate_after_iters=100, log_after_iters=100,
                        validate_iters=None, num_iterations=1, num_epoch=0,
                        eval_score_higher_is_better=True, best_eval_score=None,
                        tensorboard_formatter=None, sample_plotter=None,
                        skip_train_validation=False, **kwargs):
        logger.info(f"Logging pre-trained model from '{pre_trained}'...")
        load_checkpoint(pre_trained, model, None)
        if 'checkpoint_dir' not in kwargs:
            checkpoint_dir = os.path.split(pre_trained)[0]
        else:
            checkpoint_dir = kwargs.pop('checkpoint_dir')
        return cls(model, optimizer, lr_scheduler,
                   loss_criterion, eval_criterion,
                   device, loaders, checkpoint_dir,
                   eval_score_higher_is_better=eval_score_higher_is_better,
                   best_eval_score=best_eval_score,
                   num_iterations=num_iterations,
                   num_epoch=num_epoch,
                   max_num_epochs=max_num_epochs,
                   max_num_iterations=max_num_iterations,
                   validate_after_iters=validate_after_iters,
                   log_after_iters=log_after_iters,
                   validate_iters=validate_iters,
                   tensorboard_formatter=tensorboard_formatter,
                   sample_plotter=sample_plotter,
                   skip_train_validation=skip_train_validation, **kwargs)

    def fit(self):
        for _ in range(self.num_epoch, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epoch += 1
            ## -- Check memory
            # os.system('nvidia-smi')

        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train(self):
        """Trains the model for 1 epoch.

        Returns:
            True if the training should be terminated immediately, False otherwise
        """

        train_losses = RunningAverage()
        train_eval_scores = RunningAverage()

        # sets the model in training mode
        self.model.train()

        # for t in enumerate(self.loaders, start=0):
        for t in self.loaders['train']:
            logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                        f'Epoch [{self.num_epoch}/{self.max_num_epochs - 1}]')

            input, target = t
            input = input.type(torch.FloatTensor).to(self.device)
            target = target.type(torch.FloatTensor).to(self.device)

            # output, loss = self._forward_pass(input, target, weight)
            output, loss = self._forward_pass(input, target, None)
            train_losses.update(loss.item(), self._batch_size(input))

            # compute gradients and update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.num_iterations % self.validate_after_iters == 0:
                # set the model in eval mode
                self.model.eval()
                # evaluate on validation set
                if self.additional_eval:
                    val_loss, val_eval_score, val_precision, val_recall = self.validate()
                else:
                    val_loss, val_eval_score, _, _ = self.validate()


                # set the model back to training mode
                self.model.train()

                ## --- IMPORTANT MODIFICATION ---
                # ## uncomment if you need scheduler
                # adjust learning rate if necessary
                # if isinstance(self.scheduler, ReduceLROnPlateau):
                #     self.scheduler.step(val_eval_score)
                # else:
                #     self.scheduler.step()

                # log current learning rate in tensorboard
                self._log_lr()

                # update current eval score
                self.current_eval_score = val_eval_score
                self.current_eval_loss = val_loss

                # remember best validation metric
                if self.early_stopping_criterion == 'eval':
                    is_best = self._is_best_eval_score(val_eval_score)
                    if self.early_stopping:
                        if is_best:
                            self.best_eval_score_ = val_eval_score
                            self.early_stopping_count = 0
                        else:
                            self.early_stopping_count += 1

                        if self.early_stopping_count >= 10:
                            return True

                else:
                    is_best = self._is_best_eval_loss(val_loss)
                    if self.early_stopping:
                        if is_best:
                            self.best_eval_loss = val_loss
                            self.early_stopping_count = 0
                        else:
                            self.early_stopping_count += 1

                        if self.early_stopping_count >= 10:
                            return True

                # save checkpoint
                self._save_checkpoint(is_best)

                # update top 5 checkpoints
                if self.save_top5:
                    self._is_top5_eval_score(val_eval_score)

                # compute eval criterion
                if not self.skip_train_validation:
                    train_eval_score = self.eval_criterion(output, target)
                    if torch.is_tensor(train_eval_score):
                        train_eval_scores.update(train_eval_score.item(), self._batch_size(input))
                    else:
                        train_eval_scores.update(train_eval_score, self._batch_size(input))

                # log stats, params
                logger.info(
                f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)

                # log val loss w/ train loss
                self._log_stats_val_train_together('valwtrain', val_loss, val_eval_score, train_losses.avg, train_eval_scores.avg)

                # log output image
                if self.visualize_image:
                    self._log_images(input, target, output, 'train_')


            if self.num_iterations % self.log_after_iters == 0:
                logger.info(
                f'Training stats. Loss: {train_losses.avg}. Evaluation score: {train_eval_scores.avg}')
                self._log_stats('train', train_losses.avg, train_eval_scores.avg)
                if self.visualize_image:
                    self._log_images(input, target, output, 'train_')

            if self.should_stop():
                return True

            self.num_iterations += 1


        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = RunningAverage()
        val_scores = RunningAverage()
        val_precision = RunningAverage()
        val_recall = RunningAverage()

        if self.sample_plotter is not None:
            self.sample_plotter.update_current_dir()

        with torch.no_grad():
            for i, t in enumerate(self.loaders['val']):
                logger.info(f'Validation iteration {i}')

                input, target = t
                input = input.to(self.device).float()
                target = target.to(self.device).float()

                output, loss = self._forward_pass(input, target)
                val_losses.update(loss.item(), self._batch_size(input))

                # if model contains final_activation layer for normalizing logits apply it, otherwise
                # the evaluation metric will be incorrectly computed
                if hasattr(self.model, 'final_activation') and self.model.final_activation is not None:
                    output = self.model.final_activation(output)

                if i % 100 == 0:
                    if self.visualize_image:
                        self._log_images(input, target, output, 'val_')
                        # self._log_input_image_target_prediction(input, target, output, 'val_')


                eval_score = self.eval_criterion(output, target)
                if torch.is_tensor(eval_score):
                    val_scores.update(eval_score.item(), self._batch_size(input))
                else:
                    val_scores.update(eval_score, self._batch_size(input))

                logger.info(f'Validation finished. Loss: {val_losses.avg}. Evaluation score: {val_scores.avg}')
                self._log_stats('val', val_losses.avg, val_scores.avg)

                if self.additional_eval:
                    # precision
                    # print(target.detach().cpu().numpy())
                    precision = precision_score(target.detach().cpu().numpy(), output.detach().cpu().numpy()>0)
                    # print(target.detach().cpu().numpy()[:5])
                    # print(output.detach().cpu().numpy()[:5])
                    val_precision.update(precision, self._batch_size(input))
                    # recall
                    recall = recall_score(target.detach().cpu().numpy(), output.detach().cpu().numpy()>0)
                    val_recall.update(recall, self._batch_size(input))
                    self._log_addtional_stats('val', val_precision.avg, val_recall.avg)

                if self.sample_plotter is not None:
                    self.sample_plotter(i, input, output, target, 'val')

                if self.validate_iters is not None and self.validate_iters <= i:
                    # stop validation
                    break

            return val_losses.avg, val_scores.avg, val_precision.avg, val_recall.avg

    def _split_training_batch(self, t):
        def _move_to_device(input):
            if isinstance(input, tuple) or isinstance(input, list):
                return tuple([_move_to_device(x) for x in input])
            else:
                return input.to(self.device)

        t = _move_to_device(t)
        input, target = t

        return input, target

    def _forward_pass(self, input, target, weight=None):
        # forward pass
        output = self.model(input)

        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss


    def _forward_pass_multipleinput3(self, x1, x2, x3, target, weight=None):
        # forward pass
        output = self.model(x1, x2, x3)

        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss


    def _forward_pass_multipleinput4(self, x1, x2, x3, x4, target, weight=None):
        # forward pass
        output = self.model(x1, x2, x3, x4)

        # compute the loss
        if weight is None:
            loss = self.loss_criterion(output, target)
        else:
            loss = self.loss_criterion(output, target, weight)

        return output, loss


    def _is_best_eval_score(self, eval_score):
        if self.eval_score_higher_is_better:
            is_best = eval_score > self.best_eval_score
        else:
            is_best = eval_score < self.best_eval_score

        if is_best:
            logger.info(f'Saving new best evaluation metric: {eval_score}')
            self.best_eval_score = eval_score

        return is_best

    def _is_best_eval_loss(self, eval_loss):
        is_best = eval_loss < self.best_eval_loss
        if is_best:
            logger.info(f'Saving new best evaluation loss: {eval_loss}')
            self.best_eval_loss = eval_loss
        return is_best

    def _is_top5_eval_score(self, eval_score):

        if os.path.exists(os.path.join(self.checkpoint_dir, 'best_evaluation_iteration.npy')):
            best_evaluation = np.load(os.path.join(self.checkpoint_dir, 'best_evaluation.npy')).astype('float')
            best_evaluation_iteration = np.load(os.path.join(self.checkpoint_dir, 'best_evaluation_iteration.npy')).astype('int')
        else:
            best_evaluation = np.array([0, 0, 0, 0, 0]).astype('float')
            best_evaluation_iteration = np.array([0, 0, 0, 0, 0]).astype('int')


        if self.eval_score_higher_is_better:
            delete_iteration_idx = np.argsort(best_evaluation - eval_score)[0]
            is_top5_eval_score = np.sum( best_evaluation < eval_score )

        else:
            delete_iteration_idx = np.argsort(eval_score - best_evaluation )[0]
            is_top5_eval_score = np.sum( best_evaluation > eval_score )

        if is_top5_eval_score:
            logger.info(f'Delete old best evaluation metric checkpoint: ')

            # delete old one first
            oldfname = os.path.join(self.checkpoint_dir, f'best_checkpoint'
                                                         f'{best_evaluation_iteration[delete_iteration_idx]}.pytorch')
            if os.path.exists(oldfname):
                os.system(f'rm {oldfname}')

            # now update
            best_evaluation_iteration[delete_iteration_idx] = self.num_iterations
            best_evaluation[delete_iteration_idx] = eval_score
            np.save(os.path.join(self.checkpoint_dir, 'best_evaluation.npy'), best_evaluation)
            np.save( os.path.join(self.checkpoint_dir, 'best_evaluation_iteration.npy'), best_evaluation_iteration)

            newfname = os.path.join(self.checkpoint_dir, f'best_checkpoint{self.num_iterations}.pytorch')
            if isinstance(self.model, nn.DataParallel):
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()

            save_dict = {
                'epoch': self.num_epoch + 1,
                'num_iterations': self.num_iterations,
                'model_state_dict': state_dict,
                'best_eval_score': eval_score,
                'eval_score_higher_is_better': self.eval_score_higher_is_better,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'device': str(self.device),
                'max_num_epochs': self.max_num_epochs,
                'max_num_iterations': self.max_num_iterations,
                'validate_after_iters': self.validate_after_iters,
                'log_after_iters': self.log_after_iters,
                'validate_iters': self.validate_iters,
                'skip_train_validation': self.skip_train_validation
            }

            logger.info(f"Save new best evaluation metric checkpoint to'{newfname}'")
            torch.save(save_dict, newfname)

    def _save_checkpoint(self, is_best):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        save_checkpoint({
            'epoch': self.num_epoch + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'best_eval_loss': self.best_eval_loss,
            'current_eval_score': self.current_eval_score,
            'current_eval_loss': self.current_eval_loss,
            'eval_score_higher_is_better': self.eval_score_higher_is_better,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'device': str(self.device),
            'max_num_epochs': self.max_num_epochs,
            'max_num_iterations': self.max_num_iterations,
            'validate_after_iters': self.validate_after_iters,
            'log_after_iters': self.log_after_iters,
            'validate_iters': self.validate_iters,
            'skip_train_validation': self.skip_train_validation
        }, is_best, checkpoint_dir=self.checkpoint_dir,
            logger=logger)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase, loss_avg, eval_score_avg):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_eval_score_avg': eval_score_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_addtional_stats(self, phase, precision, recall):
        tag_value = {
            f'{phase}_precision': precision,
            f'{phase}_recall': recall
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)


    def _log_stats_val_train_together(self, phase, val_loss, val_score, train_loss, train_score):
        tag_value = {
            f'{phase}_loss_avg': [val_loss, train_loss],
            f'{phase}_eval_score_avg': [val_score, train_score]
        }

        for tag, value in tag_value.items():
            self.writer.add_scalars(tag, {
                'val': value[0],
                'train': value[1],
            }, self.num_iterations)



    def _log_stats_multipleLoss(self, phase, losses, loss_total, eval_score_avg):
        tag_value = {}
        for l in range(len(losses)):
            tag_value[f'{phase}_loss{l}_avg'] =  losses[l]

        tag_value[f'{phase}_loss_avg'] = loss_total
        tag_value[f'{phase}_eval_score_avg'] = eval_score_avg

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_images(self, input, target, prediction, prefix=''):
        inputs_map = {
            'inputs': input,
            'targets': target,
            'predictions': prediction
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(prefix + tag, image, self.num_iterations, dataformats='CHW')

    def _log_input_image_target_prediction(self, input, target, prediction, prefix=''):
        inputs_map = {
            'inputs': input
        }
        img_sources = {}
        for name, batch in inputs_map.items():
            if isinstance(batch, list) or isinstance(batch, tuple):
                for i, b in enumerate(batch):
                    img_sources[f'{name}{i}'] = b.data.cpu().numpy()
            else:
                img_sources[name] = batch.data.cpu().numpy()

        for name, batch in img_sources.items():
            for tag, image in self.tensorboard_formatter(name, batch):
                self.writer.add_image(f't:{target}/p:{prediction}'+ prefix + tag, image, self.num_iterations, dataformats='CHW')

    @staticmethod
    def _batch_size(input):
        if isinstance(input, list) or isinstance(input, tuple):
            return input[0].size(0)
        else:
            return input.size(0)


class TrainerBuilder:
    @staticmethod
    def build(config):
        # Create the model
        model = get_model(config['model'])
        # use DataParallel if more than 1 GPU available
        device = config['device']
        if torch.cuda.device_count() > 1 and not device.type == 'cpu':
            model = nn.DataParallel(model)
            logger.info(f'Using {torch.cuda.device_count()} GPUs for training')

        # put the model on GPUs
        logger.info(f"Sending the model to '{config['device']}'")
        # model = model().to(device)
        model = model.to(device)

        # Log the number of learnable parameters
        logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

        # Create data loaders
        if 'csvfname' in config['loaders'].keys() and 'imagedir' in config['loaders'].keys():
            trainloader = get_loader(config['loaders'])(csvfname=config['loaders']['csvfname'],
                                                        imagedir=config['loaders']['imagedir'],
                                                        maskdir=config['loaders']['maskdir'],
                                                        mode='train', config_mode=config['loaders']['train'])
            valloader = get_loader(config['loaders'])(csvfname=config['loaders']['csvfname'],
                                                      imagedir=config['loaders']['imagedir'],
                                                      maskdir=config['loaders']['maskdir'],
                                                      mode='val', config_mode=config['loaders']['val'])

            if 'test' in config['loaders'].keys():
                testloader = get_loader(config['loaders'])(csvfname=config['loaders']['csvfname'],
                                                      imagedir=config['loaders']['imagedir'],
                                                      maskdir=config['loaders']['maskdir'],
                                                      mode='test', config_mode=config['loaders']['test'])


        elif 'csvfname' in config['loaders'].keys() and 'imagedir' not in config['loaders'].keys():
            trainloader = get_loader(config['loaders'])(csvfname=config['loaders']['csvfname'],
                                                        mode='train', config_mode=config['loaders']['train'])
            valloader = get_loader(config['loaders'])(csvfname=config['loaders']['csvfname'],
                                                      mode='val', config_mode=config['loaders']['val'])
            if 'test' in config['loaders'].keys():
                testloader = get_loader(config['loaders'])(csvfname=config['loaders']['csvfname'],
                                                      mode='test', config_mode=config['loaders']['test'])

        else:
            trainloader = get_loader(config['loaders'])(mode='train', config_mode=config['loaders']['train'])
            valloader = get_loader(config['loaders'])(mode='val', config_mode=config['loaders']['val'])
            if 'test' in config['loaders'].keys():
                testloader = get_loader(config['loaders'])(mode='test', config_mode=config['loaders']['test'])

        batch_size = config['loaders']['batch_size']

        # train set weight for weigthed sampler
        if config['loaders']['train']['weighted_sampler']:
            train_weight = torch.tensor((1 / np.bincount(trainloader.target.astype('int'))), dtype=torch.float)

        manual_seed = config.get('manual_seed', None)
        if manual_seed is not None:
            g = torch.Generator()
            g.manual_seed(manual_seed)
            # set_random_seed(manual_seed)


            loaders = {}
            for train_val_test in ['train', 'val']: # , 'test'
                loadername = eval(f'{train_val_test}loader')
                # if loadername in locals():
                if config['loaders'][train_val_test]['weighted_sampler']:
                    samples_weight = torch.tensor(
                        [train_weight[t] for t in np.array(loadername.target).astype('int')])
                    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight,
                                                                                   len(loadername))

                    loaders[train_val_test] = torch.utils.data.DataLoader(loadername, batch_size=batch_size, shuffle=False, sampler=sampler,
                                                            drop_last=False, worker_init_fn=seed_worker, generator=g)

                else:
                    loaders[train_val_test] = torch.utils.data.DataLoader(loadername, batch_size=batch_size, shuffle=False,
                                                                   drop_last=False,
                                                                   worker_init_fn=seed_worker, generator=g)

        else:
            loaders = {}
            for train_val_test in ['train', 'val']: # , 'test'
                loadername = eval(f'{train_val_test}loader')

                if config['loaders'][train_val_test]['weighted_sampler']:
                    samples_weight = torch.tensor(
                        [train_weight[t] for t in np.array(loadername.target).astype('int')])
                    sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight,
                                                                                   len(loadername))

                    loaders[train_val_test] = torch.utils.data.DataLoader(loadername, batch_size=batch_size, shuffle=False, sampler=sampler,
                                                            drop_last=False)

                else:
                    loaders[train_val_test] = torch.utils.data.DataLoader(loadername, batch_size=batch_size, shuffle=False,
                                                                   drop_last=False)


        # Create loss criterion
        if config['loss']['class_weight'] > 0:
            # Calculate the weights for each class so that we can balance the data
            # class_weights = class_weight.compute_class_weight('balanced',
            #                                                   np.unique(trainloader.target),
            #                                                   np.array(trainloader.target))
            class_weights = config['loss']['class_weight']
            loss_criterion = get_loss(config['loss'])(class_weights=torch.tensor(class_weights).to(device))
        else:
            loss_criterion = get_loss(config['loss'])()

        # Create evaluation metric
        eval_criterion = get_loss(config['evaluation'])()

        # Create the optimizer
        optimizer = create_optimizer(config['optimizer'], model)

        # Create learning rate adjustment strategy
        lr_scheduler = create_lr_scheduler(config.get('lr_scheduler', None), optimizer)

        # Create model trainer
        trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                  loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders)

        return trainer

