import logging
import os
import pickle
import collections
from tqdm import tqdm
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from inclearn.lib import calibration, herding, losses, utils, factory, schedulers, network
from inclearn.lib.network import hook
from inclearn.models.base import IncrementalLearner
from inclearn.lib.data import samplers

EPSILON = 1e-8

logger = logging.getLogger(__name__)


class CovNorm(IncrementalLearner):
    """Implements CovNorm.
    """

    def __init__(self, args:dict):
        logger.info("Initializing CovNorm")
        args["convnet_config"]["device"] = args["device"][0]
        super().__init__(args)
        self._disable_progressbar = args.get("no_progressbar", False)

        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]

        self._scheduling = args["scheduling"]
        self._lr_decay = args["lr_decay"]

        self._warmup_config = args.get("warmup", {})
        if self._warmup_config and self._warmup_config["total_epoch"] > 0:
            self._lr /= self._warmup_config["multiplier"]

        self._eval_every_x_epochs = args.get("eval_every_x_epochs")
        self._rotations_config = args.get("rotations_config", {})
        self._n_classes = 0
        self.task_max = args.get("task_max", None)
        self._increments = []
        self.merge = args.get("merge", False)
        self.dynamic_wd = args.get("dynamic_weight_decay", False)
        self.temperature = args.get("temperature", 1.0)
        self._early_stopping = args.get("early_stopping", {})

        self._network = network.BasicNet(
            args["convnet"],
            convnet_kwargs=args.get("convnet_config", {
                "pretrained": True,
                "pretrained_model": "ImageNet-800",
                "pretrained_mode_path": "/home/zhiyuan/CIL-CN/saved_models/model_best.pth.tar"
            }),
            classifier_kwargs=args.get("classifier_config", {
                "type": "multi",
                "use_bias": True
            }),
            device=self._device,
            extract_no_act=args.get("extract_no_act", True),
            classifier_no_act=False,
            rotations_predictor=bool(self._rotations_config)
        )

    def eval(self):
        if len(self._multiple_devices) > 1:
            self.network.module.covnet.set_status(self._task, False)
        else:
            self.network.convnet.set_status(self._task, False)
        self._network.eval()

    def train(self):
        self._network.train()
        if len(self._multiple_devices) > 1:
            self.network.module.covnet.set_status(self._task, True)
        else:
            self.network.convnet.set_status(self._task, True)

    def _before_task(self, train_loader, val_loader):
        self._n_classes += self._task_size
        self._increments.append(self._task_size)
        if self._is_task_level:
            self._network.add_classes(1)
        else:
            self._network.add_classes(self._task_size)

        self.set_optimizer()

    def set_optimizer(self, lr=None):
        if lr is None:
            lr = self._lr

        if self.dynamic_wd:
            # used in BiC official implementation
            weight_decay = self._weight_decay * self.task_max / (self._task + 1)
        else:
            weight_decay = self._weight_decay
        logger.info("Step {} weight decay {:.5f}".format(self._task, weight_decay))
        self.weight_decay = weight_decay

        if self._task > (1 if self.merge else 0):
            for i in range(self._task):
                self.network.convnet.freeze_adapter(i)
                self.network.classifier.freeze_classifier(i)
            self.network.convnet.freeze_backbone()


        self._optimizer = factory.get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()),
                                                self._opt_name, lr, weight_decay)

        base_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self._optimizer, self._scheduling, gamma=self._lr_decay
        )

        if self._warmup_config:
            if self._warmup_config.get("only_first_step", True) and self._task != 0:
                pass
            else:
                logger.info("Using WarmUp")
                self._scheduler = schedulers.GradualWarmupScheduler(
                    optimizer=self._optimizer,
                    after_scheduler=base_scheduler,
                    **self._warmup_config
                )
        else:
            self._scheduler = base_scheduler

    def _train_task(self, train_loader, val_loader):
        logger.debug("nb {}.".format(len(train_loader.dataset)))
        nb_epochs = self._n_epochs
        temperature = self.temperature
        record_bn = True

        best_epoch, best_acc = -1, -1.
        wait = 0

        grad, act = None, None
        if len(self._multiple_devices) > 1:
            logger.info("Duplicating model on {} gpus.".format(len(self._multiple_devices)))
            training_network = nn.DataParallel(self._network, self._multiple_devices)
            if self._network.gradcam_hook:
                grad, act, back_hook, for_hook = hook.get_gradcam_hook(training_network)
                training_network.module.convnet.last_conv.register_backward_hook(back_hook)
                training_network.module.convnet.last_conv.register_forward_hook(for_hook)
        else:
            training_network = self._network

        self._optimizer.zero_grad()
        self._optimizer.step()

        for epoch in range(nb_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / nb_epochs

            if epoch == nb_epochs - 1 and record_bn and len(self._multiple_devices) == 1 and \
                    hasattr(training_network.convnet, "record_mode"):
                logger.info("Recording BN means & vars for MCBN...")
                training_network.convnet.clear_records()
                training_network.convnet.record_mode()


            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                self.train()
                inputs = input_dict["inputs"]
                targets = input_dict["targets_task"] if self._is_task_level else input_dict["targets"]
                targets_class = None

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                self._optimizer.zero_grad()
                loss = self._forward_loss(
                    training_network,
                    inputs,
                    targets,
                    targets_class=targets_class,
                    temperature=temperature,
                    example=(i == len(prog_bar) - 1)
                )

                if not utils.check_loss(loss):
                    import pdb
                    pdb.set_trace()


                loss.backward()
                self._optimizer.step()

                self._print_metrics(prog_bar, epoch, nb_epochs, i)


            logger.info("Task {} / {}, epoch {} / {}, loss {}".format(self._task + 1, self._n_tasks,
                                                                      epoch + 1, nb_epochs,
                                                                      round(self._metrics["loss"] / i, 3)))

            if "total_epoch" in self._warmup_config and epoch == self._warmup_config["total_epoch"] - 1:
                if len(self._multiple_devices) > 1:
                    training_network.module.classifier.classifier.reset_parameters()
                else:
                    training_network.classifier.classifier.reset_parameters()

            if self._scheduler:
                self._scheduler.step(epoch)

            if self._eval_every_x_epochs and epoch != 0 and epoch % self._eval_every_x_epochs == 0:
                self.eval()
                ypred, ytrue = self._eval_task(val_loader)
                ypred = ypred.argmax(axis=-1)
                acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
                logger.info("Val accuracy: {}".format(acc))
                self.train()

                if acc > best_acc:
                    best_epoch = epoch
                    best_acc = acc
                    wait = 0
                else:
                    wait += 1

                if self._early_stopping and self._early_stopping["patience"] > wait:
                    logger.warning("Early stopping!")
                    break

            if self._eval_every_x_epochs:
                logger.info("Best accuracy reached at epoch {} with {}%.".format(best_epoch, best_acc))

            if len(self._multiple_devices) == 1 and hasattr(training_network.convnet, "record_mode"):
                training_network.convnet.normal_mode()



    def _forward_loss(self,
        training_network,
        inputs,
        targets,
        targets_class=None,
        accu=None, new_accu=None, old_accu=None, temperature=1.0, example=False):
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
        bias = sum(self._increments[:-1])
        targets -= bias

        outputs = training_network(inputs)
        if accu is not None:
            accu.add(outputs['logit'], targets)
        loss = F.cross_entropy(outputs['logit'] / temperature, targets)
        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()

        pred = outputs["logit"].argmax(dim=-1)
        acc = (pred == targets).sum()
        acc = acc / targets.shape[0] * 100
        self._metrics["acc"] += acc.item()

        if example and False:
            print()
            print("{} pred".format(pred.tolist()[:60]))
            print("{} targets".format(targets.tolist()[:60]))

        return loss


    def _eval_task(self, data_loader):
        ypreds, ytrue = self._compute_accuracy_by_netout(data_loader)
        print(ypreds.shape)
        print(ypreds.argmax(axis=1)[:100])
        print(ytrue[:100])
        #c = input()

        return ypreds, ytrue

    def _compute_accuracy_by_netout(self, data_loader):
        preds, targets = [], []
        self.eval()
        with torch.no_grad():
            for input_dict in data_loader:
                inputs = input_dict["inputs"]
                lbls = input_dict["targets_task"] if self._is_task_level else input_dict["targets"]
                inputs = inputs.to(self._device, non_blocking=True)
                _preds = self._network(inputs)['logit']
                preds.append(_preds.detach().cpu().numpy())
                targets.append(lbls.long().cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        return preds, targets

    def _print_metrics(self, prog_bar, epoch, nb_epochs, nb_batches):
        pretty_metrics = ", ".join(
            "{}: {}".format(metric_name, round(metric_value / nb_batches, 3))
            for metric_name, metric_value in self._metrics.items()
        )

        prog_bar.set_description(
            "T{}/{}, E{}/{} => {}".format(
                self._task + 1, self._n_tasks, epoch + 1, nb_epochs, pretty_metrics
            )
        )


def _clean_list(l):
    for i in range(len(l)):
        l[i] = None
