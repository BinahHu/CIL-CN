import logging
import os
import pickle
import collections
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from inclearn.lib import calibration, herding, losses, utils, factory, schedulers
from inclearn.lib.network import hook
from inclearn.models.icarl import ICarl
from inclearn.lib.data import samplers

EPSILON = 1e-8

logger = logging.getLogger(__name__)


class DER(ICarl):
    """Implements DER.

    * https://arxiv.org/abs/1905.13260
    """

    def __init__(self, args):
        logger.info("Initializing DER")
        args["convnet_config"]["device"] = args["device"][0]
        super().__init__(args)
        self.dynamic_wd = args.get("dynamic_weight_decay", False)
        self.task_max = args.get("task_max", None)
        self.aux_nplus1 = args["classifier_config"].get("aux_nplus1", True)
        self._finetuning_config = args.get("finetuning_config")
        self.temperature = args.get("temperature", 1.0)

        self._increments = []

    def eval(self):
        self._network.eval()

    def train(self):
        self._network.train()
        self._network.convnet.convnets[-1].train()
        if self._task >= 1:
            for i in range(self._task):
                self._network.convnet.convnets[i].eval()

    def _before_task(self, train_loader, val_loader):
        self._n_classes += self._task_size
        self._increments.append(self._task_size)
        self._network.add_classes(self._task_size)
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

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

        if self._task > 0:
            for i in range(self._task):
                for p in self.network.convnet.convnets[i].parameters():
                    p.requires_grad = False

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
        self._training_step(train_loader, val_loader, 0, self._n_epochs, temperature=self.temperature)

        self._post_processing_type = None

        if self._finetuning_config and self._task != 0:
            logger.info("Fine-tuning")
            if self._finetuning_config["scaling"]:
                logger.info(
                    "Custom fine-tuning scaling of {}.".format(self._finetuning_config["scaling"])
                )
                self._post_processing_type = self._finetuning_config["scaling"]

            if self._finetuning_config["sampling"] == "undersampling":
                self._data_memory, self._targets_memory, _, _ = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                loader = self.inc_dataset.get_memory_loader(*self.get_memory())
            elif self._finetuning_config["sampling"] == "oversampling":
                _, loader = self.inc_dataset.get_custom_loader(
                    list(range(self._n_classes - self._task_size, self._n_classes)),
                    memory=self.get_memory(),
                    mode="train",
                    sampler=samplers.MemoryOverSampler
                )

            if self._finetuning_config["tuning"] == "all":
                parameters = self._network.parameters()
            elif self._finetuning_config["tuning"] == "convnet":
                parameters = self._network.convnet.parameters()
            elif self._finetuning_config["tuning"] == "classifier":
                parameters = self._network.classifier.parameters()
            elif self._finetuning_config["tuning"] == "classifier_scale":
                parameters = [
                    {
                        "params": self._network.classifier.parameters(),
                        "lr": self._finetuning_config["lr"]
                    }, {
                        "params": self._network.post_processor.parameters(),
                        "lr": self._finetuning_config["lr"]
                    }
                ]
            else:
                raise NotImplementedError(
                    "Unknwown finetuning parameters {}.".format(self._finetuning_config["tuning"])
                )

            self._optimizer = factory.get_optimizer(
                parameters, self._opt_name, self._finetuning_config["lr"], self._finetuning_config["weight_decay"]
            )
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                                   self._finetuning_config["scheduling"],
                                                                   gamma=self._finetuning_config["lr_decay"])
            self._training_step(
                loader,
                val_loader,
                self._n_epochs,
                self._n_epochs + self._finetuning_config["epochs"],
                record_bn=False,
                temperature=self._finetuning_config["temperature"]
            )


    def _training_step(
            self, train_loader, val_loader, initial_epoch, nb_epochs, record_bn=True, clipper=None, temperature=1
    ):

        #utils.display_weight_norm(logger, self._network, self._increments, "Initial trainset")
        #utils.display_feature_norm(logger, self._network, train_loader, self._n_classes,
        #                           self._increments, "Initial trainset")

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

        for epoch in range(initial_epoch, nb_epochs):
            self._metrics = collections.defaultdict(float)

            self._epoch_percent = epoch / (nb_epochs - initial_epoch)

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
                inputs, targets = input_dict["inputs"], input_dict["targets"]
                memory_flags = input_dict["memory_flags"]

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                self._optimizer.zero_grad()
                old_classes = targets < (self._n_classes - self._task_size)
                new_classes = targets >= (self._n_classes - self._task_size)
                loss_ce, loss_aux = self._forward_loss(
                    training_network,
                    inputs,
                    targets,
                    memory_flags,
                    old_classes=old_classes,
                    new_classes=new_classes,
                    temperature=temperature
                )

                if epoch > self._n_epochs:
                    #Fine tuning
                    loss = loss_ce
                else:
                    loss = loss_ce + loss_aux

                if not utils.check_loss(loss):
                    import pdb
                    pdb.set_trace()

                loss.backward()
                self._optimizer.step()

                self._print_metrics(prog_bar, epoch, nb_epochs, i)

            if self._scheduler:
                self._scheduler.step(epoch)

            if self._eval_every_x_epochs and epoch != 0 and epoch % self._eval_every_x_epochs == 0:
                self._network.eval()
                self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
                    self.inc_dataset, self._herding_indexes
                )
                ytrue, ypred = self._eval_task(val_loader)
                acc = 100 * round((ypred == ytrue).sum() / len(ytrue), 3)
                logger.info("Val accuracy: {}".format(acc))
                self._network.train()

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

        #utils.display_weight_norm(logger, self._network, self._increments, "After training")
        #utils.display_feature_norm(logger, self._network, train_loader, self._n_classes,
        #                           self._increments, "Trainset")

    def _forward_loss(self,
        training_network,
        inputs,
        targets,
        memory_flags,
        old_classes=None, new_classes=None, accu=None, new_accu=None, old_accu=None, temperature=1.0):
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)

        outputs = training_network(inputs)
        if accu is not None:
            accu.add(outputs['logit'], targets)
            # accu.add(logits.detach(), targets.cpu().numpy())
        # if new_accu is not None:
        #     new_accu.add(logits[new_classes].detach(), targets[new_classes].cpu().numpy())
        # if old_accu is not None:
        #     old_accu.add(logits[old_classes].detach(), targets[old_classes].cpu().numpy())
        return self._compute_loss(inputs, targets, outputs, old_classes, new_classes, temperature=temperature)

    def _compute_loss(self, inputs, targets, outputs, old_classes, new_classes, temperature=1):
        loss = F.cross_entropy(outputs['logit'] / temperature, targets)

        if outputs['aux_logit'] is not None:
            aux_targets = targets.clone()
            if self.aux_nplus1:
                aux_targets[old_classes] = 0
                aux_targets[new_classes] -= sum(self.inc_dataset.increments[:self._task]) - 1
            aux_loss = F.cross_entropy(outputs['aux_logit'], aux_targets)
        else:
            aux_loss = torch.zeros([1]).to(self._device)

        return loss, aux_loss



def _clean_list(l):
    for i in range(len(l)):
        l[i] = None
