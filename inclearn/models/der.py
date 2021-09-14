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
        self.aux_weight = args["classifier_config"].get("aux_weight", 1.0)
        self._increments = []
        self.merge = args.get("merge", False)
        self.with_class = args["classifier_config"].get("with_class", False)

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
        if self._is_task_level:
            if self.with_class:
                self._network.add_classes(self._task_increment, with_class_classes=self._task_size)
            else:
                self._network.add_classes(self._task_increment)
        else:
            self._network.add_classes(self._task_size)
        logger.info("Now {} examplars per class.".format(self._memory_per_class))

        self.set_optimizer()

    #@property
    #def _memory_per_class(self):
    #    """Returns the number of examplars per class."""
    #    if self._fixed_memory:
    #        return self._memory_size // (self._n_tasks if self._is_task_level else self._total_n_classes)
    #    return self._memory_size // (self._n_current_task if self._is_task_level else self._n_classes)

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
        #if self._is_task_level and self._task == 0:
        #logger.log("Task level model skip the first task")
        self._training_step(train_loader, val_loader, 0, self._n_epochs, temperature=self.temperature)

        self._post_processing_type = None

        if self._finetuning_config and self._task > (1 if self.merge else 0):
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
            elif self._finetuning_config["sampling"] == "decouple":
                data_memory, targets_memory = self.get_memory()
                loader = self.inc_dataset.get_balanced_memory_loader(data_memory, targets_memory,
                                                                     low_range=self._n_classes - self._task_size,
                                                                     high_range=self._n_classes,
                                                                     is_task=self._is_task_level)

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
            finetune_scheduling = []
            for e in self._finetuning_config["scheduling"]:
                finetune_scheduling.append(e + self._n_epochs)
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                                   finetune_scheduling,
                                                                   gamma=self._finetuning_config["lr_decay"])

            self._network.classifier.classifier.reset_parameters()
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

            if epoch == self._warmup_config["total_epoch"]:
                training_network.classifier.classifier.reset_parameters()
                training_network.classifier.aux_classifier.reset_parameters()

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
                targets_class = input_dict["targets"] if self.with_class else None
                memory_flags = input_dict["memory_flags"]

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                self._optimizer.zero_grad()
                old_classes = targets < (self._n_classes - self._task_size)
                new_classes = targets >= (self._n_classes - self._task_size)
                loss_ce, loss_aux, loss_class = self._forward_loss(
                    training_network,
                    inputs,
                    targets,
                    memory_flags,
                    targets_class=targets_class,
                    old_classes=old_classes,
                    new_classes=new_classes,
                    temperature=temperature,
                    example=(i == len(prog_bar) - 1)
                )

                if epoch >= self._n_epochs or self._task == 0:
                    #Fine tuning or the first task
                    loss = loss_ce
                else:
                    if self.aux_weight == 0 or self._is_task_level:
                        loss = loss_ce
                    else:
                        loss = loss_ce + self.aux_weight * loss_aux
                if self.with_class:
                    loss += loss_class

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
                #self._data_memory, self._targets_memory, self._herding_indexes, self._class_means = self.build_examplars(
                #    self.inc_dataset, self._herding_indexes
                #)
                ypred, ytrue = self._eval_task(val_loader)
                ypred = ypred.argmax(axis=-1)
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
        targets_class=None,
        old_classes=None, new_classes=None, accu=None, new_accu=None, old_accu=None, temperature=1.0, example=False):
        inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
        if self.with_class:
            targets_class = targets_class.to(self._device, non_blocking=True)

        outputs = training_network(inputs)
        if accu is not None:
            accu.add(outputs['logit'], targets)
            # accu.add(logits.detach(), targets.cpu().numpy())
        # if new_accu is not None:
        #     new_accu.add(logits[new_classes].detach(), targets[new_classes].cpu().numpy())
        # if old_accu is not None:
        #     old_accu.add(logits[old_classes].detach(), targets[old_classes].cpu().numpy())
        loss, aux_loss, loss_class = self._compute_loss(inputs, targets, outputs, old_classes, new_classes, targets_class=targets_class, temperature=temperature)

        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        if not utils.check_loss(aux_loss):
            raise ValueError("A aux_loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()
        self._metrics["aux_loss"] += -1 if self.aux_weight == 0 else aux_loss.item()
        self._metrics["loss_class"] += loss_class.item() if self.with_class else -1

        pred = outputs["logit"].argmax(dim=-1)
        acc = (pred == targets).sum()
        acc = acc / targets.shape[0] * 100
        self._metrics["acc"] += acc.item()

        if self.with_class:
            pred = outputs['logit_class'].argmax(dim=-1)
            acc = (pred == targets_class).sum()
            acc = acc / targets_class.shape[0] * 100
            self._metrics["acc_class"] += acc.item()

        if example and False:
            print()
            print("{} pred".format(pred.tolist()[:60]))
            print("{} targets".format(targets.tolist()[:60]))

        return loss, aux_loss, loss_class

    def _compute_loss(self, inputs, targets, outputs, old_classes, new_classes, targets_class=None, temperature=1.0):
        loss = F.cross_entropy(outputs['logit'] / temperature, targets)

        if outputs['aux_logit'] is not None:
            aux_targets = targets.clone()
            if self.aux_nplus1:
                aux_targets[old_classes] = 0
                delta = self._task if self._is_task_level else sum(self.inc_dataset.increments[:self._task])
                aux_targets[new_classes] -= delta - 1
            aux_loss = F.cross_entropy(outputs['aux_logit'], aux_targets)
        else:
            aux_loss = torch.zeros([1]).to(self._device)

        loss_class = F.cross_entropy(outputs['logit_class'] / temperature, targets_class) if self.with_class else None

        return loss, aux_loss, loss_class

    def _eval_task(self, data_loader):
        ypreds, ytrue = self._compute_accuracy_by_netout(data_loader)

        return ypreds, ytrue

    def _compute_accuracy_by_netout(self, data_loader):
        preds, targets = [], []
        self._network.eval()
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


    def build_examplars(
        self, inc_dataset, herding_indexes, memory_per_class=None, data_source="train"
    ):
        logger.info("Building & updating memory.")
        memory_per_class = memory_per_class or self._memory_per_class
        herding_indexes = copy.deepcopy(herding_indexes)

        data_memory, targets_memory = [], []
        class_means = np.zeros((self._n_classes, self._network.features_dim))

        for class_idx in range(self._n_classes):
            # We extract the features, both normal and flipped:
            inputs, loader = inc_dataset.get_custom_loader(
                class_idx, mode="test", data_source=data_source
            )
            features, targets = utils.extract_features(self._network, loader)

            if class_idx >= self._n_classes - self._task_size:
                # New class, selecting the examplars:
                if self._herding_selection["type"] == "icarl":
                    selected_indexes = herding.icarl_selection(features, memory_per_class)
                elif self._herding_selection["type"] == "closest":
                    selected_indexes = herding.closest_to_mean(features, memory_per_class)
                elif self._herding_selection["type"] == "random":
                    selected_indexes = herding.random(features, memory_per_class)
                elif self._herding_selection["type"] == "first":
                    selected_indexes = np.arange(memory_per_class)
                elif self._herding_selection["type"] == "kmeans":
                    selected_indexes = herding.kmeans(
                        features, memory_per_class, k=self._herding_selection["k"]
                    )
                elif self._herding_selection["type"] == "confusion":
                    selected_indexes = herding.confusion(
                        *self._last_results,
                        memory_per_class,
                        class_id=class_idx,
                        minimize_confusion=self._herding_selection["minimize_confusion"]
                    )
                elif self._herding_selection["type"] == "var_ratio":
                    selected_indexes = herding.var_ratio(
                        memory_per_class, self._network, loader, **self._herding_selection
                    )
                elif self._herding_selection["type"] == "mcbn":
                    selected_indexes = herding.mcbn(
                        memory_per_class, self._network, loader, **self._herding_selection
                    )
                else:
                    raise ValueError(
                        "Unknown herding selection {}.".format(self._herding_selection)
                    )

                herding_indexes.append(selected_indexes)

            # Reducing examplars:
            try:
                selected_indexes = herding_indexes[class_idx][:memory_per_class]
                herding_indexes[class_idx] = selected_indexes
            except:
                import pdb
                pdb.set_trace()

            data_memory.append(inputs[selected_indexes])
            targets_memory.append(targets[selected_indexes])


        data_memory = np.concatenate(data_memory)
        targets_memory = np.concatenate(targets_memory)

        return data_memory, targets_memory, herding_indexes, class_means


def _clean_list(l):
    for i in range(len(l)):
        l[i] = None
