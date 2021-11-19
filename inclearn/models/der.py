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
        self.merge2 = args.get("merge_two", False)
        args["convnet_config"]["device"] = args["device"][0]
        args["classifier_config"]["merge_two"] = self.merge2
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
        self._task_level_herding = args.get("task_level_herding", False)
        self.shortcut_epochs = args.get("shortcut_epochs", 0)
        self.with_super_split = args["convnet_config"].get("with_super_split", False)
        self.max_super_split = args["convnet_config"].get("max_super_split", 0)
        self.use_oracle_super_label = args["convnet_config"].get("use_oracle_super_label", True)
        self.use_soft_mask = args["convnet_config"].get("use_soft_mask", False)
        self.lambda_s = args.get("lambda_s", 0)
        self.convnet_type = args.get("convnet", "multinet")

        self._is_consine_scheduler = False
        if self._finetuning_config:
            if isinstance(self._finetuning_config["scheduling"], str):
                if self._finetuning_config["scheduling"] == "cosine":
                    self._is_consine_scheduler = True


    def eval(self):
        self._network.eval()

    def train(self):
        true_task = self._task
        if self.merge2:
            assert self._task % 2 == 1
            true_task = self._task // 2
        self._network.train()
        if self.convnet_type == "resnet18_adapter":
            self._network.convnet.train()
            if true_task >= 1:
                for i in range(self._task):
                    self._network.convnet.freeze_adapter(i)
        else:
            self._network.convnet.convnets[-1].train()
            if true_task >= 1:
                for i in range(true_task):
                    self._network.convnet.convnets[i].eval()

    def _before_task(self, train_loader, val_loader):
        #c = input()
        self._n_classes += self._task_size
        if self.merge2:
            self._increments.append(self.first_inc)
            self._increments.append(self._task_size - self.first_inc)
        else:
            self._increments.append(self._task_size)
        if self._is_task_level:
            if self.with_class:
                self._network.add_classes(1, with_class_classes=self._task_size)
            else:
                if self.merge and self._task == 1:
                    self._network.add_classes(1)
                    self._network.add_classes(1)
                elif self.merge2:
                    self._network.classifier.ntask += 1
                    self._network.add_classes(2)
                else:
                    self._network.add_classes(1)
        else:
            if self.with_class:
                self._network.add_classes(self._task_size, with_class_classes=1)
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
            #weight_decay = self._weight_decay * (self._task + 1) / self.task_max
        else:
            weight_decay = self._weight_decay
        logger.info("Step {} weight decay {:.5f}".format(self._task, weight_decay))
        self.weight_decay = weight_decay

        true_task = self._task
        if self.merge2:
            assert self._task % 2 == 1
            true_task  = self._task // 2

        if true_task > (1 if self.merge else 0):
            if self.convnet_type == "resnet18_adapter":
                self.network.convnet.freeze_adapter(true_task - 1)
            else:
                for i in range(true_task):
                    for p in self.network.convnet.convnets[i].parameters():
                        p.requires_grad = False

            if self.with_super_split and (not self.use_oracle_super_label) and self.use_soft_mask and hasattr(
                self.network.convnet, "soft_mask_generator"):
                for i in range(true_task):
                    for p in self.network.convnet.soft_mask_generator[i].parameters():
                        p.requires_grad = False

        #for name, p in self.network.named_parameters():
        #    if p.requires_grad:
        #        print(name)
        #c = input()

        self._optimizer = factory.get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()),
                                                self._opt_name, lr=lr, weight_decay= weight_decay)

        base_scheduler = factory.get_lr_scheduler(self._scheduling, self._optimizer,
                                                  nb_epochs=self._n_epochs,
                                                  lr_decay=self._lr_decay,
                                                  task=true_task,
                                                  warmup_config=self._warmup_config)
        self._scheduler = base_scheduler


        if self.shortcut_epochs > 0:
            self._optimizer = factory.get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()),
                                                    self._opt_name, lr, weight_decay)

    def _train_task(self, train_loader, val_loader):
        logger.debug("nb {}.".format(len(train_loader.dataset)))
        #if self._is_task_level and self._task == 0:
        #logger.log("Task level model skip the first task")
        self._training_step(train_loader, val_loader, 0, self._n_epochs, temperature=self.temperature)

        self._post_processing_type = None

        true_task = self._task
        if self.merge2:
            assert self._task % 2 == 1
            true_task = self._task // 2

        if self._finetuning_config and true_task > (1 if self.merge else 0):
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
                                                                     high_range=self._n_classes)

            if self._finetuning_config["tuning"] == "all":
                parameters = self._network.parameters()
            elif self._finetuning_config["tuning"] == "convnet":
                parameters = self._network.convnet.parameters()
            elif self._finetuning_config["tuning"] == "classifier":
                if self.with_super_split and (not self.use_oracle_super_label) and hasattr(self.network.convnet,
                                                                                           "super_cat_classifier"):
                    parameters = [
                        {
                            "params": self._network.classifier.parameters(),
                            "lr": self._finetuning_config["lr"]
                        }, {
                            "params": self.network.convnet.super_cat_classifier.parameters(),
                            "lr": self._finetuning_config["lr"]
                        }
                    ]
                else:
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

            if self.dynamic_wd:
                # used in BiC official implementation
                weight_decay = self._weight_decay * self.task_max / (self._task + 1)
                #weight_decay = self._finetuning_config["weight_decay"] * (self._task + 1) / self.task_max
            else:
                weight_decay = self._finetuning_config["weight_decay"]
            self._optimizer = factory.get_optimizer(
                parameters, self._opt_name, self._finetuning_config["lr"], weight_decay
            )
            finetune_scheduling = self._finetuning_config["scheduling"]
            #if isinstance(self._finetuning_config["scheduling"], list):
            #    for e in self._finetuning_config["scheduling"]:
            #        finetune_scheduling.append(e + self._n_epochs)

            if self._is_consine_scheduler:
                nb_epochs = self._finetuning_config["epochs"]
            else:
                nb_epochs = self._n_epochs + self._finetuning_config["epochs"]
            self._scheduler = factory.get_lr_scheduler(finetune_scheduling, self._optimizer, nb_epochs=nb_epochs)

            self._network.classifier.classifier.reset_parameters()
            self._network.eval()
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

        #utils.display_weight_norm(logger, self._network, self._increments, "Initial trainset",
        #                          multi=(len(self._multiple_devices) > 1),
        #                          task_level=self._is_task_level)
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

            if self.shortcut_epochs != 0 and epoch == self.shortcut_epochs:
                logger.info("Switch from shortcut mode to normal mode")
                training_network.classifier.switch()

            if epoch == nb_epochs - 1 and record_bn and len(self._multiple_devices) == 1 and \
                    hasattr(training_network.convnet, "record_mode"):
                logger.info("Recording BN means & vars for MCBN...")
                training_network.convnet.clear_records()
                training_network.convnet.record_mode()

            if self._warmup_config and self._scheduler and epoch >= self.shortcut_epochs:
                if self._is_consine_scheduler and epoch >= self._n_epochs:
                    self._scheduler.step(epoch - self._n_epochs)
                else:
                    self._scheduler.step()

            if "total_epoch" in self._warmup_config and epoch == self._warmup_config["total_epoch"]:
                if len(self._multiple_devices) > 1:
                    training_network.module.classifier.classifier.reset_parameters()
                    training_network.module.classifier.aux_classifier.reset_parameters()
                else:
                    training_network.classifier.classifier.reset_parameters()
                    training_network.classifier.aux_classifier.reset_parameters()


            prog_bar = tqdm(
                train_loader,
                disable=self._disable_progressbar,
                ascii=True,
                bar_format="{desc}: {percentage:3.0f}% | {n_fmt}/{total_fmt} | {rate_fmt}{postfix}"
            )
            for i, input_dict in enumerate(prog_bar, start=1):
                if epoch < self._n_epochs:
                    self.train()
                inputs = input_dict["inputs"]
                targets = input_dict["targets"]
                targets_task = input_dict["targets_task"]
                targets_dataset = input_dict["targets_dataset"]

                #targets_class = input_dict["targets"] if self.with_class else None
                memory_flags = input_dict["memory_flags"]

                if grad is not None:
                    _clean_list(grad)
                    _clean_list(act)

                self._optimizer.zero_grad()
                old_classes = targets < (self._n_classes - self._task_size)
                new_classes = targets >= (self._n_classes - self._task_size)
                #print("old classes : {}".format(old_classes))
                #print("new classes : {}".format(new_classes))
                loss_ce, loss_aux, loss_class, loss_super, mask_norm = self._forward_loss(
                    training_network,
                    inputs,
                    targets_task if self._is_task_level else targets,
                    memory_flags,
                    targets_dataset=targets_dataset,
                    targets_class=(targets if self._is_task_level else targets_task),
                    old_classes=old_classes,
                    new_classes=new_classes,
                    temperature=temperature,
                    example=(i == len(prog_bar) - 1)
                )
                #print("ce loss : {}, aux loss : {}, class loss : {}".format(loss_ce, loss_aux, loss_class))

                if epoch >= self._n_epochs or self._task == 0:
                    #Fine tuning or the first task
                    loss = loss_ce
                else:
                    if self.aux_weight == 0 or self._is_task_level:
                        loss = loss_ce
                    else:
                        loss = loss_ce + self.aux_weight * loss_aux
                    if self.with_super_split and self.use_soft_mask:
                        loss += self.lambda_s * mask_norm
                if self.with_class:
                    loss += loss_class
                if self.with_super_split and (not self.use_oracle_super_label):
                    loss += loss_super

                if not utils.check_loss(loss):
                    import pdb
                    pdb.set_trace()

                #print("loss {}".format(loss))
                #print("lr {}".format(self._scheduler.get_lr()))
                #print(self._network.convnet.convnets[0].conv1.weight)
                #print(self._network.classifier.classifier.weight)
                #c = input()

                loss.backward()
                self._optimizer.step()

                current_lr = self._optimizer.param_groups[0]['lr']
                current_wd = self._optimizer.param_groups[0]['weight_decay']
                self._metrics["lr"] = current_lr * i
                self._metrics["weight_decay"] = current_wd * i

                self._print_metrics(prog_bar, epoch, nb_epochs, i)

            logger.info("Task {} / {}, epoch {} / {}, loss {}".format(self._task + 1, self._n_tasks,
                                                                      epoch + 1, nb_epochs,
                                                                      round(self._metrics["loss"] / i, 3)))


            if (not self._warmup_config) and self._scheduler and epoch >= self.shortcut_epochs:
                if self._is_consine_scheduler and epoch >= self._n_epochs:
                    self._scheduler.step(epoch - self._n_epochs)
                else:
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
            if self._task == self._n_tasks - 1:
                pass
                #weight = training_network.classifier.classifier.weight.detach().cpu().numpy()
                #np.save("weight/test4/epoch_{}".format(epoch), weight)

        #utils.display_weight_norm(logger, self._network, self._increments, "After training",
        #                          multi=(len(self._multiple_devices) > 1),
        #                          task_level=self._is_task_level)
        #utils.display_feature_norm(logger, self._network, train_loader, self._n_classes,
        #                           self._increments, "Trainset")

    def _forward_loss(self,
        training_network,
        inputs,
        targets,
        memory_flags,
        targets_dataset=None,
        targets_class=None,
        old_classes=None, new_classes=None, accu=None, new_accu=None, old_accu=None, temperature=1.0, example=False):
        inputs = inputs.to(self._device, non_blocking=True)
        targets = targets.to(self._device, non_blocking=True)
        targets_dataset = targets_dataset.to(self._device, non_blocking=True)
        if self.with_class:
            targets_class = targets_class.to(self._device, non_blocking=True)

        if self.with_super_split and self.use_oracle_super_label:
            outputs = training_network([inputs, targets_dataset])
        else:
            outputs = training_network(inputs)
        if accu is not None:
            accu.add(outputs['logit'], targets)
            # accu.add(logits.detach(), targets.cpu().numpy())
        # if new_accu is not None:
        #     new_accu.add(logits[new_classes].detach(), targets[new_classes].cpu().numpy())
        # if old_accu is not None:
        #     old_accu.add(logits[old_classes].detach(), targets[old_classes].cpu().numpy())
        loss, aux_loss, loss_class, loss_super, mask_norm = self._compute_loss(inputs, targets, outputs,
                                                                    old_classes, new_classes,
                                                                    targets_class=targets_class,
                                                                    targets_dataset=targets_dataset,
                                                                    temperature=temperature)
        if not utils.check_loss(loss):
            raise ValueError("A loss is NaN: {}".format(self._metrics))

        if not utils.check_loss(aux_loss):
            raise ValueError("A aux_loss is NaN: {}".format(self._metrics))

        self._metrics["loss"] += loss.item()
        self._metrics["aux_loss"] += -1 if self.aux_weight == 0 else aux_loss.item()
        self._metrics["loss_class"] += loss_class.item() if self.with_class else -1
        self._metrics["loss_super"] += loss_super.item() if (loss_super is not None) else -1
        self._metrics["mask_norm"] += mask_norm.item() if (mask_norm is not None) else -1

        pred = outputs["logit"].argmax(dim=-1)
        acc = (pred == targets).sum()
        acc = acc / targets.shape[0] * 100
        self._metrics["acc"] += acc.item()
        #print(outputs['logit'])
        #print(pred)
        #print(targets)

        if self.with_class:
            pred = outputs['logit_class'].argmax(dim=-1)
            acc = (pred == targets_class).sum()
            acc = acc / targets_class.shape[0] * 100
            self._metrics["acc_class"] += acc.item()

        if loss_super is not None:
            pred = outputs["super_cat_logits"].argmax(dim=-1)
            acc = (pred == targets_dataset).sum()
            acc = acc / targets_dataset.shape[0] * 100
            self._metrics["acc_super"] += acc.item()

        #print()
        #print(outputs['logit'][0])
        #print("{} pred".format(pred.tolist()[:60]))
        #print("{} targets".format(targets.tolist()[:60]))
        #c = input()

        if example and False:
            print()
            print("{} pred".format(pred.tolist()[:60]))
            print("{} targets".format(targets.tolist()[:60]))

        return loss, aux_loss, loss_class, loss_super, mask_norm

    def _compute_loss(self, inputs, targets, outputs, old_classes, new_classes, targets_class=None, targets_dataset=None, temperature=1.0):
        loss = F.cross_entropy(outputs['logit'] / temperature, targets)
        #loss = utils.CosFaceLoss(outputs['logit'], targets)


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

        loss_super = None
        if self.with_super_split and (not self.use_oracle_super_label):
            loss_super = F.cross_entropy(outputs["super_cat_logits"], targets_dataset)

        mask_norm = None
        if "soft_mask" in outputs:
            if outputs["soft_mask"] is not None:
                soft_mask = outputs["soft_mask"]
                mask_norm = torch.mean(torch.abs(soft_mask))

        return loss, aux_loss, loss_class, loss_super, mask_norm

    def _eval_task(self, data_loader):
        ypreds, ytrue = self._compute_accuracy_by_netout(data_loader)

        return ypreds, ytrue

    def _compute_accuracy_by_netout(self, data_loader, save_features=False):
        preds, targets = [], []
        if save_features:
            features = []
        self._network.eval()
        with torch.no_grad():
            for input_dict in data_loader:
                inputs = input_dict["inputs"]
                lbls = input_dict["targets_task"] if self._is_task_level else input_dict["targets"]
                inputs = inputs.to(self._device, non_blocking=True)
                if save_features:
                    feature = self._network.extract(inputs)
                    features.append(feature.detach().cpu().numpy())
                    targets.append(lbls.long().cpu().numpy())
                    continue
                if self.with_super_split and self.use_oracle_super_label:
                    targets_dataset = input_dict["targets_dataset"].to(self._device, non_blocking=True)
                    _preds = self._network([inputs, targets_dataset])['logit']
                else:
                    _preds = self._network(inputs)['logit']
                preds.append(_preds.detach().cpu().numpy())
                targets.append(lbls.long().cpu().numpy())
        if save_features:
            features = np.concatenate(features, axis=0)
            np.save("task_features_margin5.npy", features)
        else:
            preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        if save_features:
            np.save("task_targets.npy", targets)
            exit()
        return preds, targets

    def build_examplars_task(self, inc_dataset, herding_indexes, memory_per_class=None, data_source="train"):
        logger.info("Building & updating memory in task level.")
        memory_per_class = memory_per_class or self._memory_per_class
        memory_per_task = memory_per_class * self._task_size
        herding_indexes = copy.deepcopy(herding_indexes)
        class_means = np.zeros((self._n_classes, self._network.features_dim))
        data_memory, targets_memory = [], []

        for task_idx in range(self._task + 1):
            class_idx = list(range(task_idx * self._task_size, (task_idx + 1) * self._task_size))
            inputs, loader = inc_dataset.get_custom_loader(
                class_idx, mode="test", data_source=data_source
            )
            if task_idx == self._task:
                features, targets = utils.extract_features(self._network, loader,
                                                           use_tgt_dataset=self.with_super_split and self.use_oracle_super_label)
                if self._herding_selection["type"] == "icarl":
                    selected_indexes = herding.icarl_selection(features, memory_per_task)
                elif self._herding_selection["type"] == "closest":
                    selected_indexes = herding.closest_to_mean(features, memory_per_task)
                elif self._herding_selection["type"] == "random":
                    selected_indexes = herding.random(features, memory_per_task)
                elif self._herding_selection["type"] == "first":
                    selected_indexes = np.arange(memory_per_task)
                elif self._herding_selection["type"] == "kmeans":
                    selected_indexes = herding.kmeans(
                        features, memory_per_task, k=self._herding_selection["k"]
                    )
                elif self._herding_selection["type"] == "confusion":
                    selected_indexes = herding.confusion(
                        *self._last_results,
                        memory_per_task,
                        class_id=class_idx,
                        minimize_confusion=self._herding_selection["minimize_confusion"]
                    )
                elif self._herding_selection["type"] == "var_ratio":
                    selected_indexes = herding.var_ratio(
                        memory_per_task, self._network, loader, **self._herding_selection
                    )
                elif self._herding_selection["type"] == "mcbn":
                    selected_indexes = herding.mcbn(
                        memory_per_task, self._network, loader, **self._herding_selection
                    )
                else:
                    raise ValueError(
                        "Unknown herding selection {}.".format(self._herding_selection)
                    )

                herding_indexes.append(selected_indexes)
            else:
                targets = []
                for input_dict in loader:
                    _targets = input_dict["targets"]
                    _targets = _targets.numpy()
                    targets.append(_targets)
                targets = np.concatenate(targets)

            # Reducing examplars:
            try:
                selected_indexes = herding_indexes[task_idx][:memory_per_task]
                herding_indexes[task_idx] = selected_indexes
            except:
                import pdb
                pdb.set_trace()

            data_memory.append(inputs[selected_indexes])
            targets_memory.append(targets[selected_indexes])

        data_memory = np.concatenate(data_memory)
        targets_memory = np.concatenate(targets_memory)


        return data_memory, targets_memory, herding_indexes, class_means

    def build_examplars(
        self, inc_dataset, herding_indexes, memory_per_class=None, data_source="train_example"
    ):
        if self._task_level_herding:
            return self.build_examplars_task(inc_dataset, herding_indexes, memory_per_class, data_source)
        logger.info("Building & updating memory.")
        memory_per_class = memory_per_class or self._memory_per_class
        herding_indexes = copy.deepcopy(herding_indexes)

        data_memory, targets_memory = [], []
        class_means = np.zeros((self._n_classes, self._network.features_dim))

        for class_idx in range(self._n_classes):
            # We extract the features, both normal and flipped:
            inputs, loader = inc_dataset.get_custom_loader(
                class_idx, mode="test", data_source=data_source, batch_size=1024
            )
            targets = loader.dataset.y

            if class_idx >= self._n_classes - self._task_size:
                # New class, selecting the examplars:
                features, targets = utils.extract_features(self._network, loader, use_tgt_dataset=True)
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
                selected_indexes = sorted(selected_indexes)
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
