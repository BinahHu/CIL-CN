import logging

import numpy as np
import torch
from torch.nn import functional as F

from inclearn.lib import factory, loops, network, utils
from inclearn.models import IncrementalLearner

logger = logging.getLogger(__name__)


class Joint(IncrementalLearner):

    def __init__(self, args):
        self._device = args["device"][0]
        self._multiple_devices = args["device"]

        self._opt_name = args["optimizer"]
        self._lr = args["lr"]
        self._lr_decay = args["lr_decay"]
        self._weight_decay = args["weight_decay"]
        self._n_epochs = args["epochs"]
        self._scheduling = args["scheduling"]
        self._eval_every_x_epochs = args.get("eval_every_x_epochs")


        self._args = args

        self._n_classes = 0

        self._network = network.BasicNet(
            self._args["convnet"],
            convnet_kwargs=self._args.get("convnet_config", {}),
            classifier_kwargs=self._args.get(
                "classifier_config", {
                    "type": "fc",
                    "use_bias": True
                }
            ),
            device=self._device
        )

    def _before_task(self, data_loader, val_loader):
        self._n_classes += self._task_size

        if self._is_task_level:
            self._network.add_classes(1)
        else:
            self._network.add_classes(self._task_size)

        self._optimizer = factory.get_optimizer(
            self._network.parameters(), self._opt_name, self._lr,
            self._weight_decay
        )
        if self._scheduling is None:
            self._scheduler = None
        else:
            self._scheduler = factory.get_lr_scheduler(self._scheduling, self._optimizer, nb_epochs=self._n_epochs)

    def _train_task(self, train_loader, val_loader):
        if self._task < self._n_tasks - 1:
            return
        # Joint model train all the data together:
        _, train_loader = self.inc_dataset.get_custom_loader(
            list(range(self._n_classes)), mode="train", data_source="train_full")
        _, val_loader = self.inc_dataset.get_custom_loader(
            list(range(self._n_classes)), data_source="val_full" if self._args.get('validation', 0) > 0 else "test_full")

        loops.single_loop(
            train_loader,
            val_loader,
            self._multiple_devices,
            self._network,
            self._n_epochs,
            self._optimizer,
            scheduler=self._scheduler,
            train_function=self._forward_loss,
            eval_function=self._accuracy,
            task=self._task,
            n_tasks=self._n_tasks,
            is_task=self._is_task_level,
            eval_every_x_epochs=self._eval_every_x_epochs
        )

    def _after_task(self, inc_dataset):
        self._network.on_task_end()

    def _eval_task(self, loader):
        ypred, ytrue = [], []

        for input_dict in loader:
            with torch.no_grad():
                logits = self._network(input_dict["inputs"].to(self._device))["logits"]

            ytrue.append(input_dict["targets_task"].numpy() if self._is_task_level else input_dict["targets"].numpy())
            ypred.append(torch.softmax(logits, dim=1).cpu().numpy())

        ytrue = np.concatenate(ytrue)
        ypred = np.concatenate(ypred)

        return ypred, ytrue

    def _accuracy(self, network, loader):
        ypred, ytrue = self._eval_task(loader)
        ypred = ypred.argmax(axis=1)
        pos = ypred != ytrue
        logger.info("{} pred".format(ypred[pos][:60]))
        logger.info("{} targets".format(ytrue[pos][:60]))

        return 100 * round(np.mean(ypred == ytrue), 3)

    def _forward_loss(self, training_network, inputs, targets, memory_flags, metrics, example, **kwargs):
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        onehot_targets = utils.to_onehot(targets, self._n_classes).to(self._device)

        outputs = training_network(inputs)

        loss = self._compute_loss(inputs, outputs, targets, onehot_targets, memory_flags, metrics)

        if not utils.check_loss(loss):
            raise ValueError("Loss became invalid ({}).".format(loss))

        metrics["loss"] += loss.item()
        pred = outputs["logits"].argmax(dim=-1)
        acc = (pred == targets).sum()
        acc = acc / targets.shape[0] * 100
        metrics["acc"] += acc.item()

        if example and False:
            pred = outputs["logits"].argmax(dim=-1)
            print()
            print("{} pred".format(pred.tolist()[:60]))
            print("{} targets".format(targets.tolist()[:60]))

        return loss

    def _compute_loss(self, inputs, outputs, targets, onehot_targets, memory_flags, metrics):
        logits = outputs["logits"]

        return F.cross_entropy(logits, targets)
