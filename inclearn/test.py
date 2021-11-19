import os
import yaml
import json
import copy
import torch
import random
import numpy as np

from inclearn import parser
from inclearn.lib import factory

def _parse_options(path):
    with open(path) as f:
        if path.endswith(".yaml") or path.endswith(".yml"):
            return yaml.load(f, Loader=yaml.FullLoader)
        elif path.endswith(".json"):
            return json.load(f)["config"]
        else:
            raise Exception("Unknown file type {}.".format(path))

def _set_up_options(args):
    options_paths = args["options"] or []

    autolabel = []
    for option_path in options_paths:
        if not os.path.exists(option_path):
            raise IOError("Not found options file {}.".format(option_path))

        args.update(_parse_options(option_path))

        autolabel.append(os.path.splitext(os.path.basename(option_path))[0])

    return "_".join(autolabel)

def _set_global_parameters(config):
    _set_seed(config["seed"], config["threads"], config["no_benchmark"], config["detect_anomaly"])
    factory.set_device(config)


def _set_seed(seed, nb_threads, no_benchmark, detect_anomaly):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = not no_benchmark  # This will slow down training.
    torch.set_num_threads(nb_threads)
    if detect_anomaly:
        torch.autograd.set_detect_anomaly(detect_anomaly)

args = parser.get_parser().parse_args()
args = vars(args)

autolabel = _set_up_options(args)
if args["autolabel"]:
    args["label"] = autolabel

seed_list = copy.deepcopy(args["seed"])

orders = copy.deepcopy(args["order"])
del args["order"]
if orders is not None:
    assert isinstance(orders, list) and len(orders)
    assert all(isinstance(o, list) for o in orders)
    assert all([isinstance(c, int) for o in orders for c in o])
else:
    orders = [None for _ in range(len(seed_list))]

seed = seed_list[0]
args["seed"] = seed
class_order = orders[0]
config = args

_set_global_parameters(args)

inc_dataset = factory.get_data(config, class_order)
covnorm_logits_path = "/home/zhiyuan/CIL-CN/features/der_org_cluster_buffer2000.npy"
model_logits_path = "/home/zhiyuan/CIL-CN/features/der_org_cluster_buffer2000.npy"
split = inc_dataset.increments
map = {}
for i in range(len(split)):
    for j in range(split[i]):
        map[j+sum(split[:i])] = i

covnorm_logits = np.load(covnorm_logits_path)
model_logits = np.load(model_logits_path)

task = False
task_num = 20
test_loader = None

if task:
    task_pred = model_logits.argmax(axis=1)
else:
    task_pred = np.zeros((model_logits.shape[0],), dtype=np.int32)
    class_pred = model_logits.argmax(axis=1)
    for i in range(model_logits.shape[0]):
        task_pred[i] = map[class_pred[i]]

for i in range(task_num):
    task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(None, None)

CIL_acc = 0
TIL_acc = 0
y = []
y_task = []

for input_dict in test_loader:
    y.append(input_dict["targets"].detach().cpu().numpy())
    y_task.append(input_dict["targets_task"].detach().cpu().numpy())
y = np.concatenate(y, axis=0)
y_task = np.concatenate(y_task, axis=0)
for i in range(model_logits.shape[0]):
    task = task_pred[i]
    l = sum(split[:task])
    r = sum(split[:task+1])
    CIL_pred = covnorm_logits[i][l:r].argmax() + l
    CIL_acc += (CIL_pred == y[i])

    task = y_task[i]
    l = sum(split[:task])
    r = sum(split[:task + 1])
    TIL_pred = covnorm_logits[i][l:r].argmax() + l
    TIL_acc += (TIL_pred == y[i])

CIL_acc = CIL_acc / y.shape[0] * 100
TIL_acc = TIL_acc / y.shape[0] * 100
TC_acc = (task_pred == y_task).sum() / y_task.shape[0] * 100
print("TIL = {}, CIL = {}, TC = {}, TIL_condition = {}".format(round(TIL_acc, 2), round(CIL_acc, 2), round(TC_acc, 2), round(CIL_acc / TC_acc * 100, 2)))



