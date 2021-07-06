from datetime import datetime
import sys
import os
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, args):
        self.root = args['logger']['root']
        self.dataset = args['dataset']['type']
        self.name = args['logger']['name']
        self.path = os.path.join(os.path.join(self.root, self.dataset), self.name)
        if not(self.name == 'test'):
            assert (not os.path.exists(self.path)), "Model name already used!"
            os.mkdir(self.path)
        self.logger = SummaryWriter(log_dir=self.path)

    def log(self, t, i, loss, loss_task, acc, acc_task):
        self.logger.add_scalar('task_{}/loss'.format(t), loss, i)
        self.logger.add_scalar('task_{}/loss_task'.format(t), loss_task, i)
        self.logger.add_scalar('task_{}/acc'.format(t), acc, i)
        self.logger.add_scalar('task_{}/acc_task'.format(t), acc_task, i)

    def close(self):
        self.logger.close()


def progress_bar(i, max_iter, epoch, task_number, loss, loss_task, acc, acc_task):
    """
    Prints out the progress bar on the stderr file.
    :param i: the current iteration
    :param max_iter: the maximum number of iteration
    :param epoch: the epoch
    :param task_number: the task index
    :param loss: the current value of the loss function
    """
    if not (i + 1) % 10 or (i + 1) == max_iter:
        progress = min(float((i + 1) / max_iter), 1)
        progress_bar = ('█' * int(50 * progress)) + ('┈' * (50 - int(50 * progress)))
        print('\r[ {} ] Task {} | epoch {}: |{}| loss: {} | loss_task: {} | acc: {} | acc_task: {}'.format(
            datetime.now().strftime("%m-%d | %H:%M"),
            task_number + 1 if isinstance(task_number, int) else task_number,
            epoch,
            progress_bar,
            round(loss, 8),
            round(loss_task, 8),
            round(acc, 2),
            round(acc_task, 2)
        ), file=sys.stderr, end='', flush=True)