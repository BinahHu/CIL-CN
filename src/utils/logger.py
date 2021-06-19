from datetime import datetime
import sys




def progress_bar(i: int, max_iter: int, epoch, task_number: int, loss: float, loss_task):
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
        print('\r[ {} ] Task {} | epoch {}: |{}| loss: {} | loss_task: {}'.format(
            datetime.now().strftime("%m-%d | %H:%M"),
            task_number + 1 if isinstance(task_number, int) else task_number,
            epoch,
            progress_bar,
            round(loss, 8),
            round(loss_task, 8)
        ), file=sys.stderr, end='', flush=True)