import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def eval_step(self):
    output = self.forward(self.data[0])
    if hasattr(self, "classifier"):
        output = self.classifier(output)
    acc1, acc5 = accuracy(output, self.data[1], topk=(1, 5))
    self.top1.update(acc1.item(), self.data[0].size(0))
    self.top5.update(acc5.item(), self.data[0].size(0))

    if self.config.log.project is not None:
        wandb.log(
            {
                "epoch": self.epoch,
                "step": self.step,
                "test/acc1": self.top1.avg,
                "test/acc5": self.top5.avg,
            }
        )


def accuracy_by_class(self, output, target, num_classes):
    """Computes the accuracy by class"""
    with torch.no_grad():
        class_correct = torch.zeros(num_classes)
        class_total = torch.zeros(num_classes)

        _, predicted = output.max(1)  # Get the index of the max log-probability

        for i in range(num_classes):
            class_indices = target == i
            class_total[i] = class_indices.sum().item()
            correct_predictions = predicted[class_indices] == target[class_indices]
            class_correct[i] = correct_predictions.sum().item()

        class_accuracy = class_correct / class_total * 100.0
        return class_accuracy
