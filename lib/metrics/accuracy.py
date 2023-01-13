import torchmetrics as tm
import torch
import torch.nn as nn
from lib.utilities.constants import TOKEN_PAD, TORCH_FLOAT
from lib.utilities.device import cpu_device


class MusicAccuracy(tm.Metric):
    def __init__(self) -> None:
        super().__init__()

        self.add_state('accuracy', default=[], dist_reduce_fx="cat")

    def __compute_accuracy(self, out, tgt):
        """
        Computes the average accuracy for the given input and output batches. Accuracy uses softmax
        of the output.
        """

        softmax = nn.Softmax(dim=-1)
        out = torch.argmax(softmax(out), dim=-1)

        out = out.flatten()
        tgt = tgt.flatten()

        mask = (tgt != TOKEN_PAD)

        out = out[mask]
        tgt = tgt[mask]

        # Empty
        if(len(tgt) == 0):
            return 1.0

        num_right = (out == tgt)
        num_right = torch.sum(num_right).type(TORCH_FLOAT)

        acc = num_right / len(tgt)

        return acc

    def update(self, gt, pred):

        gt = torch.stack(gt.to(cpu_device()))
        predictions = torch.stack(pred.to(cpu_device()))

        accuracy = self.__compute_accuracy(gt, predictions)
        self.accuracy.append(accuracy)

    def compute(self):
        accuracy = torch.mean(self.accuracy)
        self.accuracy = []
        return accuracy