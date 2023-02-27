import torch.nn as nn
import pytorch_lightning as pl
import torch
import torchmetrics as tm
from lib.utilities.constants import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from lib.utilities.constants import KEY_DICT, GENRE_DICT, ARTIST_DICT
import matplotlib.pyplot as plt

class MusicTransformerClassifier(pl.LightningModule):

    def __init__(self, music_transformer, n_classes, loss_fn, lr) -> None:
        super(MusicTransformerClassifier, self).__init__()
        
        n_hidden_backbone = music_transformer.d_model
        self.backbone = music_transformer
        # Backbone is also a pl module
        self.backbone.freeze()

        self.n_classes = n_classes
        if n_classes == len(KEY_DICT):
            self.labels = list(KEY_DICT.keys())
        elif n_classes == len(GENRE_DICT):
            self.labels = list(GENRE_DICT.keys())
        elif n_classes == len(ARTIST_DICT):
            self.labels = list(ARTIST_DICT.keys())   
        else:
            raise Exception('Unrecognized number of classes!')   

        self.classifier = nn.Linear(n_hidden_backbone, n_classes)

        self.softmax    = nn.Softmax(dim=-1)

        self.loss_fn = loss_fn
        self.lr = lr

        self.train_acc = tm.Accuracy('multiclass', num_classes=self.n_classes)
        self.val_acc = tm.Accuracy('multiclass', num_classes=self.n_classes)
        self.test_acc = tm.Accuracy('multiclass', num_classes=self.n_classes)
        self.train_f1 = tm.F1Score('multiclass', num_classes=self.n_classes)
        self.val_f1 = tm.F1Score('multiclass', num_classes=self.n_classes)
        self.test_f1 = tm.F1Score('multiclass', num_classes=self.n_classes)


    def forward(self, x):
        t_out = self.backbone(x)
        t_out_pooled = t_out.mean(dim=1)[0]
        c_out = self.classifier(t_out_pooled)
        y_pred = self.softmax(c_out)

        return y_pred

    def step(self, batch, acc_metric, f1_metric):
        x, y = batch

        y_pred = self.forward(x)

        loss = self.loss_fn.forward(y_pred, y)

        acc_metric.update(y, y_pred)
        f1_metric.update(y, y_pred)

        return loss, y_pred, y


    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, self.train_acc, self.train_f1)

        self.log('training loss', loss)

        return loss

    def training_epoch_end(self, outs):
        self.log('train accuracy', self.train_acc)
        self.log('train perplexity', self.train_f1)

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, self.val_acc, self.val_f1)

        self.log('validation loss', loss)

        return loss

    def validation_epoch_end(self, outs):
        self.log('test accuracy', self.val_acc)
        self.log('test perplexity', self.val_f1)

    def test_step(self, batch, batch_idx):
        loss, y_pred, y_tru = self.step(batch, self.test_acc, self.test_f1)
        
        return loss, y_pred, y_tru
    
    def test_epoch_end(self, outs):
        _, y_pred, y_tru = zip(*outs)
        self.log('test accuracy', self.test_acc)
        self.log('test perplexity', self.test_f1)

        cm = confusion_matrix(y_true=y_tru, y_pred=y_pred, labels=self.labels)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels).plot()
        plt.savefig("confusion_matrix.png", bbox_inches="tight", dpi=300)
        self.logger.log_image(key="confusion_matrix", images=["confusion_matrix.png"])


    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(ADAM_BETA_1, ADAM_BETA_2), eps=ADAM_EPSILON)
        return opt
    