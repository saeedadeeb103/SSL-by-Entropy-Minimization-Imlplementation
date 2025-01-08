import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from utils.helper_functions import entropy_loss
from torchmetrics import Accuracy
from itertools import cycle

class CombinedDataLoader:
    def __init__(self, labeled_loader, unlabeled_loader):
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader

    def __iter__(self):
        return zip(cycle(self.labeled_loader), self.unlabeled_loader)

    def __len__(self):
        return len(self.unlabeled_loader)

class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class SemiSupervisedModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim, optimizer="Adam"):
        super(SemiSupervisedModel, self).__init__()
        self.model = LogisticRegression(input_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=output_dim)
        self.lambda_entropy = 0.1  
        self.weight_decay = 0.0001 
        # configure optimizer
        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=0.001, weight_decay=self.weight_decay)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=0.001, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
    # Unpack the batch
        (x_labeled, y_labeled, z_labeled), (x_unlabeled, z_unlabeled) = batch

        # Forward pass for labeled data
        outputs_labeled = self(x_labeled)
        loss_labeled = self.criterion(outputs_labeled, y_labeled)

        # Compute accuracy on labeled data
        preds_labeled = torch.argmax(outputs_labeled, dim=1)
        accuracy_labeled = self.accuracy(preds_labeled, y_labeled)

        # forward pass for unlabeled data
        outputs_unlabeled = self(x_unlabeled)
        probs_unlabeled = torch.softmax(outputs_unlabeled, dim=1)

        # compute the likelihood term for unlabeled data
        log_likelihood_unlabeled = torch.log(torch.sum(z_unlabeled * probs_unlabeled, dim=1) + 1e-10)
        loss_unlabeled = -torch.mean(log_likelihood_unlabeled)

        # Total loss
        loss = loss_labeled + self.lambda_entropy * loss_unlabeled

        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, logger=True)
        self.log('train_acc', accuracy_labeled, prog_bar=True, on_epoch=True, on_step=False, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if len(batch) == 2:  # if the batch contains only features and labels
            x, y = batch
        elif len(batch) > 2:  # if the batch contains additional tensors
            x, y = batch[:2]  # use only the first two tensors (features and labels)
        else:
            raise ValueError("Unexpected batch structure.")
        
        y = y.long()

        logits = self(x)
        loss = self.criterion(logits, y)

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        accuracy = self.accuracy(preds, y)

        # Log validation metrics
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log('val_acc', accuracy, prog_bar=True, on_epoch=True, on_step=True)

        return loss
    
    # def on_validation_epoch_end(self):
    #     avg_loss = self.trainer.logged_metrics['val_loss_epoch']
    #     accuracy = self.trainer.logged_metrics['val_acc_epoch']

    #     self.log('val_loss', avg_loss, prog_bar=True, on_epoch=True)
    #     self.log('val_acc', accuracy, prog_bar=True, on_epoch=True)

    #     return {'Average Loss:': avg_loss, 'Accuracy:': accuracy}


    def configure_optimizers(self):
        optimizer = self.optimizer
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}



from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import numpy as np
def train_generative_model(x_labeled, y_labeled, x_unlabeled, x_val, y_val, a, max_iter, case=1):
    x_train = np.vstack((x_labeled, x_unlabeled))
    y_train = np.hstack((y_labeled, -np.ones(len(x_unlabeled))))  # Unlabeled data has label -1


    if case == 1:
        means_init = [
            np.full(x_labeled.shape[1], a),  # Mean for class 1
            np.full(x_labeled.shape[1], -a)  # Mean for class 2
        ]
    elif case ==2:
        means_init=[np.full(x_labeled.shape[1], a) + np.random.normal(0, 0.1, x_labeled.shape[1]),
                        np.full(x_labeled.shape[1], -a) + np.random.normal(0, 0.1, x_labeled.shape[1])]
    
    gmm = GaussianMixture(
            n_components=2,
            means_init=means_init,
            random_state=42,
            max_iter=max_iter
        )
    gmm.fit(x_train)
    y_pred = gmm.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

def train_supervised_logistic_regression(x_labeled, y_labeled, x_val, y_val, max_iter):
    model = SklearnLogisticRegression(random_state=42, max_iter=max_iter)
    model.fit(x_labeled, y_labeled)
    y_pred = model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy

def train_all_labels_known(x_labeled, y_labeled, x_unlabeled, y_unlabeled_true, x_val, y_val, max_iter):
    """
    Train logistic regression with all labels known (best-case scenario).
    """
    # Combine labeled and unlabeled data with true labels
    x_all = np.vstack((x_labeled, x_unlabeled))
    y_all = np.hstack((y_labeled, y_unlabeled_true))

    model = SklearnLogisticRegression(random_state=42, max_iter=max_iter)
    model.fit(x_all, y_all)
    y_pred = model.predict(x_val)
    accuracy = accuracy_score(y_val, y_pred)
    return accuracy