
from src.LSTM import LSTM
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
import torch


class LSTMNextWordPredictor(pl.LightningModule):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.model = LSTM(vocab_size, embedding_dim, hidden_size, num_layers)

        # Guardar hiperparámetros en directorio de logs
        self.save_hyperparameters(ignore=["model"])

        # Definición de métricas
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=vocab_size)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=vocab_size)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=vocab_size)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        sequences, targets = batch
        logits = self(sequences)

        # Reshape logits to (batch_size * sequence_length, vocab_size)
        logits = logits.view(-1, self.vocab_size)

        # Reshape targets to (batch_size * sequence_length)
        targets = targets.view(-1)

        # Now both should have compatible shapes
        loss = F.cross_entropy(logits, targets)
        predicted_words = torch.argmax(logits, dim=-1)

        return loss, targets, predicted_words

    def training_step(self, batch, batch_idx):
        loss, targets, predicted_words = self._shared_step(batch)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.train_acc(predicted_words.view(-1), targets.view(-1))
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, targets, predicted_words = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.valid_acc(predicted_words.view(-1), targets.view(-1))
        self.log("valid_acc", self.valid_acc, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, targets, predicted_words = self._shared_step(batch)
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        self.test_acc(predicted_words.view(-1), targets.view(-1))
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def set_vocab(self, token_to_idx, idx_to_token):
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token