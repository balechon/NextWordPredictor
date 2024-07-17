import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import torch
from dataset import TextDataset

class TextDataModule(pl.LightningDataModule):
    def __init__(self, data_path, batch_size=32, sequence_length=128, limit=None,
                 vocab_size=10000, val_split=0.1, test_split=0.1, num_workers=4):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.limit = limit
        self.vocab_size = vocab_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Crear el dataset completo
        full_dataset = TextDataset(
            self.data_path,
            sequence_length=self.sequence_length,
            limit=self.limit,
            vocab_size=self.vocab_size
        )

        # Calcular tamaños de los splits
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * self.val_split)
        test_size = int(dataset_size * self.test_split)
        train_size = dataset_size - val_size - test_size

        # Dividir el dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # Para reproducibilidad
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def get_vocab_size(self):
        return len(self.train_dataset.dataset.vocab)