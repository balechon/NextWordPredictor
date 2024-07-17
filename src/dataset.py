import os
from pathlib import Path
import sys
import re

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torchmetrics

import pandas as pd

from nltk.tokenize import word_tokenize
from collections import Counter
import nltk

def get_the_main_path() -> Path:
    return Path(__file__).resolve().parents[1]


class TextDataset(Dataset):
    def __init__(self, data_path, sequence_length=50, limit=None, vocab_size=10000):
        if limit:
            self.data = pd.read_csv(data_path).head(limit)
        else:
            self.data = pd.read_csv(data_path)

        self.sequence_length = sequence_length

        # Preprocesar texto
        all_text = ' '.join(self.data['transcript']).lower()

        # Tokenizar por palabras
        words = nltk.word_tokenize(all_text)

        # Construir vocabulario
        word_counts = Counter(words)
        self.vocab = ['<PAD>', '<UNK>'] + [word for word, _ in word_counts.most_common(vocab_size - 2)]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        # Función de tokenización
        self.tokenize = lambda x: [self.token_to_idx.get(word, self.token_to_idx['<UNK>']) for word in
                                   word_tokenize(x.lower())]

        # Codificar discursos
        self.encoded_speeches = [self.tokenize(speech) for speech in self.data['transcript']]

        # Crear secuencias
        self.sequences = []
        for encoded_speech in self.encoded_speeches:
            for i in range(0, len(encoded_speech) - sequence_length):
                self.sequences.append(encoded_speech[i:i + sequence_length + 1])  # +1 para incluir la palabra objetivo

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence[:-1]), torch.tensor(sequence[1:])