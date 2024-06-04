import os
import sys
import json
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Hyperparameters
batch_size = 16
seq_len = 5  # Veri uzunluğunuza göre bu değeri ayarlayın
embedding_dim = 128
hidden_dim = 256
num_layers = 2
lr = 1e-3
epochs = 1000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
class LanguageModelDataset(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.seq_len], dtype=torch.long),
            torch.tensor(self.data[idx+self.seq_len], dtype=torch.long)
        )

# Model
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# Train
def train(model, data_loader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            # Hata ayıklama çıktıları ekleyin
            print(f"x shape: {x.shape}, y shape: {y.shape}, y_pred shape: {y_pred.shape}")
            loss = criterion(y_pred.view(-1, y_pred.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch: {epoch+1}/{epochs}, Loss: {total_loss/len(data_loader)}')

# Generate
def generate(model, data, seq_len, idx_to_word, n_words=100):
    model.eval()
    idx = random.randint(0, len(data) - seq_len)
    x = torch.tensor([data[idx:idx+seq_len]], dtype=torch.long).to(device)
    result = []
    with torch.no_grad():
        for _ in range(n_words):
            y_pred = model(x)
            y_pred = y_pred.view(-1)
            y_pred = F.softmax(y_pred, dim=0)
            idx = torch.multinomial(y_pred, num_samples=1).item()
            result.append(idx_to_word[idx])
            x = torch.cat([x[:, 1:], torch.tensor([[idx]], dtype=torch.long).to(device)], dim=1)
    return result

# Main
if __name__ == '__main__':
    # Load data
    with open('data.json', 'r') as f:
        data = json.load(f)

    # Ensure data is a list of words/tokens
    assert isinstance(data, list), "Data should be a list of words/tokens"

    if len(data) <= seq_len:
        raise ValueError("Data length should be greater than seq_len")

    vocab = list(set(data))
    vocab_size = len(vocab)
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    data = [word_to_idx[word] for word in data]

    # Create model
    model = LanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Create data loader
    dataset = LanguageModelDataset(data, seq_len)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train
    train(model, data_loader, optimizer, criterion, epochs)

    # Generate
    result = generate(model, data, seq_len, idx_to_word)
    print(' '.join(result))

    # Save model
    torch.save(model.state_dict(), 'model.pth')
