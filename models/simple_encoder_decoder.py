"""
Simple Encoder-Decoder Architecture without Attention
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm


class SimpleEncoder(nn.Module):
    """Simple Encoder using LSTM"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.5):
        super(SimpleEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            n_layers, 
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src):
        """
        src: (batch_size, seq_len)
        """
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell


class SimpleDecoder(nn.Module):
    """Simple Decoder using LSTM"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.5):
        super(SimpleDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            n_layers, 
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg, hidden, cell):
        """
        trg: (batch_size, seq_len)
        hidden: (n_layers, batch_size, hidden_dim)
        cell: (n_layers, batch_size, hidden_dim)
        """
        embedded = self.dropout(self.embedding(trg))
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc(outputs)
        return predictions, hidden, cell


class SimpleEncoderDecoder(nn.Module):
    """Simple Encoder-Decoder Model"""
    def __init__(self, encoder, decoder):
        super(SimpleEncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch_size, src_seq_len)
        trg: (batch_size, trg_seq_len)
        """
        batch_size = trg.shape[0]
        trg_seq_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_seq_len, trg_vocab_size)
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        
        hidden, cell = self.encoder(src)
        
        decoder_input = trg[:, 0].unsqueeze(1)  # BOS token
        
        for t in range(1, trg_seq_len):
            output, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = output.squeeze(1)
            
            teacher_force = np.random.random() < teacher_forcing_ratio
            if teacher_force:
                decoder_input = trg[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(2)
        
        return outputs


def train_simple_encoder_decoder(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
    """Train simple encoder-decoder model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model = model.to(device)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        
        for src, trg in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            src, trg = src.to(device), trg.to(device)
            
            optimizer.zero_grad()
            outputs = model(src, trg)
            
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]),
                trg.reshape(-1)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for src, trg in val_loader:
                src, trg = src.to(device), trg.to(device)
                outputs = model(src, trg)
                
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[2]),
                    trg.reshape(-1)
                )
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses
