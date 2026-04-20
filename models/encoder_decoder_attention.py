"""
Encoder-Decoder Architecture with Attention Mechanism
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm


class Attention(nn.Module):
    """Bahdanau Attention Mechanism"""
    def __init__(self, hidden_dim, encoder_hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim + encoder_hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        """
        hidden: (batch_size, hidden_dim)
        encoder_outputs: (batch_size, seq_len, encoder_hidden_dim)
        """
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        # Expand hidden to match seq_len
        hidden_expanded = hidden.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Concatenate hidden and encoder outputs
        combined = torch.cat((hidden_expanded, encoder_outputs), dim=2)
        
        # Calculate attention scores
        attn_scores = self.attn(combined)
        attn_scores = torch.tanh(attn_scores)
        attention = self.v(attn_scores).squeeze(2)
        
        # Apply softmax
        attention_weights = torch.softmax(attention, dim=1)
        
        # Calculate context vector
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)
        
        return context, attention_weights


class AttentionEncoder(nn.Module):
    """Encoder with attention"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers=2, dropout=0.5):
        super(AttentionEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            n_layers, 
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc_hidden = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_cell = nn.Linear(hidden_dim * 2, hidden_dim)
        self.n_layers = n_layers
    
    def forward(self, src):
        """
        src: (batch_size, seq_len)
        """
        embedded = self.dropout(self.embedding(src))
        encoder_outputs, (hidden, cell) = self.lstm(embedded)
        
        # Convert bidirectional to unidirectional for each layer
        # hidden: (n_layers * 2, batch_size, hidden_dim)
        # We need to combine the forward and backward hidden states
        new_hidden = []
        new_cell = []
        
        for layer in range(self.n_layers):
            # Get forward and backward hidden states for this layer
            forward_h = hidden[layer * 2, :, :]
            backward_h = hidden[layer * 2 + 1, :, :]
            forward_c = cell[layer * 2, :, :]
            backward_c = cell[layer * 2 + 1, :, :]
            
            # Combine forward and backward
            combined_h = torch.cat((forward_h, backward_h), dim=1)
            combined_c = torch.cat((forward_c, backward_c), dim=1)
            
            # Transform to decoder hidden dimension
            h = self.fc_hidden(combined_h)
            c = self.fc_cell(combined_c)
            
            new_hidden.append(h)
            new_cell.append(c)
        
        hidden = torch.stack(new_hidden)
        cell = torch.stack(new_cell)
        
        return encoder_outputs, hidden, cell


class AttentionDecoder(nn.Module):
    """Decoder with attention"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, encoder_hidden_dim, n_layers=2, dropout=0.5):
        super(AttentionDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.attention = Attention(hidden_dim, encoder_hidden_dim)
        self.lstm = nn.LSTM(
            embedding_dim + encoder_hidden_dim,  # embedding + context
            hidden_dim, 
            n_layers, 
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        # FC output layer: hidden_dim + encoder_hidden_dim -> vocab_size
        self.fc = nn.Linear(hidden_dim + encoder_hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg, hidden, cell, encoder_outputs):
        """
        trg: (batch_size, seq_len)
        hidden: (n_layers, batch_size, hidden_dim)
        cell: (n_layers, batch_size, hidden_dim)
        encoder_outputs: (batch_size, seq_len, encoder_hidden_dim)
        """
        embedded = self.dropout(self.embedding(trg))
        
        # Get context from attention
        context, attention_weights = self.attention(hidden[-1], encoder_outputs)
        # context: (batch_size, encoder_hidden_dim)
        
        # Concatenate embedding and context
        # embedded: (batch_size, 1, embedding_dim)
        # context.unsqueeze(1): (batch_size, 1, encoder_hidden_dim)
        lstm_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
        
        outputs, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        # outputs: (batch_size, 1, hidden_dim)
        
        # Repeat context to match sequence length of outputs
        batch_size = outputs.size(0)
        seq_len = outputs.size(1)
        context_expanded = context.unsqueeze(1).expand(batch_size, seq_len, -1)
        # context_expanded: (batch_size, 1, encoder_hidden_dim)
        
        # Concatenate output and context for final prediction
        combined = torch.cat((outputs, context_expanded), dim=2)
        # combined: (batch_size, 1, hidden_dim + encoder_hidden_dim)
        predictions = self.fc(combined)
        
        return predictions, hidden, cell, attention_weights


class EncoderDecoderWithAttention(nn.Module):
    """Encoder-Decoder Model with Attention"""
    def __init__(self, encoder, decoder, encoder_hidden_dim=512):
        super(EncoderDecoderWithAttention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_hidden_dim = encoder_hidden_dim
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: (batch_size, src_seq_len)
        trg: (batch_size, trg_seq_len)
        """
        batch_size = trg.shape[0]
        trg_seq_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        
        outputs = torch.zeros(batch_size, trg_seq_len, trg_vocab_size)
        attention_weights_all = []
        
        if torch.cuda.is_available():
            outputs = outputs.cuda()
        
        encoder_outputs, hidden, cell = self.encoder(src)
        
        decoder_input = trg[:, 0].unsqueeze(1)  # BOS token
        
        for t in range(1, trg_seq_len):
            output, hidden, cell, attention_weights = self.decoder(
                decoder_input, hidden, cell, encoder_outputs
            )
            outputs[:, t, :] = output.squeeze(1)
            attention_weights_all.append(attention_weights)
            
            teacher_force = np.random.random() < teacher_forcing_ratio
            if teacher_force:
                decoder_input = trg[:, t].unsqueeze(1)
            else:
                decoder_input = output.argmax(2)
        
        return outputs, attention_weights_all


def train_encoder_decoder_with_attention(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
    """Train encoder-decoder model with attention"""
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
            outputs, _ = model(src, trg)
            
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
                outputs, _ = model(src, trg)
                
                loss = criterion(
                    outputs.reshape(-1, outputs.shape[2]),
                    trg.reshape(-1)
                )
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return train_losses, val_losses
