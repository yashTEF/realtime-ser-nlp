"""
Neural network model definitions for emotion recognition.
Contains the multimodal LSTM emotion classifier architecture.
"""

import torch
import torch.nn as nn
from config import EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, N_MFCC, MAX_AUDIO_LENGTH


class Multimodal_LSTMEmotionClassifier(nn.Module):
    """
    Multimodal LSTM-based emotion classifier that processes both text and audio features.
    
    Architecture:
    - Text branch: Embedding layer + Bidirectional LSTM
    - Audio branch: Layer normalization + Bidirectional LSTM  
    - Combined: Concatenated features + Fully connected layers
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout, embedding_matrix, n_mfcc=13):
        super(Multimodal_LSTMEmotionClassifier, self).__init__()
        
        # Text branch (BiLSTM)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.text_bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.text_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Audio branch (BiLSTM)
        self.layer_norm = nn.LayerNorm(n_mfcc)
        self.audio_bilstm = nn.LSTM(
            input_size=n_mfcc,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.audio_fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Combined
        self.combined_fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids, mfcc, seq_lens):
        """
        Forward pass through the multimodal network.
        
        Args:
            input_ids: Text token indices [batch, max_length]
            mfcc: MFCC features [batch, max_audio_length, n_mfcc]
            seq_lens: Actual sequence lengths for audio [batch]
            
        Returns:
            Emotion class logits [batch, num_classes]
        """
        # Text branch
        embedded = self.embedding(input_ids)  # [batch, max_length, embedding_dim]
        text_out, (hn_text, _) = self.text_bilstm(embedded)  # [batch, max_length, hidden_dim * 2]
        text_features = self.text_fc(torch.cat((hn_text[-2], hn_text[-1]), dim=1))  # [batch, hidden_dim]
        
        # Audio branch
        mfcc = self.layer_norm(mfcc)  # [batch, max_audio_length, n_mfcc]
        packed_audio = nn.utils.rnn.pack_padded_sequence(
            mfcc, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hn_audio, _) = self.audio_bilstm(packed_audio)  # hn_audio: [num_layers * 2, batch, hidden_dim]
        audio_features = self.audio_fc(torch.cat((hn_audio[-2], hn_audio[-1]), dim=1))  # [batch, hidden_dim]
        
        # Combine
        combined_features = torch.cat((text_features, audio_features), dim=1)  # [batch, hidden_dim * 2]
        output = self.combined_fc(combined_features)  # [batch, num_classes]
        return output


def create_model(vocab_size, embedding_matrix, num_classes):
    """
    Create a new instance of the multimodal emotion classifier.
    
    Args:
        vocab_size: Size of the vocabulary
        embedding_matrix: Pre-trained embedding matrix
        num_classes: Number of emotion classes
        
    Returns:
        Initialized model instance
    """
    model = Multimodal_LSTMEmotionClassifier(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
        embedding_matrix=embedding_matrix,
        n_mfcc=N_MFCC
    )
    return model