"""
Text processing utilities for preprocessing text input for the emotion recognition model.
Handles tokenization, vocabulary mapping, and tensor conversion.
"""

import torch
import nltk
from config import MAX_LENGTH


class TextProcessor:
    """Handles all text preprocessing operations."""
    
    def __init__(self):
        self.word2idx = None
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            print(f"Warning: Could not download NLTK data: {e}")
    
    def set_vocabulary(self, word2idx):
        """Set the word-to-index vocabulary mapping."""
        self.word2idx = word2idx
    
    def preprocess_text(self, text):
        """
        Preprocess text for model input.
        
        Args:
            text: Input text string
            
        Returns:
            Torch tensor of token indices with batch dimension
        """
        try:
            if self.word2idx is None:
                raise ValueError("Vocabulary not loaded")
            
            # Tokenize
            tokens = nltk.word_tokenize(text.lower())[:MAX_LENGTH]
            
            # Convert to indices
            input_ids = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
            
            # Pad to max length
            if len(input_ids) < MAX_LENGTH:
                input_ids += [self.word2idx['<PAD>']] * (MAX_LENGTH - len(input_ids))
            
            return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # Add batch dimension
            
        except Exception as e:
            print(f"Error preprocessing text: {e}")
            # Return dummy tensor
            return torch.zeros(1, MAX_LENGTH, dtype=torch.long)
    
    def create_default_vocabulary(self, vocab_size=1000):
        """
        Create a default vocabulary for demo purposes.
        
        Args:
            vocab_size: Target vocabulary size
            
        Returns:
            Dictionary mapping words to indices
        """
        word2idx = {'<PAD>': 0, '<UNK>': 1}
        
        # Add common words
        common_words = [
            'i', 'you', 'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 
            'that', 'have', 'for', 'not', 'with', 'he', 'as', 'his', 'on', 'be',
            'feel', 'good', 'bad', 'angry', 'happy', 'sad', 'neutral', 'today',
            'really', 'very', 'much', 'like', 'love', 'hate', 'this', 'that',
            'am', 'are', 'was', 'were', 'will', 'would', 'could', 'should',
            'me', 'my', 'mine', 'we', 'us', 'our', 'they', 'them', 'their'
        ]
        
        for word in common_words:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
        
        # Pad vocabulary to target size
        while len(word2idx) < vocab_size:
            word2idx[f'word_{len(word2idx)}'] = len(word2idx)
        
        return word2idx
    
    def get_vocabulary_info(self):
        """Get information about the current vocabulary."""
        if self.word2idx is None:
            return {"status": "No vocabulary loaded"}
        
        return {
            "vocab_size": len(self.word2idx),
            "special_tokens": {
                "PAD": self.word2idx.get('<PAD>', 'Not found'),
                "UNK": self.word2idx.get('<UNK>', 'Not found')
            },
            "sample_words": list(self.word2idx.keys())[:20]
        }


# Global text processor instance
text_processor = TextProcessor()