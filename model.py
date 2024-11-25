from transformers import ViTModel
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn as nn
import torch

class ImageLabelModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super().__init__()
        self.vocab_size = vocab_size 
        self.start_token = 1 
        self.end_token = 2 
        
        self.vision_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224',
        )
            
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, 1024, hidden_dim))  # Max length for positional embedding
        
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=6)
        
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, batch):
        images = batch['images']
        labels = batch['labels']
        attention_mask = batch['attention_mask']
        seq_len = labels.size(1)
        
        # Encode images
        vision_output = self.vision_encoder(images).last_hidden_state
        
        # Always use teacher forcing during training and validation
        tgt_embeddings = self.token_embedding(labels)
        pos_embeddings = self.position_embedding[:, :seq_len, :]
        
        decoder_input = (tgt_embeddings + pos_embeddings).transpose(0, 1)
        
        decoder_output = self.decoder(
            tgt=decoder_input,
            memory=vision_output.transpose(0, 1),
            tgt_key_padding_mask=~attention_mask.bool()
        )
        logits = self.output_layer(decoder_output.transpose(0, 1))
        return logits