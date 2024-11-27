from transformers import ViTModel
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import torch.nn as nn
import torch
    
def generate_square_subsequent_mask(sz):
        """Generate mask for transformer decoder to prevent attending to future tokens."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
class ImageLabelModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768):
        super().__init__()
        self.vocab_size = vocab_size 
        self.start_token = 1 
        self.end_token = 2 
        self.max_length = 512
        
        self.vision_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224'
        )
        
        # Project vision features to sequence length
        self.vision_projection = nn.Linear(hidden_dim, hidden_dim)
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_length, hidden_dim))
        
        # Decoder will operate on vision features directly
        decoder_layer = TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=6)
        
        # Output layer predicts all tokens at once
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, batch, teacher_forcing_ratio=1.0):
        images = batch['images']
        labels = batch['labels']
        
        if labels.size(1) > self.max_length:
            labels = labels[:, :self.max_length]
            
        with torch.cuda.amp.autocast():  # Use mixed precision to reduce memory usage
            # Encode images
            vision_output = self.vision_encoder(images).last_hidden_state
            vision_features = self.vision_projection(vision_output)
            
            # Clear vision output as it's no longer needed
            del vision_output
            
            # Prepare decoder input sequence
            decoder_input = labels[:, :-1]  # Remove last token
            decoder_input_embedded = self.token_embedding(decoder_input)
            
            # Add positional embeddings
            positions = self.pos_embedding[:, :decoder_input_embedded.size(1), :]
            decoder_input_embedded = decoder_input_embedded + positions
            
            # Generate mask
            tgt_mask = generate_square_subsequent_mask(decoder_input.size(1)).to(decoder_input.device)
            
            # Decode with teacher forcing
            decoder_output = self.decoder(
                tgt=decoder_input_embedded.transpose(0, 1),
                memory=vision_features.transpose(0, 1),
                tgt_mask=tgt_mask
            )
            
            del decoder_input_embedded, vision_features, tgt_mask  # Clear intermediate tensors
            
            logits = self.output_layer(decoder_output.transpose(0, 1))
            del decoder_output  # Clear decoder output
            
        return logits, labels[:, 1:]  # Return logits and targets (removing START token)

    def generate(self, images, max_length=500):
        batch_size = images.size(0)
        
        # Encode images
        vision_output = self.vision_encoder(images).last_hidden_state
        vision_features = self.vision_projection(vision_output)
        
        # Start with START token for each sequence in batch
        current_tokens = torch.full((batch_size, 1), self.start_token, 
                                dtype=torch.long, device=images.device)
        
        for i in range(max_length):
            # Get token embeddings
            token_embeddings = self.token_embedding(current_tokens)
            
            # Add positional embeddings
            positions = self.pos_embedding[:, :token_embeddings.size(1), :]
            decoder_input = token_embeddings + positions
            
            # Create attention mask
            tgt_mask = generate_square_subsequent_mask(current_tokens.size(1)).to(images.device)
            
            # Decode
            decoder_output = self.decoder(
                tgt=decoder_input.transpose(0, 1),
                memory=vision_features.transpose(0, 1),
                tgt_mask=tgt_mask
            )
            
            # Get predictions for next token
            logits = self.output_layer(decoder_output.transpose(0, 1))
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append predicted token
            current_tokens = torch.cat([current_tokens, next_token], dim=1)
            
            # Stop if all sequences have END token
            if (current_tokens == self.end_token).any(dim=1).all():
                break
        
        # Trim sequences at END token
        final_sequences = []
        for seq in current_tokens:
            end_pos = (seq == self.end_token).nonzero()
            if len(end_pos) > 0:
                final_sequences.append(seq[:end_pos[0]])
            else:
                final_sequences.append(seq)
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in final_sequences)
        padded_sequences = torch.zeros(batch_size, max_len, 
                                    dtype=torch.long, device=images.device)
        for i, seq in enumerate(final_sequences):
            padded_sequences[i, :len(seq)] = seq
            
        return padded_sequences