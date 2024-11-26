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
            nhead=4,
            dim_feedforward=1024,
            dropout=0.1
        )
        self.decoder = TransformerDecoder(decoder_layer, num_layers=3)
        
        # Output layer predicts all tokens at once
        self.output_layer = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim

    def forward(self, batch, teacher_forcing_ratio=0.8):
        images = batch['images']
        labels = batch['labels']
        
        if labels.size(1) > self.max_length:
            labels = labels[:, :self.max_length]
            
        batch_size = labels.size(0)
        max_len = labels.size(1) - 1  # -1 because we don't need to predict after the last token
        
        with torch.cuda.amp.autocast():  # Use mixed precision to reduce memory usage
            # Encode images
            vision_output = self.vision_encoder(images).last_hidden_state
            vision_features = self.vision_projection(vision_output)
            del vision_output
            
            # Initialize decoder input with START token
            decoder_input = labels[:, 0].unsqueeze(1)  # Get just the START token
            outputs = []
            
            for t in range(max_len):
                # Embed current input tokens
                decoder_input_embedded = self.token_embedding(decoder_input)
                
                # Add positional embeddings
                positions = self.pos_embedding[:, :decoder_input_embedded.size(1), :]
                decoder_input_embedded = decoder_input_embedded + positions
                
                # Generate mask
                tgt_mask = generate_square_subsequent_mask(decoder_input.size(1)).to(decoder_input.device)
                
                # Decode
                decoder_output = self.decoder(
                    tgt=decoder_input_embedded.transpose(0, 1),
                    memory=vision_features.transpose(0, 1),
                    tgt_mask=tgt_mask
                )
                
                # Get prediction for current timestep
                step_output = self.output_layer(decoder_output.transpose(0, 1))
                current_step_logits = step_output[:, -1:]  # Get just the last token prediction
                outputs.append(current_step_logits)
                
                # Teacher forcing decision for next input
                if t < max_len - 1:  # Don't need to generate input for the last prediction
                    use_teacher_forcing = (torch.rand(1).item() < teacher_forcing_ratio)
                    if use_teacher_forcing:
                        # Use ground truth token as next input
                        next_input = labels[:, t+1].unsqueeze(1)
                    else:
                        # Use model's prediction as next input
                        next_input = current_step_logits.argmax(dim=-1)
                    decoder_input = torch.cat([decoder_input, next_input], dim=1)
                
                # Clear unnecessary tensors
                del decoder_input_embedded, decoder_output, step_output, tgt_mask
            
            # Concatenate all outputs
            logits = torch.cat(outputs, dim=1)
            del outputs, vision_features
            
        return logits, labels[:, 1:]  # Return logits and targets (removing START token)

    
    
    def generate(self, images, max_length=500):
        batch_size = images.size(0)
        
        # Encode images
        vision_output = self.vision_encoder(images).last_hidden_state
        vision_features = self.vision_projection(vision_output)
        
        # Generate full sequence in one pass
        decoder_output = self.decoder(
            tgt=torch.zeros_like(vision_features).transpose(0, 1),
            memory=vision_features.transpose(0, 1)
        )
        
        logits = self.output_layer(decoder_output.transpose(0, 1))
        
        # Get most likely tokens for each position
        predicted_tokens = logits.argmax(dim=-1)
        
        # Trim sequences at END token if present
        final_sequences = []
        for seq in predicted_tokens:
            # Find position of first END token
            end_pos = (seq == self.end_token).nonzero()
            if len(end_pos) > 0:
                # Keep sequence up to first END token
                final_sequences.append(seq[:end_pos[0]])
            else:
                # Keep full sequence if no END token
                final_sequences.append(seq)
                
        # Pad sequences to same length
        max_len = max(len(seq) for seq in final_sequences)
        padded_sequences = torch.zeros(batch_size, max_len, device=images.device)
        for i, seq in enumerate(final_sequences):
            padded_sequences[i, :len(seq)] = seq
            
        return padded_sequences