
import torch,os, pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import ViTModel
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import pyarrow.parquet as pq
env = os.path.dirname(os.path.abspath(__file__))


import matplotlib.pyplot as plt
import torchvision.transforms as T
from textwrap import wrap

def plot_predictions(model, batch, dataset, save_path, device, num_samples=3):
    # Denormalize images for display
    denorm = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # Get predictions
    with torch.no_grad():
        outputs = model(batch)
        predictions = outputs.argmax(dim=-1)
    
    # Create subplot grid with more height for text
    fig, axes = plt.subplots(num_samples, 1, figsize=(20, 8*num_samples))
    
    for i in range(min(num_samples, len(batch['images']))):
        # Get and denormalize image
        img = denorm(batch['images'][i]).cpu()
        img = torch.clamp(img, 0, 1)
        
        # Convert predicted indices to tags
        pred_tags = []
        for idx in predictions[i]:
            if idx.item() == dataset.vocab.word2index['<END>']:
                break
            if idx.item() not in [dataset.vocab.word2index['<PAD>'], 
                                 dataset.vocab.word2index['<START>'], 
                                 dataset.vocab.word2index['<UNK>']]:
                pred_tags.append(dataset.vocab.index2word[idx.item()])
        
        # Get ground truth tags
        true_tags = []
        for idx in batch['labels'][i]:
            if idx.item() == dataset.vocab.word2index['<END>']:
                break
            if idx.item() not in [dataset.vocab.word2index['<PAD>'], 
                                 dataset.vocab.word2index['<START>'], 
                                 dataset.vocab.word2index['<UNK>']]:
                true_tags.append(dataset.vocab.index2word[idx.item()])
        
        # Wrap text for better display
        pred_text = "Predicted: " + ", ".join(pred_tags)
        true_text = "Actual: " + ", ".join(true_tags)
        
        # Split text into multiple lines if too long
        pred_wrapped = "\n".join(wrap(pred_text, width=120))
        true_wrapped = "\n".join(wrap(true_text, width=120))
        
        # Plot
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].axis('off')
        
        # Add text with smaller font and more space
        axes[i].set_title(f'{pred_wrapped}\n\n{true_wrapped}',
                         fontsize=8, wrap=True, pad=20)
    
    plt.tight_layout(pad=3.0)  # Increase padding between subplots
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    
class Vocabulary:
    def __init__(self):
        self.word2index = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "<UNK>": 3,
        }
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.num_words = len(self.word2index)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.index2word[self.num_words] = word
            self.num_words += 1

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.word2index, f)

    @classmethod
    def load(cls, path):
        vocab = cls()
        with open(path, 'rb') as f:
            vocab.word2index = pickle.load(f)
            vocab.index2word = {v: k for k, v in vocab.word2index.items()}
            vocab.num_words = len(vocab.word2index)
        return vocab
    
    def __len__(self):
        return self.num_words
    
def load_tag_map(file_path):
    df = pd.read_csv(file_path)
    return dict(zip(df['index'], df['tag']))

class E621Dataset(Dataset):
    def __init__(self, parquet_file, img_dir, tag_map_file, transform=None, vocab_path=None):
        self.img_dir = img_dir
        self.transform = transform
        
        print("Loading data...")
        self.data = pq.read_table(parquet_file).to_pandas()
        
        # Filter for valid = 0 (training set)
        self.data = self.data[self.data['Valid'] == 0].reset_index(drop=True)

        # Load tag map
        self.tag_map = load_tag_map(tag_map_file)

        # Clean tag map - remove non-string entries
        self.tag_map = {k: v for k, v in self.tag_map.items() if isinstance(v, str)}

        # Get all unique tags
        self.all_tags = set()
        for tags in self.data['tag_indices']:
            self.all_tags.update(tags)
            
        print(f"Total unique tags: {len(self.all_tags)}")
        print(f"Total training samples: {len(self.data)}")

        if vocab_path and os.path.exists(vocab_path):
            self.vocab = Vocabulary.load(vocab_path)
            print(f"Loaded vocabulary with {len(self.vocab)} words")
        else:
            self.vocab = Vocabulary()
            self.build_vocab()
            if vocab_path:
                self.vocab.save(vocab_path)
                print(f"Saved vocabulary with {len(self.vocab)} words to {vocab_path}")

    def build_vocab(self):
        for tags in tqdm(self.data['tag_indices'], desc="Building vocabulary"):
            tag_labels = [self.tag_map[idx] for idx in tags if idx in self.tag_map]
            for tag in tag_labels:
                self.vocab.add_word(tag)
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, f"{self.data.iloc[idx]['ID']}.png")
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        tag_indices = self.data.iloc[idx]['tag_indices']
        
        tags = [self.tag_map[idx] for idx in tag_indices if idx in self.tag_map]
        
        # Randomly shuffle tags during training
        random.shuffle(tags)
        
        # Modified label creation - simpler separator
        label_string = "<START> " + " ".join(tags) + " <END>"
        
        # Convert to indices
        label_indices = [self.vocab.word2index.get(word.strip(), self.vocab.word2index['<UNK>']) 
                        for word in label_string.split()]
            
        label_tensor = torch.LongTensor(label_indices)
        
        return image, label_tensor


def custom_collate(batch):
    images, label_tensors = zip(*batch)
    images = torch.stack(images)
    
    # Pad sequences to the same length
    max_len = max(tensor.size(0) for tensor in label_tensors)
    padded_labels = torch.full((len(label_tensors), max_len), 
                             fill_value=0,  # PAD token index
                             dtype=torch.long)
    
    # Create attention mask
    attention_mask = torch.zeros(len(label_tensors), max_len)
    
    for i, tensor in enumerate(label_tensors):
        end = tensor.size(0)
        padded_labels[i, :end] = tensor[:end]
        attention_mask[i, :end] = 1
    
    return {
        'images': images,
        'labels': padded_labels,
        'attention_mask': attention_mask,
        'label_lengths': torch.LongTensor([len(t) for t in label_tensors])
    }

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
        batch_size = images.shape[0]
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

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, batch_limit=1000):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=batch_limit, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in progress_bar:
        if batch_idx >= batch_limit:
            break
            
        # Move data to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(batch)
        
        loss = criterion(outputs.view(-1, outputs.size(-1)), 
                        batch['labels'].view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return avg_loss

def validate(model, dataloader, criterion, device, epoch, save_dir, dataset, batch_limit=200):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=batch_limit, desc='Validation')
    
    # Create directory for validation plots if it doesn't exist
    plot_dir = os.path.join(save_dir, 'validation_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            if batch_idx >= batch_limit:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get model outputs
            outputs = model(batch)
            
            # Calculate loss
            loss = criterion(
                outputs.contiguous().view(-1, outputs.size(-1)),
                batch['labels'].contiguous().view(-1)
            )
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Save plot for first batch
            if batch_idx == 0:
                plot_predictions(
                    model, 
                    batch,
                    dataset,
                    os.path.join(plot_dir, f'val_epoch_{epoch}.png'),
                    device
                )
            
    return total_loss / min(batch_limit, len(dataloader))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = E621Dataset(f'{env}/data/dataset.parquet', 'F:/Temp_SSD_Data/ME621/images_300/', 'F:/CODE/AI/e621-tagger/data/tag_map.csv', transform=transform, vocab_path='F:/CODE/AI/e621-tagger/data/e621_vocabulary.pkl')

    # Model setup
    vocab_size = len(dataset.vocab)
    model = ImageLabelModel(vocab_size).to(device)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")  
    
    # For training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.word2index['<PAD>']).to(device)



    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=custom_collate)

    num_epochs = 5
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    save_dir = f'{env}/checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # Train on 1000 batches
        train_loss = train_one_epoch(model, train_loader, criterion, 
                                   optimizer, device, epoch, batch_limit=500)
        
        # Validate on batches
        val_loss = validate(model, val_loader, criterion, device, 
                        epoch, save_dir, dataset, batch_limit=100)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab': dataset.vocab
            }
            torch.save(checkpoint, f'{save_dir}/best_model.pth')
            print(f'Saved new best model with val_loss: {val_loss:.4f}')
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        # Save latest model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab': dataset.vocab
        }
        torch.save(checkpoint, f'{save_dir}/latest_model.pth')

    print('Training completed!')