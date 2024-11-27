
import torch, os, pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import sys
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import torchvision.transforms as T
from textwrap import wrap

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from train.model import *


current_dir = os.path.dirname(os.path.abspath(__file__))
env = os.path.dirname(current_dir)


def plot_predictions(model, batch, dataset, save_path, device, num_samples=3):
    # Denormalize images for display
    denorm = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    
    # Get predictions
    with torch.no_grad():
        logits, _ = model(batch)  # Unpack the tuple
        # Reshape predictions
        predictions = logits.view(-1, logits.size(-1))
        pred_indices = torch.argmax(predictions, dim=1).cpu().numpy()
        # Reshape back to batch dimension
        pred_indices = pred_indices.reshape(logits.size(0), -1)
        
        # Also get full sequence generations for comparison
        generated = model.generate(batch['images'])
    
    # Create subplot grid with more height for text
    fig, axes = plt.subplots(num_samples, 1, figsize=(20, 8*num_samples))
    if num_samples == 1:
        axes = [axes]
    
    for i in range(min(num_samples, len(batch['images']))):
        # Get and denormalize image
        img = denorm(batch['images'][i]).cpu()
        img = torch.clamp(img, 0, 1)
        
        # Convert predicted indices to tags
        pred_tags = []
        for idx in pred_indices[i]:
            pred_tags.append(dataset.vocab.index2word[idx])
            
        # Convert generated indices to tags
        gen_tags = []
        for idx in generated[i]:
            tag = dataset.vocab.index2word[idx.item()]
            if tag not in ['<PAD>', '<START>', '<END>']:
                gen_tags.append(tag)
        
        # Get ground truth tags
        true_tags = []
        for idx in batch['labels'][i]:
            tag = dataset.vocab.index2word[idx.item()]
            if tag not in ['<PAD>', '<START>', '<END>']:
                true_tags.append(tag)
        
        # Wrap text for better display
        pred_text = "Next Token Predictions: " + ", ".join(pred_tags)
        gen_text = "Full Generation: " + ", ".join(gen_tags)
        true_text = "Ground Truth: " + ", ".join(true_tags)
        
        # Split text into multiple lines if too long
        pred_wrapped = "\n".join(wrap(pred_text, width=120))
        gen_wrapped = "\n".join(wrap(gen_text, width=120))
        true_wrapped = "\n".join(wrap(true_text, width=120))
        
        # Plot
        axes[i].imshow(img.permute(1, 2, 0))
        axes[i].axis('off')
        
        # Add text with smaller font and more space
        axes[i].set_title(f'{pred_wrapped}\n\n{gen_wrapped}\n\n{true_wrapped}',
                         fontsize=8, wrap=True, pad=20)
    
    plt.tight_layout(pad=3.0)
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
    
    
    def __getitem__(self, idx):
        return self.index2word.get(idx, self.index2word[3])  # Return <UNK> if idx not found
    
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
        
        # Get tags and sort them alphabetically
        tags = [self.tag_map[idx] for idx in tag_indices if idx in self.tag_map]
        tags = sorted(tags)  # Sort alphabetically
        
        # Create label string with sorted tags
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


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, batch_limit=100):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=batch_limit, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in progress_bar:
        if batch_idx >= batch_limit:
            break
            
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        logits, target_tokens = model(batch, teacher_forcing_ratio=1.0)
        
        B, S, V = logits.shape
        loss = criterion(logits.reshape(B*S, V), target_tokens.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():  # Prevent memory accumulation during metric calculation
            avg_loss = loss.item()
            predictions = logits.argmax(dim=-1)
            mask = target_tokens != model.end_token
            accuracy = (predictions == target_tokens)[mask].float().mean().item()
        
        # Explicitly clear memory
        del loss, logits, predictions, mask
        torch.cuda.empty_cache()  # Clear unused memory
        
        total_loss += avg_loss
        avg_loss = total_loss / (batch_idx + 1)
        
        progress_bar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'acc': f'{accuracy:.3f}'
        })
    
    return avg_loss

def validate(model, dataloader, criterion, device, epoch, save_dir, dataset, batch_limit=200):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    progress_bar = tqdm(enumerate(dataloader), total=batch_limit, desc='Validation')
    
    plot_dir = os.path.join(save_dir, 'validation_plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in progress_bar:
            if batch_idx >= batch_limit:
                break
                
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Get model outputs using teacher forcing (for loss calculation)
            logits, target_tokens = model(batch, teacher_forcing_ratio=1.0)
            
            # Calculate loss
            B, S, V = logits.shape
            loss = criterion(logits.reshape(B*S, V), target_tokens.reshape(-1))
            
            # Calculate accuracy (ignoring padding tokens)
            predictions = logits.argmax(dim=-1)
            mask = target_tokens != model.end_token
            accuracy = (predictions == target_tokens)[mask].float().mean().item()
            
            total_loss += loss.item()
            total_accuracy += accuracy
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.3f}'
            })
            
            # Save plot for first batch
            if batch_idx == 0:
                plot_predictions(
                    model, 
                    batch,
                    dataset,
                    os.path.join(plot_dir, f'val_epoch_{epoch}.png'),
                    device
                )
    
    avg_loss = total_loss / min(batch_limit, len(dataloader))
    avg_accuracy = total_accuracy / min(batch_limit, len(dataloader))
    
    print(f"\nValidation Results - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.3f}")
    
    return avg_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = E621Dataset(f'{env}/data/dataset.parquet', 'F:/Temp_SSD_Data/ME621/images_300/', 
                         f'{env}/data/tag_map.csv', transform=transform, 
                         vocab_path=f'{env}/data/e621_vocabulary.pkl')

    # Model setup
    vocab_size = len(dataset.vocab)
    model = ImageLabelModel(vocab_size).to(device)
    
    # Load checkpoint
    checkpoint_path = f'{env}/checkpoints/best_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch} with validation loss: {best_val_loss:.4f}")
    else:
        print("No checkpoint found, starting from scratch")
        start_epoch = 0
        best_val_loss = float('inf')

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters())}")  
    
    # For training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    if os.path.exists(checkpoint_path):
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.word2index['<PAD>']).to(device)

    # Split dataset into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=1, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=1, collate_fn=custom_collate)

    num_epochs = 500
    patience = 5
    patience_counter = 0
    save_dir = f'{env}/checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, 
                                   device, epoch, batch_limit=5000)
        
        val_loss = validate(model, val_loader, criterion, device, epoch, 
                          save_dir, dataset, batch_limit=100)
        
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
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
            
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    print('Training completed!')
    
    
