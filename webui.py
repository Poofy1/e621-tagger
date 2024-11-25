import os
from flask import Flask, render_template, request, jsonify
import torch
import webbrowser
import threading
from PIL import Image
import torchvision.transforms as transforms
from model import ImageLabelModel
import pickle
from model import *
from train_text import *

env = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for model and vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
vocab = None

# Load model immediately when the module is imported
def load_model():
    global model, vocab
    
    # Load vocabulary
    with open(f'{env}/data/e621_vocabulary.pkl', 'rb') as f:
        vocab_dict = pickle.load(f)
        vocab = {v: k for k, v in vocab_dict.items()}
    
    # Initialize model
    model = ImageLabelModel(len(vocab)).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(f'{env}/checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")

load_model()  # Call this immediately

def predict(image_path):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load and transform image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    # Create dummy batch data
    batch = {
        'images': image,
        'labels': torch.tensor([[1]]).to(device),  # START token
        'attention_mask': torch.tensor([[1]]).to(device)
    }
    
    # Generate predictions
    with torch.no_grad():
        outputs = model(batch)
        predictions = outputs.argmax(dim=-1)[0]
        
    # Convert predictions to tags
    predicted_tags = []
    for idx in predictions:
        idx = idx.item()
        if idx == 2:  # END token
            break
        if idx not in [0, 1, 2, 3]:  # Skip special tokens
            predicted_tags.append(vocab[idx])
    
    return predicted_tags

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_tags():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        # Save uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # Get predictions
            predicted_tags = predict(filepath)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'tags': predicted_tags
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    return jsonify({'error': 'Unknown error'}), 500



def run_flask():
    app.run(debug=False, use_reloader=False)


    
def launch():
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Open the browser
    webbrowser.open_new('http://127.0.0.1:5000/')

    # Run main thread tasks
    #main_thread_tasks()



if __name__ == "__main__":
    launch()