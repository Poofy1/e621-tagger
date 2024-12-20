import os, sys
import io
from flask import Flask, render_template, request, jsonify
import torch
import webbrowser
import threading
from PIL import Image
import torchvision.transforms as transforms
from train.model import *
from train.train_text import *
import logging
from huggingface_hub import hf_hub_download
import warnings

# Suppress Flask development server warning
cli = sys.modules['flask.cli']
cli.show_server_banner = lambda *x: None
warnings.filterwarnings('ignore')
logging.getLogger('werkzeug').setLevel(logging.ERROR)

env = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# Global variables for model and vocabulary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
vocab = None

app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024  # 256MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

def download_model():
    # Create necessary directories
    os.makedirs(f'{env}/checkpoints', exist_ok=True)
    os.makedirs(f'{env}/data', exist_ok=True)
    
    # Define files to download
    files_to_download = [
        {'filename': 'best_model.pth', 'local_path': f'{env}/checkpoints/best_model.pth'},
        {'filename': 'e621_vocabulary.pkl', 'local_path': f'{env}/data/e621_vocabulary.pkl'},
        {'filename': 'tag_map.csv', 'local_path': f'{env}/data/tag_map.csv'}
    ]
    
    # Download each file if it doesn't exist
    for file in files_to_download:
        if not os.path.exists(file['local_path']):
            print(f"Downloading {file['filename']}...")
            hf_hub_download(
                repo_id="poofy38/e621-tagger-01",
                filename=file['filename'],
                local_dir=os.path.dirname(file['local_path']),
                repo_type="model"
            )
            print(f"{file['filename']} downloaded successfully")
    
    return f'{env}/checkpoints/best_model.pth'


def load_model():
    global model, vocab
    
    # Download model if needed
    model_path = download_model()
    
    # Load vocabulary
    vocab = Vocabulary.load(f'{env}/data/e621_vocabulary.pkl')
    
    # Initialize model
    model = ImageLabelModel(len(vocab)).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully")
    print(f"Using device: {device}")




def predict(image):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transform image
    image = transform(image).unsqueeze(0).to(device)

    # Generate predictions
    with torch.no_grad():
        outputs = model.generate(image)  # This already returns the token sequence
        predictions = outputs[0]  # Get first sequence from batch
        
    # Convert predictions to tags
    predicted_tags = []
    for idx in predictions:
        idx = idx.item()
        if idx == 2:  # END token
            break
        if idx not in [0, 1, 2, 3]:  # Skip special tokens
            predicted_tags.append(vocab[idx])
    
    output = ", ".join(predicted_tags)

    return output



@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Get prediction first
        prediction_text = predict(image)
        

        # Save file
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        return jsonify({
            'message': 'File uploaded and processed successfully',
            'prediction': prediction_text,
            'path': filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    
@app.route('/')
def home():
    return render_template('index.html')


def run_flask():
    app.run(debug=False, use_reloader=False)


def launch():
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Open the browser
    link = 'http://127.0.0.1:5000/'
    webbrowser.open_new('http://127.0.0.1:5000/')
    print(f'Opened webpage interface at: {link}')


if __name__ == "__main__":
    load_model()
    launch()
    
    #image = Image.open("C:/Users/Tristan/Pictures/6f30d53506600951da6effea9cd16833_planetary-landscape-alien-landscape-planets-mountains-clouds-fantasy-planet-drawing_3840-2160.jpg")
    #predict(image)
    