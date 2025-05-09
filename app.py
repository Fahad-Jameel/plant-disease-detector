# app.py
import os
import uuid
import json
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import gdown  # For downloading models from Google Drive

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_FOLDER'] = 'models'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Ensure upload and model directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Disease information dictionary
with open('disease_info.json', 'r') as f:
    DISEASE_INFO = json.load(f)

# Class mapping
CLASS_MAPPING = {
    'Pepper__bell___Bacterial_spot': 'Pepper Bell Bacterial Spot',
    'Pepper__bell___healthy': 'Pepper Bell Healthy',
    'Potato___Early_blight': 'Potato Early Blight',
    'Potato___Late_blight': 'Potato Late Blight',
    'Potato___healthy': 'Potato Healthy',
    'Tomato_Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato_Early_blight': 'Tomato Early Blight',
    'Tomato_Late_blight': 'Tomato Late Blight',
    'Tomato_Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato_Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'Tomato Spider Mites',
    'Tomato__Target_Spot': 'Tomato Target Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato__Tomato_mosaic_virus': 'Tomato Mosaic Virus',
    'Tomato_healthy': 'Tomato Healthy'
}

# Available models
AVAILABLE_MODELS = {
    'custom_cnn': {
        'name': 'Custom CNN',
        'file': 'custom_cnn_final.keras',
        'description': 'A custom convolutional neural network designed specifically for plant disease detection.'
    },
    'resnet50': {
        'name': 'ResNet50',
        'file': 'resnet50_final.keras',
        'description': 'A deep residual network with 50 layers that performs well on a variety of image classification tasks.'
    },
    'efficientnet': {
        'name': 'EfficientNet B3',
        'file': 'efficientnet_final.keras',
        'description': 'An efficient architecture that uses a compound scaling method to balance network depth, width, and resolution.'
    },
    'mobilenet': {
        'name': 'MobileNet V2',
        'file': 'mobilenet_final.keras',
        'description': 'A lightweight model designed for mobile and edge devices with limited computational resources.'
    }
}

# Model Google Drive URLs
MODEL_URLS = {
    'custom_cnn': 'https://drive.google.com/uc?id=1p4aEkqeKyC0q8pz40TNWZ9qlHt_TKmDT',
    'resnet50': 'https://drive.google.com/uc?id=1k2994YpbE-4Y8NTXihQ4uoHxPVHr0JSB',
    'efficientnet': 'https://drive.google.com/uc?id=10qQfIIn-v4Ps5gXmmCz_QN_8UwBzDdZl',
    'mobilenet': 'https://drive.google.com/uc?id=1p4aEkqeKyC0q8pz40TNWZ9qlHt_TKmDT'
}

# Default model
DEFAULT_MODEL = 'efficientnet'
IMG_SIZE = 256

# Function to ensure models are available
def ensure_models_available():
    """Download models if not available locally"""
    os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)
    
    for model_name, url in MODEL_URLS.items():
        model_path = os.path.join(app.config['MODEL_FOLDER'], AVAILABLE_MODELS[model_name]['file'])
        if not os.path.exists(model_path):
            print(f"Downloading {model_name} model...")
            try:
                gdown.download(url, model_path, quiet=False)
                print(f"Downloaded {model_name} model")
            except Exception as e:
                print(f"Error downloading {model_name} model: {e}")

# Call this function at app startup
ensure_models_available()

# Load the default model at startup
loaded_models = {}

def load_model(model_name):
    """Load a model if it's not already loaded"""
    if model_name not in loaded_models:
        model_path = os.path.join(app.config['MODEL_FOLDER'], AVAILABLE_MODELS[model_name]['file'])
        if os.path.exists(model_path):
            try:
                loaded_models[model_name] = tf.keras.models.load_model(model_path)
                print(f"Loaded model: {model_name}")
            except Exception as e:
                print(f"Error loading model {model_name}: {e}")
                return None
        else:
            print(f"Model file not found: {model_path}")
            
            # Try to download the model
            try:
                url = MODEL_URLS.get(model_name)
                if url:
                    print(f"Attempting to download {model_name} model...")
                    gdown.download(url, model_path, quiet=False)
                    print(f"Downloaded {model_name} model, loading...")
                    loaded_models[model_name] = tf.keras.models.load_model(model_path)
                    print(f"Loaded model: {model_name}")
                else:
                    print(f"No URL available for {model_name} model")
                    return None
            except Exception as e:
                print(f"Error downloading or loading model {model_name}: {e}")
                return None
    
    return loaded_models[model_name]

# Preload the default model
try:
    default_model = load_model(DEFAULT_MODEL)
except Exception as e:
    print(f"Failed to load default model: {e}")
    default_model = None

# Utility Functions
def preprocess_image(image_path):
    """Preprocess the image for model input"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def generate_gradcam(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM visualization for the given image and model"""
    # Create a model that maps the input image to the activations of the last conv layer
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Compute the gradient of the top predicted class with respect to the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Gradient of the predicted class with respect to the output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Average the gradients over all dimensions except channels
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps by their gradient importance
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def get_last_conv_layer(model, model_name):
    """Get the name of the last convolutional layer for the given model"""
    if model_name == 'custom_cnn':
        # Find the last convolutional layer in the custom CNN
        for i in range(len(model.layers) - 1, -1, -1):
            if isinstance(model.layers[i], tf.keras.layers.Conv2D):
                return model.layers[i].name
    elif model_name == 'resnet50':
        return 'conv5_block3_out'
    elif model_name == 'efficientnet':
        # For EfficientNet, try to find a specific layer or the last conv layer
        try:
            model.get_layer('top_activation')
            return 'top_activation'
        except:
            # Find the last convolutional layer
            for i in range(len(model.layers) - 1, -1, -1):
                if isinstance(model.layers[i], tf.keras.layers.Conv2D):
                    return model.layers[i].name
    elif model_name == 'mobilenet':
        # For MobileNet, try to find a specific layer or the last conv layer
        try:
            model.get_layer('Conv_1')
            return 'Conv_1'
        except:
            # Find the last convolutional layer
            for i in range(len(model.layers) - 1, -1, -1):
                if isinstance(model.layers[i], tf.keras.layers.Conv2D):
                    return model.layers[i].name
    
    # Default fallback - find any convolutional layer
    for i in range(len(model.layers) - 1, -1, -1):
        if isinstance(model.layers[i], tf.keras.layers.Conv2D):
            return model.layers[i].name
    
    # If no convolutional layer is found
    raise ValueError("Could not find a convolutional layer in the model")

def save_heatmap_visualization(img_path, heatmap, output_path):
    """Create and save a heatmap visualization"""
    # Load the original image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Create a heatmap visualization
    plt.figure(figsize=(10, 10))
    
    # Display heatmap over image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Routes
@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html', class_count=len(CLASS_MAPPING), models=AVAILABLE_MODELS)

@app.route('/about')
def about():
    """Render the about page"""
    return render_template('about.html')

@app.route('/models')
def models_page():
    """Render the models page"""
    return render_template('models.html', models=AVAILABLE_MODELS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and disease prediction"""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    # Get the selected model (or use default)
    model_name = request.form.get('model', DEFAULT_MODEL)
    if model_name not in AVAILABLE_MODELS:
        model_name = DEFAULT_MODEL
    
    # Load the selected model
    model = load_model(model_name)
    if model is None:
        return jsonify({'success': False, 'error': f'Failed to load {model_name} model'})

    try:
        # Save the uploaded file
        filename = secure_filename(str(uuid.uuid4()) + os.path.splitext(file.filename)[1])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess the image
        img_array = preprocess_image(filepath)

        # Make prediction
        preds = model.predict(img_array)
        pred_class_idx = np.argmax(preds[0])
        confidence = float(preds[0][pred_class_idx]) * 100

        # Get the class name
        class_names = list(CLASS_MAPPING.keys())
        pred_class = class_names[pred_class_idx]
        display_name = CLASS_MAPPING[pred_class]

        # Generate Grad-CAM heatmap
        try:
            last_conv_layer = get_last_conv_layer(model, model_name)
            heatmap = generate_gradcam(img_array, model, last_conv_layer, pred_class_idx)
            
            # Save the heatmap visualization
            heatmap_filename = f"heatmap_{filename}"
            heatmap_filepath = os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename)
            save_heatmap_visualization(filepath, heatmap, heatmap_filepath)
        except Exception as e:
            print(f"Error generating heatmap: {e}")
            heatmap_filename = None

        # Get disease information
        if pred_class in DISEASE_INFO:
            disease_info = DISEASE_INFO[pred_class]
        else:
            disease_info = {
                'name': display_name,
                'description': 'No detailed information available for this condition.',
                'symptoms': 'Information not available.',
                'treatment': 'Please consult with a plant pathologist.',
                'prevention': 'General care recommendations apply.'
            }

        return jsonify({
            'success': True,
            'filename': filename,
            'heatmap': heatmap_filename,
            'class': pred_class,
            'display_name': display_name,
            'confidence': confidence,
            'disease_info': disease_info,
            'model_used': AVAILABLE_MODELS[model_name]['name']
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/dataset/info')
def dataset_info():
    """Return information about the dataset"""
    return jsonify({
        'total_classes': len(CLASS_MAPPING),
        'class_names': list(CLASS_MAPPING.values())
    })

# Model status route to check if models are available
@app.route('/models/status')
def models_status():
    """Return the status of model downloads"""
    status = {}
    for model_name in AVAILABLE_MODELS:
        model_path = os.path.join(app.config['MODEL_FOLDER'], AVAILABLE_MODELS[model_name]['file'])
        status[model_name] = {
            'name': AVAILABLE_MODELS[model_name]['name'],
            'available': os.path.exists(model_path),
            'size_mb': round(os.path.getsize(model_path) / (1024 * 1024), 2) if os.path.exists(model_path) else None
        }
    
    return jsonify(status)

# Download a specific model
@app.route('/models/download/<model_name>')
def download_model(model_name):
    """Trigger download of a specific model"""
    if model_name not in AVAILABLE_MODELS:
        return jsonify({'success': False, 'error': f'Unknown model: {model_name}'})
    
    model_path = os.path.join(app.config['MODEL_FOLDER'], AVAILABLE_MODELS[model_name]['file'])
    
    if os.path.exists(model_path):
        return jsonify({'success': True, 'message': f'Model {model_name} is already downloaded'})
    
    url = MODEL_URLS.get(model_name)
    if not url:
        return jsonify({'success': False, 'error': f'No download URL for model: {model_name}'})
    
    try:
        gdown.download(url, model_path, quiet=False)
        return jsonify({'success': True, 'message': f'Model {model_name} downloaded successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error downloading model: {str(e)}'})

# Error handling
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Use environment variable for port if available (for deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)