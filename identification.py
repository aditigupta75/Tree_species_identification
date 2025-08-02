import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import json
import tempfile
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import io

# Force CPU usage
tf.config.set_visible_devices([], 'GPU')

class TreeSpeciesIdentifier:
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.model_loaded = False
    
    def load_model(self, model_file, encoder_file):
        """Load trained model and encoder"""
        try:
            # Load model
            self.model = tf.keras.models.load_model(model_file)
            
            # Load encoder
            with open(encoder_file, 'r') as f:
                encoder_data = json.load(f)
            
            self.label_encoder = LabelEncoder()
            self.label_encoder.classes_ = np.array(encoder_data['classes'])
            
            self.model_loaded = True
            
            return f"✅ Model loaded successfully!\n🌳 Available species: {', '.join(self.label_encoder.classes_)}"
            
        except Exception as e:
            self.model_loaded = False
            return f"❌ Failed to load model: {str(e)}"
    
    def preprocess_image(self, image, target_size=(224, 224)):
        """Preprocess image for prediction"""
        try:
            # Convert PIL image to numpy array
            img_array = np.array(image)
            
            # Convert to RGB if needed
            if len(img_array.shape) == 3 and img_array.shape[2] == 4:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img_resized = cv2.resize(img_array, target_size)
            
            # Normalize pixel values
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            return img_batch
            
        except Exception as e:
            raise Exception(f"Error preprocessing image: {str(e)}")
    
    def predict_species(self, image):
        """Predict tree species from image"""
        if not self.model_loaded:
            return "❌ Please load a model first!", None, None
        
        try:
            # Preprocess image
            img_batch = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(img_batch)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_species = self.label_encoder.classes_[predicted_class_idx]
            
            # Create results text
            result_text = f"🎯 Predicted Species: {predicted_species}\n"
            result_text += f"🔍 Confidence: {confidence:.4f} ({confidence*100:.2f}%)\n\n"
            result_text += "📊 All Predictions:\n"
            
            # Sort predictions by probability
            species_probs = list(zip(self.label_encoder.classes_, predictions[0]))
            species_probs.sort(key=lambda x: x[1], reverse=True)
            
            for species, prob in species_probs:
                result_text += f"• {species}: {prob:.4f} ({prob*100:.2f}%)\n"
            
            # Create probability bar chart
            plt.figure(figsize=(10, 6))
            species_names = [item[0] for item in species_probs]
            probabilities = [item[1] for item in species_probs]
            
            bars = plt.bar(species_names, probabilities, color='skyblue', edgecolor='navy', alpha=0.7)
            plt.title('Species Identification Probabilities', fontsize=16, fontweight='bold')
            plt.xlabel('Tree Species', fontsize=12)
            plt.ylabel('Probability', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Highlight the predicted species
            bars[0].set_color('lightcoral')
            bars[0].set_edgecolor('darkred')
            
            # Add percentage labels on bars
            for bar, prob in zip(bars, probabilities):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{prob*100:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot to bytes
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close()
            
            return result_text, Image.open(img_buffer), predicted_species
            
        except Exception as e:
            return f"❌ Prediction failed: {str(e)}", None, None

# Initialize identifier
identifier = TreeSpeciesIdentifier()

def load_model_files(model_file, encoder_file):
    """Load model and encoder files"""
    if model_file is None or encoder_file is None:
        return "❌ Please upload both model (.h5) and encoder (.json) files!"
    
    return identifier.load_model(model_file.name, encoder_file.name)

def identify_tree(image):
    """Identify tree species from uploaded image"""
    if image is None:
        return "❌ Please upload an image!", None
    
    result_text, plot_image, predicted_species = identifier.predict_species(image)
    
    return result_text, plot_image

# Create Gradio interface for identification
with gr.Blocks(title="Tree Species Identification", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🌲 Tree Species Classification - Identification Module")
    gr.Markdown("Load your trained model files and identify tree species from images with confidence scores and probability graphs.")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📂 Load Trained Model")
            model_file = gr.File(label="Upload Model File (.h5)", file_types=[".h5"])
            encoder_file = gr.File(label="Upload Encoder File (.json)", file_types=[".json"])
            load_btn = gr.Button("Load Model", variant="primary")
            model_status = gr.Textbox(label="Model Status", lines=3, interactive=False)
            
            gr.Markdown("## 🖼️ Upload Image for Identification")
            input_image = gr.Image(label="Upload Tree Image", type="pil")
            identify_btn = gr.Button("Identify Species", variant="secondary")
        
        with gr.Column(scale=2):
            gr.Markdown("## 🎯 Identification Results")
            result_text = gr.Textbox(label="Prediction Results", lines=10, interactive=False)
            probability_plot = gr.Image(label="Probability Distribution")
    
    # Event handlers
    load_btn.click(
        fn=load_model_files,
        inputs=[model_file, encoder_file],
        outputs=[model_status]
    )
    
    identify_btn.click(
        fn=identify_tree,
        inputs=[input_image],
        outputs=[result_text, probability_plot]
    )
    
    # Auto-identify when image is uploaded
    input_image.change(
        fn=identify_tree,
        inputs=[input_image],
        outputs=[result_text, probability_plot]
    )

if __name__ == "__main__":
    import socket
    
    # Get local IP address
    def get_local_ip():
        try:
            # Connect to a remote server to get local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "127.0.0.1"
    
    local_ip = get_local_ip()
    port = 7861
    
    print("🌲 Tree Species Classification - Identification Module")
    print("=" * 60)
    print("🚀 Starting web interface...")
    print(f"📍 Local access: http://localhost:{port}")
    print(f"🌐 Network access: http://{local_ip}:{port}")
    print("=" * 60)
    print("💡 You can access this from any device on your network!")
    print("🔧 Press Ctrl+C to stop the server")
    print("=" * 60)
    
    app.launch(
        server_name="0.0.0.0",  # Allow access from any IP
        server_port=port,       # Different port from training
        share=False,            # Set to True if you want a public link
        debug=True,
        show_api=False,         # Hide API docs
        quiet=True              # Reduce gradio startup messages
    )