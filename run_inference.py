import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import json
import sys
import os

# Same network architecture as training
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x, record_activations=False):
        x = x.view(-1, 784)
        activations = {}
        
        x = self.fc1(x)
        x = self.relu(x)
        if record_activations:
            activations['layer1'] = x.detach().cpu().numpy()
        
        x = self.fc2(x)
        x = self.relu(x)
        if record_activations:
            activations['layer2'] = x.detach().cpu().numpy()
        
        x = self.fc3(x)
        x = self.relu(x)
        if record_activations:
            activations['layer3'] = x.detach().cpu().numpy()
        
        x = self.fc4(x)
        if record_activations:
            activations['output'] = x.detach().cpu().numpy()
        
        return x, activations if record_activations else x

def preprocess_image(image_path):
    """Convert user drawing to MNIST format (28x28 grayscale, normalized)"""
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to 28x28
    img = img.resize((28, 28), Image.LANCZOS)
    
    # Convert to numpy array
    img_array = np.array(img, dtype=np.float32)
    
    # Invert colors (MNIST is white digit on black background)
    img_array = 255 - img_array
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Apply MNIST normalization (mean=0.1307, std=0.3081)
    img_array = (img_array - 0.1307) / 0.3081
    
    # Convert to tensor
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
    
    return tensor

def run_inference(image_path, model_path='mnist_model.pth'):
    """Run inference on user drawing"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = MNISTNet().to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please run the training script first to generate the model.")
        sys.exit(1)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Preprocess image
    img_tensor = preprocess_image(image_path).to(device)
    
    # Run inference
    with torch.no_grad():
        output, activations = model(img_tensor, record_activations=True)
        predicted = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][predicted].item()
    
    print(f"Predicted digit: {predicted}")
    print(f"Confidence: {confidence:.2%}")
    
    # Create inference trace
    inference_trace = {
        "true_label": -1,  # Unknown for user drawings
        "predicted_label": predicted,
        "confidence": float(confidence),
        "layers": []
    }
    
    # Normalize activations to 0-1 range
    for layer_name, act_values in activations.items():
        act_values = act_values.flatten()
        # Normalize to 0-1
        act_min, act_max = act_values.min(), act_values.max()
        if act_max - act_min > 0:
            normalized = (act_values - act_min) / (act_max - act_min)
        else:
            normalized = act_values
        
        inference_trace["layers"].append({
            "name": layer_name,
            "activations": [float(v) for v in normalized]
        })
    
    # Save inference trace
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'inference_trace.json')
    with open(output_path, 'w') as f:
        json.dump(inference_trace, f, indent=2)
    
    print(f"Inference trace saved to: {output_path}")
    return predicted, confidence

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)
    
    run_inference(image_path)