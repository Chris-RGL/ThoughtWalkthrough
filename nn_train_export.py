import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import numpy as np
import json
import sys
import os
from PIL import Image

# --- 1. MODEL ARCHITECTURE (Must match exactly) ---
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

# --- 2. TRAINING LOGIC (Only runs if manually executed) ---
def train_and_export():
    print("Running in TRAINING mode...")
    epochs = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved to mnist_model.pth")
    
    # Generate Atlas
    generate_atlas(model)

def generate_atlas(model):
    print("Generating 3D atlas...")
    atlas_structure = { "layers": [] }
    
    layer_configs = [
        ("layer1", 256, model.fc1.weight.data),
        ("layer2", 128, model.fc2.weight.data),
        ("layer3", 64, model.fc3.weight.data),
        ("output", 10, model.fc4.weight.data)
    ]
    
    z_offset = 0
    for layer_name, neuron_count, weights in layer_configs:
        weight_features = weights.cpu().numpy()
        
        if neuron_count > 3:
            tsne = TSNE(n_components=3, perplexity=min(30, neuron_count-1), random_state=42)
            coords_3d = tsne.fit_transform(weight_features)
        else:
            coords_3d = np.array([[i*10, 0, 0] for i in range(neuron_count)])
        
        # Normalize
        coords_3d = (coords_3d - coords_3d.mean(axis=0)) / (coords_3d.std(axis=0) + 1e-8) * 20
        
        neurons = []
        for i in range(neuron_count):
            neurons.append({
                "id": i,
                "x": float(coords_3d[i, 0]),
                "y": float(coords_3d[i, 1]),
                "z": float(coords_3d[i, 2]) + z_offset
            })
        
        atlas_structure["layers"].append({
            "name": layer_name,
            "neuron_count": neuron_count,
            "z_offset": z_offset,
            "neurons": neurons
        })
        z_offset += 50
    
    with open('atlas_structure.json', 'w') as f:
        json.dump(atlas_structure, f, indent=2)

# --- 3. INFERENCE LOGIC (Runs when Unity calls script) ---
def predict_custom_image(image_path, model_path):
    print(f"Running in INFERENCE mode.")
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Model
    model = MNISTNet().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Process Image
    try:
        # Load image, convert to grayscale (L)
        img = Image.open(image_path).convert('L')
        
        # Resize to 28x28 (MNIST standard)
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy and normalize
        img_np = np.array(img)
        
        # NOTE: Your C# DrawingInterface uses White brush on Black background.
        # MNIST is White digits on Black background. No inversion needed.
        
        # Transform to Tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        img_tensor = transform(img).unsqueeze(0).to(device) # Add batch dimension
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return

    # 3. Run Inference
    with torch.no_grad():
        output, activations = model(img_tensor, record_activations=True)
        predicted = output.argmax(dim=1).item()

    print(f"Predicted Label: {predicted}")

    # 4. Save JSON Trace
    inference_trace = {
        "true_label": -1, # Unknown for user drawings
        "predicted_label": predicted,
        "layers": []
    }

    for layer_name, act_values in activations.items():
        act_values = act_values.flatten()
        
        # Normalize activations 0-1 for Unity visualization
        act_min, act_max = act_values.min(), act_values.max()
        if act_max - act_min > 0:
            normalized = (act_values - act_min) / (act_max - act_min)
        else:
            normalized = act_values
            
        inference_trace["layers"].append({
            "name": layer_name,
            "activations": [float(v) for v in normalized]
        })

    # Save to the same directory as the image (Unity Assets folder)
    output_path = os.path.join(os.path.dirname(image_path), 'inference_trace.json')
    
    with open(output_path, 'w') as f:
        json.dump(inference_trace, f, indent=2)
    
    print(f"Trace saved to: {output_path}")

# --- 4. MAIN ENTRY POINT ---
if __name__ == "__main__":
    # If arguments are passed (from Unity), run inference
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        
        # If Unity sends the model path as 2nd arg, use it
        mod_path = sys.argv[2] if len(sys.argv) > 2 else "mnist_model.pth"
        
        predict_custom_image(img_path, mod_path)
        
    # If no arguments, run training (Manual setup)
    else:
        train_and_export()