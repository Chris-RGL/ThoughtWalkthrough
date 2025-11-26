import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.manifold import TSNE
import numpy as np
import json

# Simple Neural Network for MNIST
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

# Training function
def train_model(epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    model = MNISTNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    print("Training model...")
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
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    print("Training complete!")
    return model, test_loader

# Generate 3D atlas using t-SNE
def generate_atlas(model):
    print("Generating 3D atlas...")
    atlas_structure = {
        "layers": []
    }
    
    layer_configs = [
        ("layer1", 256, model.fc1.weight.data),
        ("layer2", 128, model.fc2.weight.data),
        ("layer3", 64, model.fc3.weight.data),
        ("output", 10, model.fc4.weight.data)
    ]
    
    z_offset = 0
    for layer_name, neuron_count, weights in layer_configs:
        print(f"Processing {layer_name} with {neuron_count} neurons...")
        
        # Use weight matrix as features for t-SNE
        weight_features = weights.cpu().numpy()
        
        # Apply t-SNE for 3D reduction
        if neuron_count > 3:
            tsne = TSNE(n_components=3, perplexity=min(30, neuron_count-1), random_state=42)
            coords_3d = tsne.fit_transform(weight_features)
        else:
            # For small layers, use simple spacing
            coords_3d = np.array([[i*10, 0, 0] for i in range(neuron_count)])
        
        # Normalize coordinates to reasonable scale
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
        
        z_offset += 50  # Space layers apart
    
    with open('atlas_structure.json', 'w') as f:
        json.dump(atlas_structure, f, indent=2)
    
    print("Atlas structure saved to atlas_structure.json")
    return atlas_structure

# Record inference trace
def record_inference(model, test_loader):
    print("Recording inference trace...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Get first test image
    data, target = next(iter(test_loader))
    data = data.to(device)
    
    with torch.no_grad():
        output, activations = model(data, record_activations=True)
        predicted = output.argmax(dim=1).item()
    
    print(f"Test image label: {target.item()}, Predicted: {predicted}")
    
    # Normalize activations to 0-1 range
    inference_trace = {
        "true_label": int(target.item()),
        "predicted_label": predicted,
        "layers": []
    }
    
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
    
    with open('inference_trace.json', 'w') as f:
        json.dump(inference_trace, f, indent=2)
    
    print("Inference trace saved to inference_trace.json")

# Main execution
if __name__ == "__main__":
    # Train model
    model, test_loader = train_model(epochs=5)
    
    # Generate atlas
    generate_atlas(model)
    
    # Record inference
    record_inference(model, test_loader)
    
    print("\n=== Export Complete ===")
    print("Files created:")
    print("  - atlas_structure.json (3D neuron positions)")
    print("  - inference_trace.json (activation values)")
    print("\nReady for Unity import!")
