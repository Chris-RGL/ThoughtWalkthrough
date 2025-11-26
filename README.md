# Neural Network Thought Process Visualizer

A Unity-based 3D visualization tool that shows how a neural network "thinks" when classifying handwritten digits. Draw a digit, watch it flow through the network layers, and see which neurons activate in real-time.

## Overview

This project combines Unity with PyTorch to create an interactive visualization of neural network inference. Users can:
- Draw digits (0-9) on a canvas
- Submit drawings for classification
- Watch an animated "orb" travel through the network layers
- See neurons light up based on activation strength
- Explore the 3D neural network structure with camera controls

## Features

- **Interactive Drawing Canvas**: Draw digits directly in Unity
- **Real-time Inference**: Python backend processes drawings using a trained MNIST model
- **3D Neural Network Visualization**: Neurons positioned using t-SNE dimensionality reduction
- **Animated Thought Process**: Visual representation of data flowing through the network
- **Camera Controls**: Follow the main orb and rotate/zoom the camera
- **Layer-by-layer Animation**: Watch activation propagate through each layer

## Requirements

### Unity
- Unity 2020.3 or later
- .NET Standard 2.1

### Python
- Python 3.7+
- PyTorch
- torchvision
- scikit-learn
- Pillow
- numpy

Install Python dependencies:
```bash
pip install torch torchvision scikit-learn pillow numpy
```

## Setup Instructions

### 1. Python Setup

**Train the Model** (First-time setup):
```bash
python nn_train_export.py
```
This will:
- Download MNIST dataset
- Train a 4-layer neural network
- Save the model as `mnist_model.pth`
- Generate `atlas_structure.json` (3D neuron positions)

**Move Files to Unity**:
- Copy `mnist_model.pth` to your Unity project's `Assets/` folder
- Copy `atlas_structure.json` to your Unity project's `Assets/` folder
- Keep the Python scripts (`nn_train_export.py` or `run_inference.py`) accessible

### 2. Unity Setup

**Scene Hierarchy**:
Create the following GameObjects:
```
- Canvas (UI Canvas)
  - DrawingPanel
    - DrawingCanvas (RawImage)
    - ClearButton (Button)
    - SubmitButton (Button)
    - InstructionText (Text)
    
- AtlasManager (Empty GameObject)
  - Attach: AtlasGenerator.cs
  
- ThoughtProcessor (Empty GameObject)
  - Attach: ThoughtProcess.cs
  
- DrawingController (Empty GameObject)
  - Attach: DrawingInterface.cs
```

**Prefabs Required**:
1. **NeuronPrefab**: A sphere with a material (represents inactive neurons)
2. **GlowingNeuronPrefab**: A sphere with emissive material + Point Light (represents active neurons)
3. **MainOrbPrefab**: A sphere with emissive material + Point Light (the traveling "thought")
4. **ChildOrbPrefab**: Smaller sphere with emissive material (activation particles)

### 3. Component Configuration

#### DrawingInterface.cs
In the Unity Inspector, set:
- **UI References**: Drag your UI elements
- **Canvas Size**: 280 (recommended)
- **Brush Size**: 15
- **Python Script Name**: Full path or filename (e.g., `nn_train_export.py` or `C:/Projects/nn_train_export.py`)
- **Model File Name**: `mnist_model.pth`
- **Python Executable Path**: Full path to your Python executable
  - Windows: `C:/Users/YourName/AppData/Local/Programs/Python/Python39/python.exe`
  - Mac/Linux: `/usr/bin/python3` or `/usr/local/bin/python3`

#### AtlasGenerator.cs
In the Unity Inspector, set:
- **Neuron Prefab**: Your neuron sphere prefab
- **Glowing Neuron Prefab**: Your glowing neuron prefab
- **Atlas Json Path**: `atlas_structure.json`
- **Neuron Scale**: 0.5 (adjust to preference)
- **Spawn Neurons Immediately**: Unchecked (layers spawn during animation)

#### ThoughtProcess.cs
In the Unity Inspector, set:
- **Atlas Generator**: Drag the AtlasManager GameObject
- **Main Orb Prefab**: Your main orb prefab
- **Child Orb Prefab**: Your child orb prefab
- **Inference Json Path**: `inference_trace.json`
- **Animation Settings**: Adjust speeds and thresholds as desired
- **Camera Settings**: Configure camera follow behavior

## Usage

1. **Run the Unity Scene**
2. **Draw a digit** (0-9) on the canvas using your mouse
3. **Click "Submit"** - The drawing is sent to Python for inference
4. **Wait for processing** - The instruction text will show "Ready! Press SPACE to visualize"
5. **Press SPACE** - Watch the animation:
   - Drawing canvas hides
   - Main orb travels through each layer
   - Neurons spawn as the orb reaches them
   - Child orbs fly to activated neurons
   - Neurons light up when activated
6. **Camera Controls during animation**:
   - Right-click + drag: Rotate camera
   - Mouse wheel: Zoom in/out
7. **Press R** to reset and draw again

## File Structure
```
YourUnityProject/
├── Assets/
│   ├── Scripts/
│   │   ├── DrawingInterface.cs
│   │   ├── ThoughtProcess.cs
│   │   └── AtlasGenerator.cs
│   ├── mnist_model.pth
│   ├── atlas_structure.json
│   ├── inference_trace.json (generated at runtime)
│   └── user_drawing.png (generated at runtime)
│
├── Python/ (or wherever you keep Python scripts)
│   ├── nn_train_export.py
│   └── run_inference.py (optional alternative)
```

## How It Works

1. **Drawing**: User draws on a 280x280 canvas in Unity
2. **Submission**: Drawing is saved as PNG to Assets folder
3. **Python Inference**: 
   - Unity calls Python script via command line
   - Image is preprocessed (resize to 28x28, normalize)
   - Model predicts digit and records layer activations
   - Results saved to `inference_trace.json`
4. **Visualization**:
   - Unity loads inference trace
   - Main orb moves through 3D space (layer by layer)
   - For each activated neuron (activation > threshold):
     - Child orb spawns and flies to that neuron
     - Neuron is replaced with glowing version
   - Process repeats for all 4 layers

## Network Architecture

- **Input**: 784 (28×28 flattened image)
- **Layer 1**: 256 neurons (ReLU activation)
- **Layer 2**: 128 neurons (ReLU activation)
- **Layer 3**: 64 neurons (ReLU activation)
- **Output**: 10 neurons (digit classes 0-9)

## Troubleshooting

### "Python exe missing" or "Script missing"
- Verify the absolute paths in the DrawingInterface inspector
- Test your Python path in command line: `C:/path/to/python.exe --version`

### "Model file not found"
- Ensure `mnist_model.pth` is in the Assets folder
- Run `nn_train_export.py` to generate the model

### Drawing appears inverted
- The current setup uses white brush on black background (MNIST standard)
- If results are poor, check the preprocessing in the Python script

### Python errors in Unity console
- Check the full error message in the Unity console
- Verify all Python dependencies are installed
- Test the Python script manually: `python nn_train_export.py path/to/test_image.png`

### Neurons not appearing
- Ensure `atlas_structure.json` exists and is valid
- Check that prefabs are assigned in AtlasGenerator
- Verify layer names match between JSON and inference trace

## Customization

### Change Activation Threshold
Modify `activationThreshold` in ThoughtProcess.cs (default: 0.5)

### Adjust Animation Speed
Change `mainOrbSpeed` and `childOrbSpeed` in ThoughtProcess.cs

### Modify Network Architecture
Edit the `MNISTNet` class in `nn_train_export.py`, then retrain and regenerate the atlas

### Change Neuron Positioning
The atlas uses t-SNE to position neurons based on their weights. Adjust parameters in the `generate_atlas()` function

## Credits

Combines Unity game engine with PyTorch deep learning framework to create an educational visualization of neural network inference.

## License

This project is provided as-is for educational purposes.
