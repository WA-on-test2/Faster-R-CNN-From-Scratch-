# Faster R-CNN Object Detection Framework

 Implementation of Faster R-CNN for object detection using PyTorch. 

---

## 📋 Table of Contents

- [Features](#features)
- [Project Architecture](#project-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Setup](#dataset-setup)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)

---


## 🏗️ Project Architecture

```
FasterRCNN-Refactored/
│
├── config/                      # Configuration files
│   ├── __init__.py
│   ├── base_config.py          # Config loader
│   └── voc_dataset.yaml        # VOC dataset configuration
│
├── core/                        # Core detection components (MODEL)
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── detector.py         # Main detection model
│   │   ├── region_proposal.py  # Region Proposal Network
│   │   └── roi_classifier.py   # ROI classification head
│   │
│   └── layers/
│       ├── __init__.py
│       └── anchor_generator.py # Anchor box generation
│
├── data/                        # Data handling
│   ├── __init__.py
│   └── loaders/
│       ├── __init__.py
│       └── voc_loader.py       # VOC dataset loader
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── bbox_operations.py      # Bounding box operations
│   ├── common.py               # Common utilities
│   ├── evaluation_metrics.py  # mAP calculation
│   └── visualization.py        # Drawing utilities
│
├── controllers/                 # Business logic (CONTROLLER)
│   ├── __init__.py
│   ├── trainer.py              # Training controller
│   └── evaluator.py            # Evaluation controller
│
├── scripts/                     # Entry points
│   ├── train_model.py          # Training script
│   ├── evaluate_model.py       # Evaluation script
│   └── run_inference.py        # Inference script
│
├── VOC2007/                     # Training dataset (you provide)
├── VOC2007-test/                # Test dataset (you provide)
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 50GB+ free disk space (for datasets)

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd FasterRCNN-Refactored
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n object_detection python=3.8
conda activate object_detection

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```txt
torch==1.11.0
torchvision==0.12.0
numpy==1.23.5
opencv-python==4.8.0.74
PyYAML==6.0.1
tqdm==4.65.0
matplotlib==3.7.2
Pillow>=9.0.0
```

---

## 🚀 Quick Start

### 1. Download and Setup Dataset

```bash
# Download VOC 2007 trainval
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
tar -xvf VOCtrainval_06-Nov-2007.tar
mv VOCdevkit/VOC2007 ./VOC2007

# Download VOC 2007 test
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
tar -xvf VOCtest_06-Nov-2007.tar
mv VOCdevkit/VOC2007 ./VOC2007-test

# Clean up
rm -rf VOCdevkit *.tar
```

### 2. Verify Dataset Structure

Your directory should look like:
```
FasterRCNN-Refactored/
├── VOC2007/
│   ├── JPEGImages/      # Training images
│   └── Annotations/      # Training annotations (XML)
├── VOC2007-test/
│   ├── JPEGImages/      # Test images
│   └── Annotations/      # Test annotations (XML)
└── ...
```

### 3. Train the Model

```bash
python -m scripts.train_model --config config/voc_dataset.yaml
```

### 4. Evaluate the Model

```bash
python -m scripts.evaluate_model --config config/voc_dataset.yaml
```

### 5. Run Inference (Generate Visualizations)

```bash
python -m scripts.run_inference --config config/voc_dataset.yaml --num_samples 20
```

---

## 📊 Dataset Setup

### PASCAL VOC Format

The framework expects datasets in PASCAL VOC format:

```
Dataset/
├── JPEGImages/          # Image files (.jpg)
└── Annotations/         # XML annotation files
```

**XML Annotation Example**:
```xml
<annotation>
    <filename>000001.jpg</filename>
    <size>
        <width>500</width>
        <height>375</height>
    </size>
    <object>
        <name>dog</name>
        <bndbox>
            <xmin>48</xmin>
            <ymin>240</ymin>
            <xmax>195</xmax>
            <ymax>371</ymax>
        </bndbox>
    </object>
</annotation>
```

### Using Custom Datasets

To use your own dataset:

1. **Convert to VOC format** or modify `data/loaders/voc_loader.py`
2. **Update class names** in `voc_loader.py`:
   ```python
   class_names = [
       'your_class_1', 'your_class_2', 'your_class_3'
   ]
   ```
3. **Update config file** `config/your_dataset.yaml`:
   ```yaml
   dataset_params:
     im_train_path: 'YourDataset/JPEGImages'
     ann_train_path: 'YourDataset/Annotations'
     num_classes: <num_classes + 1>  # +1 for background
   ```

---

## 🎓 Training

### Basic Training

```bash
python -m scripts.train_model --config config/voc_dataset.yaml
```

### Training Options

Modify `config/voc_dataset.yaml` to customize training:

```yaml
train_params:
  task_name: 'my_experiment'      # Output directory name
  seed: 1111                       # Random seed
  num_epochs: 20                   # Number of epochs
  lr: 0.001                        # Learning rate
  lr_steps: [12, 16]              # LR decay at these epochs
  acc_steps: 1                     # Gradient accumulation steps
  ckpt_name: 'model.pth'          # Checkpoint filename
```

### Training Output

During training, you'll see:
```
Epoch 0/20
100%|████████████| 2501/2501 [15:23<00:00, 2.71it/s]
Epoch 0 completed
RPN Cls: 0.0234 | RPN Loc: 0.0156 | ROI Cls: 0.3421 | ROI Loc: 0.0892
```

**Outputs saved to**: `<task_name>/`
- Model checkpoint: `<task_name>/<ckpt_name>`

### Resume Training

To resume training from a checkpoint:

```python
# In scripts/train_model.py, add before trainer.train():
checkpoint_path = 'detection_output/model_checkpoint.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f'Resumed from {checkpoint_path}')
```

---

## 📈 Evaluation

### Basic Evaluation

```bash
python -m scripts.evaluate_model --config config/voc_dataset.yaml
```

### Evaluation Output

```
=== Evaluation Results ===
Class-wise Average Precisions:
  aeroplane: 0.7123
  bicycle: 0.7834
  bird: 0.6945
  ...
  tvmonitor: 0.7012

Mean Average Precision: 0.6842
```

### Understanding mAP

- **mAP (mean Average Precision)**: Average of AP across all classes
- **AP**: Area under the precision-recall curve for each class
- **Higher is better**: mAP ranges from 0.0 to 1.0

---

## 🎨 Inference

### Generate Visualizations

```bash
python -m scripts.run_inference \
    --config config/voc_dataset.yaml \
    --num_samples 20 \
    --output_dir inference_results
```

### Arguments

- `--config`: Path to configuration file
- `--num_samples`: Number of random images to visualize (default: 10)
- `--output_dir`: Output directory for images (default: inference_samples)

### Output

For each sample, two images are generated:
- `ground_truth_<n>.png`: Ground truth annotations (green boxes)
- `prediction_<n>.jpg`: Model predictions (red boxes with scores)

### Adjusting Confidence Threshold

To change the detection confidence threshold:

```python
# In scripts/run_inference.py
model.classification_head.min_confidence_threshold = 0.5  # Lower = more detections
```

---

## ⚙️ Configuration

### Configuration File Structure

```yaml
dataset_params:
  im_train_path: 'VOC2007/JPEGImages'
  ann_train_path: 'VOC2007/Annotations'
  im_test_path: 'VOC2007-test/JPEGImages'
  ann_test_path: 'VOC2007-test/Annotations'
  num_classes: 21                    # Including background

model_params:
  # Image preprocessing
  min_im_size: 600                   # Minimum image dimension
  max_im_size: 1000                  # Maximum image dimension
  
  # Backbone
  backbone_out_channels: 512         # VGG16 output channels
  fc_inner_dim: 1024                 # FC layer dimensions
  
  # Anchor generation
  scales: [128, 256, 512]            # Anchor sizes in pixels
  aspect_ratios: [0.5, 1, 2]         # Height/width ratios
  
  # RPN parameters
  rpn_bg_threshold: 0.3              # IoU threshold for background
  rpn_fg_threshold: 0.7              # IoU threshold for foreground
  rpn_nms_threshold: 0.7             # NMS IoU threshold
  rpn_train_prenms_topk: 12000       # Pre-NMS proposals (train)
  rpn_test_prenms_topk: 6000         # Pre-NMS proposals (test)
  rpn_train_topk: 2000               # Post-NMS proposals (train)
  rpn_test_topk: 300                 # Post-NMS proposals (test)
  rpn_batch_size: 256                # Anchors per image
  rpn_pos_fraction: 0.5              # Positive anchor fraction
  
  # ROI Head parameters
  roi_iou_threshold: 0.5             # IoU for positive proposals
  roi_low_bg_iou: 0.0                # Minimum IoU for background
  roi_pool_size: 7                   # ROI pooling output size
  roi_nms_threshold: 0.3             # Final NMS threshold
  roi_topk_detections: 100           # Max detections per image
  roi_score_threshold: 0.05          # Minimum confidence score
  roi_batch_size: 128                # Proposals per image
  roi_pos_fraction: 0.25             # Positive proposal fraction

train_params:
  task_name: 'detection_output'      # Experiment name
  seed: 1111                         # Random seed
  acc_steps: 1                       # Gradient accumulation
  num_epochs: 20                     # Training epochs
  lr_steps: [12, 16]                 # Learning rate decay steps
  lr: 0.001                          # Initial learning rate
  ckpt_name: 'model_checkpoint.pth'  # Checkpoint filename
```

---

## 📁 Project Structure Details

### Core Components

**`core/models/detector.py`**
- Main detection model combining all components
- Handles image preprocessing and normalization
- Coordinates backbone, RPN, and ROI head

**`core/models/region_proposal.py`**
- Generates region proposals from feature maps
- Implements anchor generation and matching
- Handles RPN training and inference

**`core/models/roi_classifier.py`**
- Classifies and refines region proposals
- Implements ROI pooling and fully connected layers
- Outputs final detections with class labels

### Utility Modules

**`utils/bbox_operations.py`**
- IoU calculation
- Bounding box encoding/decoding
- Coordinate transformations

**`utils/evaluation_metrics.py`**
- Mean Average Precision (mAP) calculation
- Precision-recall curve computation
- Per-class AP metrics

**`utils/visualization.py`**
- Drawing bounding boxes on images
- Adding labels and confidence scores
- Creating comparison visualizations

### Controllers

**`controllers/trainer.py`**
- Training loop implementation
- Loss computation and backpropagation
- Checkpoint saving and learning rate scheduling

**`controllers/evaluator.py`**
- Model evaluation on test set
- Metric computation and reporting
- Result aggregation

---

## 🔄 Customization

### 1. Using a Different Backbone

Edit `core/models/detector.py`:

```python
# Replace VGG16 with ResNet50
import torchvision.models as models

# In ObjectDetectionModel.__init__()
resnet = models.resnet50(pretrained=True)
self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

# Update config
model_params:
  backbone_out_channels: 2048  # ResNet50 output channels
```

### 2. Adding New Dataset Support

Create `data/loaders/coco_loader.py`:

```python
class COCODatasetLoader(Dataset):
    def __init__(self, split, data_dir, ann_file):
        # Load COCO annotations
        # Implement __getitem__ to return:
        #   - image_tensor (C, H, W)
        #   - targets dict with 'bboxes' and 'labels'
        #   - file_path
        pass
```

### 3. Hard Negative Mining

To enable hard negative mining, modify config:

```yaml
model_params:
  roi_low_bg_iou: 0.1  # Ignore proposals with IoU < 0.1
```

### 4. Gradient Accumulation

For simulating larger batch sizes:

```yaml
train_params:
  acc_steps: 4  # Accumulate gradients over 4 steps
```

### 5. Custom Loss Weights

Edit `controllers/trainer.py`:

```python
# In train_epoch()
rpn_loss = (2.0 * rpn_out['rpn_classification_loss'] + 
            1.0 * rpn_out['rpn_localization_loss'])
roi_loss = (1.0 * roi_out['frcnn_classification_loss'] + 
            2.0 * roi_out['frcnn_localization_loss'])
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Out of Memory Error

**Symptoms**: CUDA out of memory during training

**Solutions**:
```yaml
# Reduce batch size components
model_params:
  rpn_batch_size: 128        # Was 256
  roi_batch_size: 64         # Was 128
  rpn_train_prenms_topk: 6000  # Was 12000
```

#### 2. No Objects Detected

**Symptoms**: Model outputs zero detections

**Solutions**:
```python
# Lower confidence threshold
model.classification_head.min_confidence_threshold = 0.01

# Check training loss - should decrease over epochs
# Verify dataset annotations are correct
```

#### 3. Training Loss Not Decreasing

**Symptoms**: Losses remain high or increase

**Solutions**:
- Check learning rate (try 0.0001 instead of 0.001)
- Verify annotations are correct
- Ensure dataset paths are valid
- Check for NaN values in loss

#### 4. Import Errors

**Symptoms**: ModuleNotFoundError

**Solutions**:
```bash
# Ensure you're in project root
cd FasterRCNN-Refactored

# Use python -m to run scripts
python -m scripts.train_model  # Correct
python scripts/train_model.py  # May fail

# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### 5. CUDA Errors

**Symptoms**: CUDA-related errors during training

**Solutions**:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU training if needed
# The code automatically detects and uses CPU if CUDA unavailable
```

---

## ⚡ Performance Tips

### 1. Speed Optimization

```yaml
# Use fewer anchors
model_params:
  scales: [128, 256]           # Instead of [128, 256, 512]
  aspect_ratios: [1, 2]        # Instead of [0.5, 1, 2]

# Reduce proposals
model_params:
  rpn_train_topk: 1000         # Instead of 2000
  rpn_test_topk: 150           # Instead of 300
```

### 2. Accuracy Improvement

```yaml
# Increase training time
train_params:
  num_epochs: 30               # Instead of 20
  lr_steps: [18, 24]           # Adjust decay schedule

# Use larger FC dimensions
model_params:
  fc_inner_dim: 4096           # Instead of 1024 (slower but better)
```

### 3. Multi-GPU Training

For multi-GPU support, modify `scripts/train_model.py`:

```python
# Wrap model with DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
```

### 4. Mixed Precision Training

For faster training with less memory:

```python
from torch.cuda.amp import autocast, GradScaler

# In trainer.py
scaler = GradScaler()

with autocast():
    rpn_out, roi_out = self.model(img_batch, target_batch)
    total_loss = ...

scaler.scale(total_loss).backward()
scaler.step(self.optimizer)
scaler.update()
```

---

## 📊 Monitoring Training

### TensorBoard Integration

Add to `controllers/trainer.py`:

```python
from torch.utils.tensorboard import SummaryWriter

class ModelTrainer:
    def __init__(self, ...):
        self.writer = SummaryWriter(log_dir=f'runs/{task_name}')
    
    def train_epoch(self):
        # After computing losses
        self.writer.add_scalar('Loss/RPN_cls', rpn_cls_loss, epoch)
        self.writer.add_scalar('Loss/RPN_loc', rpn_loc_loss, epoch)
        # ... add more metrics

# Run tensorboard
# tensorboard --logdir=runs
```

### Logging

All scripts print progress to console. Redirect to file:

```bash
python -m scripts.train_model --config config/voc_dataset.yaml 2>&1 | tee training.log
```

---

## 📚 Advanced Usage

### Batch Inference

Process multiple images efficiently:

```python
from data.loaders.voc_loader import VOCDatasetLoader
from torch.utils.data import DataLoader

test_dataset = VOCDatasetLoader('test', 'path/to/images', 'path/to/annotations')
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model.eval()
all_predictions = []

for img_batch, _, file_paths in test_loader:
    with torch.no_grad():
        _, detections = model(img_batch.to(device), None)
    all_predictions.append(detections)
```

### Export to ONNX

Export model for deployment:

```python
import torch.onnx

dummy_input = torch.randn(1, 3, 600, 800).to(device)
model.eval()

torch.onnx.export(
    model,
    dummy_input,
    "detector_model.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['boxes', 'labels', 'scores']
)
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{ren2015faster,
  title={Faster R-CNN: Towards real-time object detection with region proposal networks},
  author={Ren, Shaoqing and He, Kaiming and Girshick, Ross and Sun, Jian},
  journal={Advances in neural information processing systems},
  volume={28},
  year={2015}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🆘 Support

For issues or questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Search existing issues on GitHub
3. Create a new issue with detailed description

---

## 🎯 Next Steps

After getting familiar with the basics:

1. **Experiment with hyperparameters** in the config file
2. **Try different backbones** (ResNet, EfficientNet)
3. **Add your own dataset**
4. **Implement feature pyramid networks (FPN)**
5. **Add test-time augmentation**

