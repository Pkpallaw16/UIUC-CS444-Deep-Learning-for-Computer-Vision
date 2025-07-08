
# Assignment 3: Self-supervised and Transfer Learning, Object Detection

## Part 1: Self-supervised and Transfer Learning

The first part of Assignment 3 is to train a neural network in a self-supervised manner, and then fine-tune it for a downstream task.  
Specifically, we first train our neural network (RotNet) to classify images based on their rotation degrees (0°, 90°, 180°, 270°).  
Then we fine-tune it on CIFAR-10 and ImageNette datasets for classification.  

We also compare:
- Fine-tuning only the late layers vs. entire model
- Pre-trained weights vs. training from scratch
- ResNet18 vs. ResNet50 as backbone
- Different dataset sizes (for extra credit)

### Training & Fine-tuning notebooks:
- Self-supervised Rotation task: `assignment3_part1/rotation_training.ipynb`
- Fine-tuning experiments: `assignment3_part1/fine_tuning.ipynb`

---

## Results

### 📋 CIFAR-10

#### Rotation Task
| Backbone  | Test Acc. | Loss |
|-----------|-----------|------|
| ResNet18  | 79.04%    | 0.527 |
| ResNet50  | 80.17%    | 0.511 |

#### Fine-tuning Late Layers
| Model            | Test Acc. | Loss |
|------------------|-----------|------|
| Pre-trained (ResNet18, layer4+fc) | 62.86% | 1.055 |
| Random Init (ResNet18, layer4+fc) | 45.90% | 1.525 |

#### Fine-tuning Entire Model
| Model            | Test Acc. | Loss |
|------------------|-----------|------|
| Pre-trained (ResNet18) | 83.46% | 0.491 |
| Random Init (ResNet18) | 82.36% | 0.521 |
| Pre-trained (ResNet50) | 84.10% | 0.473 |

#### Effect of Dataset Size (Extra Credit)
| #Examples/Category | RotNet + CIFAR10 | Supervised CIFAR10 |
|--------------------|-------------------|---------------------|
| 20                 | 24.34%           | 22.06%             |
| 100                | 37.29%           | 34.24%             |
| 500                | 55.14%           | 53.32%             |
| 1000               | 69.00%           | 64.92%             |
| 5000               | 82.85%           | 84.10%             |

### 📋 ImageNette
| Backbone  | Test Acc. | Loss |
|-----------|-----------|------|
| ResNet50  | 58.70%    | 0.970 |

---

## Part 2: Object Detection

We implemented the YOLO v1 loss function (Eq. 3 from [YOLO paper](https://pjreddie.com/media/files/papers/yolo_1.pdf)) and trained a detector on the PASCAL VOC 2007 dataset.

### Backbone Network
- Inspired by DetNet
- First 4 blocks: ResNet50 (pretrained on ImageNet)
- Added extra block to preserve resolution for detecting small objects

#### Code:
- YOLO loss: `yolo_loss.py`
- Backbone: `src/resnet_yolo.py`

---

## Results

| Metric | Value |
|-------|-------|
| Best mAP (on Kaggle) | 0.48466 |
| Team Name (Kaggle) | Ninja4 |
| Final Loss | Train: 1.925, Best Test: 2.97 |

### Notes:
- Initially, NMS threshold of 0.1 resulted in too many boxes; improved with threshold = 0.5.
- Video produced as extra credit:
  - Input: 500 frames from "SNL Digital Short - YOLO - SNL.mp4"
  - Predicted objects per frame
  - Output video: `video.mp4`

---

## 👥 Team
- **Name(s):** Pallaw Kumar, Neha Jain
- **NetID(s):** pallawk2, nehaj4 
- **Team Name (Kaggle):** Ninja4

---
