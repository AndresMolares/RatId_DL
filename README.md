# Who’s that rat? Setting the score for unmarked rat identification with Deep Learning

## Authors

Andres Molares-Ulloa, Ehsan Noshahri, Maria del Rocio Ortega-Femia, Alejandro Puente-Castro and Alvaro Rodriguez

## Citation



DOI: 


## Abstract
A major challenge in analysing laboratory experiments with unmarked animals is being able to tell which animal is which at any given moment and over time. We explore the effectiveness of Deep Learning techniques for identifying laboratory rats, by training and testing Convolutional Neural Networks and other models on a dataset of 1.44 million images of 16 rats under controlled conditions. Establishing the first baseline for unmarked animal identification in behavioural research.
We evaluate the scalability of these techniques and examine how image properties and learning parameters influence their performance. Our results demonstrate that the identification accuracy can exceed 90% for a moderate number of individuals without using additional knowledge such as heuristics or body part information. However, they also cast doubt on widely accepted claims in the scientific literature that Deep Learning techniques can reliably distinguish up to 100 unmarked animals in experimental settings.

# Questions?
If you have any questions, please feel free to contact us!

# User Manual

# 🐭 Deep Learning Pipeline for Unmarked Rat Identification  
### Dataset Preparation • TFRecord Generation • Model Training • Evaluation  

Code reference: *functions.py*

---

## Overview

This repository provides a complete workflow for training deep-learning models to identify unmarked laboratory rats from video frames. It includes utilities for TFRecord generation, dataset construction, model training with transfer learning, and evaluation at both image and sequence level.

The functions integrate with TensorFlow/Keras and support several state-of-the-art convolutional architectures (Xception, ResNet50, InceptionV3, MobileNetV2).

---

## Repository Structure

```
project/
│
├── functions.py      # Implementation of dataset, model, and evaluation utilities
├── Images/           # (User-provided) Raw dataset directory
└── tfRecords/        # Auto-generated TFRecords after preprocessing
```

---

## 🔄 Workflow Summary

1. Convert raw images into optimized TFRecord files  
2. Build training/validation/test datasets with uniform sampling  
3. Train a transfer-learning model (with optional data augmentation)  
4. Evaluate using image-level or sequence-level accuracy  

---

# 📌 Function Reference

---

## 1. `create_tfrecords(base_path)`

### Purpose
Converts a directory-based dataset into TFRecord format.

Expected input structure:

```
base_path/
    Images/
        train/Video01/Rat01/*.jpg
        test/Video01/Rat01/*.jpg
```

Output:

```
base_path/
    tfRecords/
        train/Video01/Rat01.tfrecord
        test/Video01/Rat01.tfrecord
```

### Description
- Reads all images for each rat and video.  
- Serializes them into TF Examples.  
- Creates one TFRecord per rat per video.

### Example
```python
create_tfrecords(base_path)
```

---

## 2. `create_datasets(ratas_selected, directory, n_imagenes_x_class_tr_val, n_imagenes_x_class_te, video_length=10000, frames=None)`

### Purpose
Builds **training**, **validation**, and **test** TensorFlow datasets from TFRecord files.

### Features
- Decodes and preprocesses images.  
- Assigns integer labels based on `ratas_selected`.  
- Uniform sampling of frames across videos.  
- Train/validation split based on video partitioning.  
- Test sampling optionally uses all frames (`frames` argument).

### Returns
```
training_dataset, test_dataset, val_dataset
```

### Example
```python
train_ds, test_ds, val_ds = create_datasets(
    ratas_selected=[0,1,2,3],
    directory=base_path,
    n_imagenes_x_class_tr_val=50,
    n_imagenes_x_class_te=100
)
```

---

## 3. `train_model(type_model, imageShape, ratas_selected, training_dataset, val_dataset, n_imagenes_x_class_tr_val, params, dataAugmentation=None)`

### Purpose
Trains a deep-learning classifier using transfer learning.

### Supported Backbones
- Xception  
- ResNet50  
- InceptionV3  
- MobileNetV2  

### Training Procedure
1. **Initial training**
   - Backbone frozen  
   - Dense head trained  
   - Early stopping applied  

2. **Fine-tuning**
   - Last 10% of backbone layers unfrozen  
   - Reduced learning rate  
   - Additional early stopping  

### Data Augmentation Options
- `"FLIP"`  
- `"ROTA"`  
- `"ZOOM"`  
- `"MIX"`  

### Example
```python
model = train_model(
    type_model="ResNet50",
    imageShape=(128,128),
    ratas_selected=[0,1,2,3],
    training_dataset=train_ds,
    val_dataset=val_ds,
    n_imagenes_x_class_tr_val=50,
    params=(0.001, [256]),
    dataAugmentation="MIX"
)
```

---

## 4. `test_model(model, test_dataset, frames=None, video_length=10000, batch_size=256, save_mode="metrics", output_file=None)`

### Purpose
Evaluates the trained network using image-level or sequence-level predictions.

### Modes

#### **Image-level (default)**
- Predicts one frame at a time  
- Returns accuracy and confusion matrix  
- Can save predictions or metrics to file  

#### **Sequence-level (`frames > 1`)**
- Groups frames into fixed-length sequences  
- Assigns a label via majority voting  
- Computes sequence-level accuracy  

### Example (image-level)
```python
acc, cm = test_model(
    model,
    test_dataset=test_ds,
    save_mode="metrics",
    output_file="results.txt"
)
```

### Example (sequence-level)
```python
acc, cm = test_model(
    model,
    test_dataset=test_ds,
    frames=16,
    save_mode="predictions",
    output_file="seq_preds.csv"
)
```

---

# 🔄 End-to-End Example

```python
# 1 — Create TFRecords
create_tfrecords(base_path)

# 2 — Build datasets
train_ds, test_ds, val_ds = create_datasets(
    ratas_selected=[0,1,2,3,4],
    directory=base_path,
    n_imagenes_x_class_tr_val=50,
    n_imagenes_x_class_te=100
)

# 3 — Train model
model = train_model(
    type_model="Xception",
    imageShape=(128,128),
    ratas_selected=[0,1,2,3,4],
    training_dataset=train_ds,
    val_dataset=val_ds,
    n_imagenes_x_class_tr_val=50,
    params=(0.0005, [512]),
    dataAugmentation="MIX"
)

# 4 — Evaluate
acc, cm = test_model(model, test_ds)
print("Accuracy:", acc)
print("Confusion Matrix:
", cm)
```

---

## Citation

If you use this code in academic work, please cite:

> “Dataset construction, model training, and evaluation were performed using the custom utilities provided in the repository (see *functions.py*).”

---

## Contributing

Pull requests, issues, and suggestions are welcome.

---

## License

Specify your license (e.g., MIT, Apache-2.0, GPL-3.0).

