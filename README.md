# RecipeEmbeddings
# Transformer-based Cross-Modal Recipe Embeddings with Large Batch Training

This repository contains the implementation of the paper **"Transformer-based Cross-Modal Recipe Embeddings with Large Batch Training"** by Jing Yang, Junwen Chen, and Keiji Yanai. This project aims to reproduce the results of the paper and explore its proposed framework, **TNLBT (Transformer-based Network for Large Batch Training)**, which focuses on cross-modal recipe retrieval and image generation.

### Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Acknowledgments](#acknowledgments)
- [References](#references)

---

## Overview

Cross-modal recipe retrieval involves retrieving recipe texts from images and vice versa. This repository implements a **Transformer-based framework** for embedding learning in a cross-modal setting. 

The key components of this framework include:
- **Transformer encoders** for both image and text modalities.
- **Vision Transformer (ViT)** for image encoding, with the option to use **CLIP-ViT**.
- **Large batch training** for improved contrastive learning.
- **Self-supervised learning loss** to enhance cross-modal embedding learning.

This implementation uses **PyTorch** and **Huggingface's Transformers** library for model building and training.

### Key Features
- Efficient exploitation of cross-modal relationships between recipe images and texts.
- High performance in **cross-modal recipe retrieval** and **image generation** tasks.
- Integration with the **Recipe1M** dataset for benchmarking.
- Support for large batch training, which has been shown to improve contrastive learning results.

---

## Requirements

To run this project, ensure you have the following dependencies installed:

- Docker
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- Huggingface Transformers
- torchvision
- Recipe1M dataset (or a custom dataset for cross-modal retrieval)
  
---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your_username/tnlbt-recipe-retrieval.git
cd tnlbt-recipe-retrieval
```

### Step 2: Build the Docker Image

We use Docker for setting up a reproducible environment. The `Dockerfile` provided in this repository uses the **PyTorch 2.4.1 with CUDA 11.8** base image for GPU support.

```bash
docker build -t tnlbt-recipe-retrieval .
```

### Step 3: Run the Docker Container

Once the image is built, you can start a container to interact with the project.

```bash
docker run -it --gpus all -p 8000:8000 tnlbt-recipe-retrieval bash
```

This will start a container with GPU access and expose port 8000 for any web-based applications (if needed).

---

## Usage

### Vision Transformer (ViT) and CLIP-ViT

This implementation supports both the original **ViT** and **CLIP-ViT** as image encoders. By default, **CLIP-ViT** is used, but you can switch to **ViT-B** as needed.

The models are implemented using the **Huggingface Transformers** library, ensuring compatibility with their pre-trained models and fine-tuning capabilities.

### Training and Inference Scripts

Training and evaluation scripts for both cross-modal retrieval and image generation tasks are included. You can configure the parameters and model settings through a configuration file.

---

## Model

The model consists of two encoders:
- **Text Encoder**: A Transformer-based encoder for text data (recipes).
- **Image Encoder**: A Vision Transformer-based encoder for image data.

Additionally, the framework incorporates **self-supervised learning** loss functions and **contrastive learning** to promote cross-modal embedding learning.

---

## Dataset

The **Recipe1M** dataset is used for training and evaluation in this implementation. This dataset contains over 1 million recipes and associated images. You can download the dataset from [here](http://pic2recipe.csail.mit.edu/).

After downloading, extract the dataset and ensure it is properly structured before training.

---

## Training

### Step 1: Set Up the Dataset

Ensure the Recipe1M dataset is correctly formatted and available in the Docker container. You can use a Docker volume or copy the dataset directly into the container.

### Step 2: Training the Model

To start training, use the following command:

```bash
python train.py --config config.yaml
```

The `config.yaml` file contains the hyperparameters and training settings such as batch size, learning rate, model architecture, etc.

### Step 3: Large Batch Training

For large batch training, make sure to adjust the batch size in the `config.yaml` file according to the available GPU memory. Larger batch sizes can significantly improve the performance of contrastive learning.

---

## Evaluation

To evaluate the model on the Recipe1M benchmark, run:

```bash
python evaluate.py --config config.yaml --checkpoint path_to_model_checkpoint
```

This will output the retrieval accuracy and image generation results based on the trained model.

---

## Acknowledgments

This implementation is based on the paper:

> Jing Yang, Junwen Chen, and Keiji Yanai. "Transformer-based Cross-Modal Recipe Embeddings with Large Batch Training." The University of Electro-Communications, Tokyo, Japan.

Special thanks to the developers of PyTorch, Huggingface Transformers, and the Recipe1M dataset.

---

## References

1. Jing Yang, Junwen Chen, and Keiji Yanai. "Transformer-based Cross-Modal Recipe Embeddings with Large Batch Training." The University of Electro-Communications, Tokyo, Japan.
2. [PyTorch](https://pytorch.org/)
3. [Huggingface Transformers](https://huggingface.co/transformers/)
4. [Recipe1M Dataset](http://pic2recipe.csail.mit.edu/)