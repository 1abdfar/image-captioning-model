# Image Captioning with Attention 👁️ 📝

A PyTorch-based Deep Learning pipeline that automatically generates descriptive natural language captions for images. The architecture leverages a Convolutional Neural Network (CNN) as the image encoder and a Recurrent Neural Network (RNN) combined with an Attention Mechanism as the decoder.

![Sample Image Captioning Output](path/to/your/sample_demo_image.gif_or_png)

## Architecture Overview

* **Encoder (ResNet-50):** Extracts high-level spatial feature maps from input images. The fully connected layers are stripped off, allowing the network to retain spatial dimensions for the attention mechanism.
* **Attention Mechanism:** Allows the decoder to dynamically "focus" on specific regions of the image while generating each word.
* **Decoder (LSTM):** An autoregressive language model that takes the attended image features and previously generated words to predict the next word in the sequence.

![Model Architecture Diagram](path/to/your/architecture_diagram.png)

## Dataset

This project is built to accept image-caption pairs (e.g., MS COCO dataset format). 
* Images should be stored in a flat directory (`data/images/`).
* Captions should be in a JSON file (`data/captions.json`) mapping image filenames to lists of string captions.

## Repository Structure

```text
├── dataset.py      # Custom PyTorch Dataset and Vocabulary builder
├── model.py        # CNN Encoder, Attention layer, and LSTM Decoder classes
├── train.py        # Training loop, data loading, and optimization logic
├── inference.py    # Autoregressive generation script and attention visualization
└── README.md