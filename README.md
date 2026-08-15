# Multimodal Clinical AI Platform

An end-to-end clinical AI research prototype for analyzing chest X-rays and associated clinical text.

## Project Goals

This project aims to:

- Build and deploy a complete machine-learning inference system.
- Establish a ResNet-18 chest X-ray classification baseline.
- Compare conventional CNN representations with modern medical vision foundation models.
- Incorporate transformer-based clinical text representations.
- Investigate multimodal image-text fusion.
- Evaluate model performance, calibration, uncertainty, and inference efficiency.

## Planned Architecture

Chest X-ray → Vision Encoder ─┐
                             ├→ Multimodal Fusion → Prediction
Clinical Text → Transformer ─┘

The application will eventually include an API, user interface, persistent inference metadata, automated testing, containerization, and monitoring.

## Status

🚧 Under active development.
