# 🖼️ Vision Transformer (ViT) for CIFAR-10 Image Classification  

This project implements a **Vision Transformer (ViT)** for **image classification** on the **CIFAR-10 dataset**. 
The model is compared with other architectures, including **ResNet and CNN-MLP hybrids**, to evaluate the effectiveness of Transformers in computer vision tasks.  

---

## 🚀 **Features**  
✔ Preprocesses the **CIFAR-10 dataset**  
✔ Applies **data augmentation techniques** (random cropping, flipping, normalization)  
✔ Implements **Vision Transformer (ViT) from scratch** using a deep learning framework  
✔ Divides images into **patches** and applies **positional encoding**  
✔ Trains the ViT model using **multi-head self-attention & feed-forward layers**  
✔ Tunes **hyperparameters** (layers, attention heads, patch size, learning rate)  
✔ Compares performance with **CNN-based hybrid models & pretrained ResNet (transfer learning)**  
✔ Evaluates models using **accuracy, precision, recall, F1-score, and confusion matrix**  
✔ Plots **training & validation accuracy/loss curves**  
✔ Deploys the models for **classification of test images**  

---

## 🔧 **Requirements**  
- Python  
- TensorFlow/PyTorch  
- NumPy & Pandas  
- Matplotlib & Seaborn  
- OpenCV  

---

## 📊 **Results & Evaluation**  
The models were trained on the **CIFAR-10 dataset** and evaluated using:  
✔ **Classification Accuracy**  
✔ **Precision & Recall**  
✔ **F1-score**  
✔ **Confusion Matrix for misclassifications**  

A comparison was made between:  
1. **Vision Transformer (ViT) from scratch**  
2. **Hybrid CNN-MLP model**  
3. **Pretrained ResNet with transfer learning**  
