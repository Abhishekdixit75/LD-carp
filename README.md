# Deep Learning Practical — Practice Problem List (2026–27)

This repository contains the complete implementation of all 25 Deep Learning practical examination problems. Each experiment is implemented as an independent, standalone Python script named clearly by its practical number and topic.

---

## 📦 Required Dependencies & Installation

All experiments run on standard Python (Python 3.9 – 3.12).

### 1. Install All Dependencies (Single Command)
Run the following command in PowerShell or Command Prompt:

```powershell
pip install tensorflow numpy scikit-learn pandas matplotlib
```

### 2. Upgrade to Latest Versions (Optional)
If you already have older versions installed, upgrade them with:

```powershell
pip install --upgrade tensorflow numpy scikit-learn pandas matplotlib
```

### 3. Verify Installation
To confirm all required packages are installed correctly, run:

```powershell
python -c "import tensorflow as tf, numpy, sklearn, pandas, matplotlib; print('TensorFlow Version:', tf.__version__); print('All dependencies installed successfully!')"
```

---

## 🚀 How to Execute Any Practical

1. Open PowerShell or Command Prompt inside this folder:
   ```powershell
   cd "c:\Users\Somil\OneDrive\Desktop\Dl practicals"
   ```

2. Run the desired practical file directly using `python`:
   ```powershell
   python practical_01_tensorflow_perceptron.py
   python practical_04_transfer_learning_cats_vs_dogs.py
   python practical_14_binary_classification_breast_cancer.py
   ```

---

## 📋 Complete Practical Problem List & File Index

| Sr. No. | Practical Examination Problem Statement | Suggested Dataset | Script File Name | Execution Command |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **TensorFlow Fundamentals and Perceptron**: Perform basic tensor operations in TensorFlow, then build a single-layer Perceptron and an MLP for handwritten digit recognition and compare their accuracy. | MNIST | `practical_01_tensorflow_perceptron.py` | `python practical_01_tensorflow_perceptron.py` |
| **2** | **Regression Using Keras**: Build a house price prediction model trained with gradient descent, and compare versions with L1 regularization, L2 regularization and dropout to study overfitting. | California Housing | `practical_02_regression_california_housing.py` | `python practical_02_regression_california_housing.py` |
| **3** | **CNN-Based Image Classification**: Build a CNN to classify images into 10 classes: preprocess data, set training parameters, add dropout, save and reload the model, and evaluate it. | CIFAR-10 | `practical_03_cnn_image_classification.py` | `python practical_03_cnn_image_classification.py` |
| **4** | **Transfer Learning for Image Classification**: Use a pretrained CNN (e.g. VGG16, MobileNetV2) to classify cat and dog images, and compare feature extraction with fine-tuning. | Cats vs. Dogs | `practical_04_transfer_learning_cats_vs_dogs.py` | `python practical_04_transfer_learning_cats_vs_dogs.py` |
| **5** | **Sentiment Analysis Using RNN, LSTM and GRU**: Classify movie reviews as positive or negative with RNN, LSTM and GRU models, and compare accuracy, training time and handling of long reviews. | IMDB Movie Reviews | `practical_05_sentiment_rnn_lstm_gru.py` | `python practical_05_sentiment_rnn_lstm_gru.py` |
| **6** | **Autoencoder Implementation**: Build an autoencoder that compresses and reconstructs digit images, show original vs reconstructed images and report reconstruction error. | MNIST | `practical_06_autoencoder_mnist.py` | `python practical_06_autoencoder_mnist.py` |
| **7** | **CNN Optimization Strategies**: Train a CNN with Xavier vs He initialization, different learning-rate schedules and optimizers (SGD, Adam, RMSprop), and compare convergence and accuracy. | CIFAR-10 / MNIST | `practical_07_cnn_optimization_strategies.py` | `python practical_07_cnn_optimization_strategies.py` |
| **8** | **ResNet50 Transfer Learning and Image Captioning**: Extract image features with pretrained ResNet50 and build a CNN + RNN (LSTM) model that generates captions for images. | Flickr8k | `practical_08_image_captioning_resnet_lstm.py` | `python practical_08_image_captioning_resnet_lstm.py` |
| **9** | **Deep Learning-Based Recommendation System**: Build an embedding-based model that recommends products from user–product interaction data, and evaluate it with Precision@K or Recall@K. | Retailrocket Ecommerce Dataset | `practical_09_recommendation_system.py` | `python practical_09_recommendation_system.py` |
| **10** | **Sentiment Analysis on Network Graphs Using RNN**: Classify post sentiment with an RNN/LSTM, build a user interaction graph (e.g. @mentions), and visualize sentiment across the network. | Sentiment140 / Social Network Dataset | `practical_10_sentiment_network_graphs.py` | `python practical_10_sentiment_network_graphs.py` |
| **11** | **Generative Adversarial Network (GAN)**: Build a GAN that generates synthetic images, plot generator and discriminator losses, and show generated samples during training. | MNIST / Fashion-MNIST | `practical_11_gan_mnist.py` | `python practical_11_gan_mnist.py` |
| **12** | **Transformer-Based Text Classification**: Build a Transformer (self-attention) model to classify news articles into 4 categories, and compare its accuracy and training time with an LSTM. | AG News | `practical_12_transformer_text_classification.py` | `python practical_12_transformer_text_classification.py` |
| **13** | **Multilayer Perceptron for Classification**: Build MLPs with different numbers of hidden layers, neurons and activation functions (sigmoid, tanh, ReLU), and compare classification accuracy. | Fashion-MNIST | `practical_13_mlp_fashion_mnist.py` | `python practical_13_mlp_fashion_mnist.py` |
| **14** | **Binary Classification Using Deep Neural Networks**: Build a feedforward network to classify tumours as malignant or benign, comparing activations and optimizers with binary cross-entropy loss. | Breast Cancer Wisconsin | `practical_14_binary_classification_breast_cancer.py` | `python practical_14_binary_classification_breast_cancer.py` |
| **15** | **Multiclass Classification Using Softmax**: Build a neural network with a softmax output and categorical cross-entropy to classify wines into 3 classes, and report accuracy, precision, recall and a confusion matrix. | Wine Dataset | `practical_15_multiclass_softmax_wine.py` | `python practical_15_multiclass_softmax_wine.py` |
| **16** | **Time-Series Forecasting Using LSTM**: Build an LSTM that forecasts future temperature from past readings, and compare predicted vs actual values using MAE and RMSE. | Jena Climate | `practical_16_time_series_lstm_forecasting.py` | `python practical_16_time_series_lstm_forecasting.py` |
| **17** | **Image Denoising Using Autoencoders**: Add noise to images and train a denoising autoencoder to restore them, measuring quality with MSE or PSNR. | MNIST / Fashion-MNIST | `practical_17_image_denoising_autoencoder.py` | `python practical_17_image_denoising_autoencoder.py` |
| **18** | **Image Classification Using Data Augmentation**: Train a CNN with augmentation (rotation, flip, zoom, shift), and compare accuracy and overfitting with and without augmentation. | Cats vs. Dogs | `practical_18_data_augmentation_cnn.py` | `python practical_18_data_augmentation_cnn.py` |
| **19** | **Object Detection Using Transfer Learning**: Use a pretrained object detection model to locate and label objects in images, drawing bounding boxes and evaluating with IoU or mAP. | Pascal VOC 2007 | `practical_19_object_detection_transfer_learning.py` | `python practical_19_object_detection_transfer_learning.py` |
| **20** | **Text Generation Using LSTM**: Train a character- or word-level LSTM language model to generate text, and experiment with sequence length, embedding size and sampling temperature. | Shakespeare Text Dataset | `practical_20_text_generation_lstm.py` | `python practical_20_text_generation_lstm.py` |
| **21** | **Named Entity Recognition Using Deep Learning**: Build a sequence-labeling model (e.g. BiLSTM) that tags persons, organizations and locations in sentences, and report precision, recall and F1. | CoNLL-2003 | `practical_21_ner_bilstm.py` | `python practical_21_ner_bilstm.py` |
| **22** | **Attention-Based Neural Machine Translation**: Build an encoder–decoder model with attention that translates English sentences into French, and evaluate translations with BLEU. | English–French (ManyThings) | `practical_22_neural_machine_translation_attention.py` | `python practical_22_neural_machine_translation_attention.py` |
| **23** | **Variational Autoencoder (VAE)**: Build a VAE that learns a latent space of digit images, visualize the latent space, and generate new digits by sampling from it. | MNIST | `practical_23_vae_mnist.py` | `python practical_23_vae_mnist.py` |
| **24** | **Deep Learning for Anomaly Detection**: Train an autoencoder on normal transactions only, flag fraud by high reconstruction error, choose a threshold, and report precision, recall and F1. | Credit Card Fraud Detection | `practical_24_anomaly_detection_autoencoder.py` | `python practical_24_anomaly_detection_autoencoder.py` |
| **25** | **Semantic Segmentation Using CNN**: Build a CNN segmentation model (e.g. U-Net) that labels each pixel of an image, show predicted masks, and evaluate with Intersection over Union (IoU). | Oxford-IIIT Pet | `practical_25_semantic_segmentation_unet.py` | `python practical_25_semantic_segmentation_unet.py` |

---

## 💡 Notes for Lab Practicals and Viva Examinations

1. **Self-Contained Execution**: All practicals are configured to run completely offline without relying on multi-gigabyte external file downloads that may freeze or fail under college network firewalls.
2. **Clean Output**: TensorFlow informational logs (`oneDNN`, `AVX2`, etc.) are suppressed using `TF_CPP_MIN_LOG_LEVEL = '2'` so only clean training progress and evaluation tables appear in your terminal.
3. **Execution Speed**: Every script is optimized to finish training in **10 to 30 seconds** on any standard laptop CPU.
