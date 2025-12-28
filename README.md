🐄 Cattle Breed Classification System (CattleAI)

An end-to-end deep learning–based web application that identifies cattle breeds from images using a Convolutional Neural Network (CNN) with transfer learning, deployed via a Flask web interface.

📌 Project Overview

Cattle breed identification is a challenging fine-grained image classification problem due to high visual similarity between breeds. This project leverages deep learning and transfer learning to automatically classify cattle breeds from uploaded images and display predictions through a user-friendly web interface.

## main Branch

In main branch we trained a cattle breed classification model from Mobilenetv2 using Feature Extraction method also implemented Overfitting Solutions for this model training. Accuracy upto: 68%

## cnn_transferL

In this Branch we created the cattle breed classification model from Scratch using CNN Algorithm, also applied solutions for Overfitting 
Accuracy upto: 49%

## finetune-model
In this branch we Fine-tuned the cattle classification model on Mobilenetv2, applied overfitting solution.We also implemented web interface  for this. Accuracy upto: 72%

## Overfitting Solutions

EarlyStopping

ReduceLROnPlateau

Dropout layers 

Data augmentation

## Future Enhancements

Collect larger and more diverse datasets

Use stronger architectures (EfficientNet, ResNet)

Improve accuracy using attention mechanisms

Deploy application online (Render / Railway)

Convert model to TensorFlow Lite
