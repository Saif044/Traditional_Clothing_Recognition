# EthnicWearClassify - Traditional Clothing Recognition

EthnicWearClassify is an image classification project that identifies traditional clothing from various cultures around the world. By leveraging deep learning techniques, this project aims to celebrate cultural diversity and provide practical applications in fashion, e-commerce, and heritage preservation.

---

## 🧭 Overview

This project uses a convolutional neural network to classify traditional garments based on images. With a robust training pipeline and high-performing model, EthnicWearClassify achieves impressive accuracy in distinguishing cultural attire.

---

## 🎯 Objectives

- Automatically classify images of traditional clothing  
- Support cultural heritage and research through AI  
- Provide a foundation for fashion-related applications such as recommendation systems, virtual try-on, or tagging tools  
- Create an interactive web interface for easy use and demonstration  

---

## 👘 Categories

The model is trained to recognize the following traditional clothing types:

- Bangladeshi Fotua  
- Bangladeshi Punjabi  
- Bangladeshi Lungi  
- Indian Saree/Sari  
- Japanese Kimono  
- Korean Hanbok  
- Scottish Kilt  
- Middle Eastern Thobe/Dishdasha  
- Mexican Huipil  
- Nigerian Agbada  
- Mongolian Deel  
- Indonesian Batik  
- Russian Sarafan  
- Native American Regalia  
- Chinese Qipao/Cheongsam  
- Thai Chut Thai (Traditional Thai dress)  
- Bavarian Dirndl/Lederhosen  
- Polynesian Lavalava  

---

## 🛠️ Tech Stack

- **Python** – Core programming language  
- **Fastai** – High-level deep learning library for model training  
- **Gradio** – For building the interactive web UI  
- **Hugging Face** – Model deployment and sharing  
- **Jupyter Notebook** – For model development and experimentation  

---

## 🤖 Model Training & Evaluation

- **Model Architecture**: `resnet50` (pretrained on ImageNet)  
- **Training Framework**: Fastai  
- **Dataset**: Custom image dataset collected and preprocessed for balanced class distribution  
- **Evaluation Metric**: Accuracy  
- **Final Accuracy**: ~99% on validation set  

The model was trained using Fastai’s fine-tuning strategy, leveraging transfer learning to optimize training efficiency and accuracy.

---

## ⚙️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/EthnicWearClassify.git
   cd EthnicWearClassify
