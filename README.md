# EthnicWearClassify - Traditional Clothing Recognition

EthnicWearClassify is an image classification project that identifies traditional clothing from various cultures around the world. By leveraging deep learning techniques, this project aims to celebrate cultural diversity and provide practical applications in fashion, e-commerce, and heritage preservation.



## 🧭 Overview

This project uses a convolutional neural network to classify traditional garments based on images. With a robust training pipeline and high-performing model, EthnicWearClassify achieves impressive accuracy in distinguishing cultural attire.



## 🎯 Objectives

- Automatically classify images of traditional clothing  
- Support cultural heritage and research through AI  
- Provide a foundation for fashion-related applications such as recommendation systems, virtual try-on, or tagging tools  
- Create an interactive web interface for easy use and demonstration  



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



## 🛠️ Tech Stack

- **Python** – Core programming language  
- **Fastai** – High-level deep learning library for model training  
- **Gradio** – For building the interactive web UI  
- **Hugging Face** – Model deployment and sharing  
- **Jupyter Notebook** – For model development and experimentation  



## 🤖 Model Training & Evaluation

- **Model Architecture**: `resnet50` (pretrained on ImageNet)  
- **Training Framework**: Fastai  
- **Dataset**: Custom image dataset collected and preprocessed for balanced class distribution  
- **Evaluation Metric**: Accuracy  
- **Final Accuracy**: ~99% on validation set  

The model was trained using Fastai’s fine-tuning strategy, leveraging transfer learning to optimize training efficiency and accuracy.

### Final Training Result
```
epoch	train_loss	valid_loss	error_rate	accuracy	time
0	    0.114393	0.048181	0.009547	0.990453	09:06
epoch	train_loss	valid_loss	error_rate	accuracy	time
0	    0.094507	0.039907	0.007160	0.992840	01:50
1	    0.083363	0.042534	0.007160	0.992840	01:50
2	    0.105858	0.061186	0.014320	0.985680	01:52
3	    0.121893	0.078579	0.014320	0.985680	01:51
4	    0.092064	0.060345	0.023866	0.976134	01:51
5	    0.076698	0.055057	0.014320	0.985680	01:51
6	    0.061697	0.057800	0.023866	0.976134	01:50
7	    0.047301	0.044533	0.014320	0.985680	01:52
8	    0.042707	0.053565	0.011933	0.988067	01:52
9	    0.046098	0.046974	0.016706	0.983294	01:53
```

## 🚀 Deployment

The model is deployed using **Gradio** and hosted on **Hugging Face Spaces**

👉 **[Try the Demo on Hugging Face Space](https://huggingface.co/spaces/SaifTusher/Traditional_Clothing_Recognition)**  

### 🖼️ Preview

Here are some examples of traditional clothing that the model can classify:

<p align="center">
  <img src="preview\preview-1.png" alt="Example 1" width="45%" style="margin-right: 10px;"/>
  <img src="preview\preview-2.png" alt="Example 2" width="45%"/>
</p>

### 🌐 Webpage

A dedicated **web application** has been built that allows users to **interact with the deployed model via an API**. Users can upload images and receive real-time predictions for traditional clothing classification.

🔗 **[Live Webpage](https://saif044.github.io/Traditional_Clothing_Recognition/)**  

## How to use

1. **Clone the repository**
   ```bash
   git clone https://github.com/Saif044/Traditional_Clothing_Recognition.git
   cd Traditional_Clothing_Recognition
   ```
2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Gradio app**
    ```bash
    python app.py
    ```
