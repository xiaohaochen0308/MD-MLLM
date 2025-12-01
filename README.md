# MD-MLLM
## Disentangled Image-Text Classification  
### Using MLLM Knowledge to Bridge Visual Representations

This repository contains the official PyTorch implementation of the paper:

> **“Disentangled Image-Text Classification: Enhancing Visual Representations with MLLM-driven Knowledge Transfer”**

---

## 🚀 Pretrained Checkpoint

We provide a pretrained checkpoint of **MD-MLLM** on the **Food-101** dataset for reproducing the results reported in our paper.

- **Food-101 Dataset**  
  - Accuracy: **95.02%**  
  - 👉 _Download checkpoint_: `https://drive.google.com/file/d/1GrR2ZeEHV_Z4PTYa99M4_BhYe8YoHJis/view?usp=sharing`

You can use this checkpoint for direct evaluation or for fine-tuning on related multimodal classification tasks.

> At this stage, we only release the checkpoint trained on Food-101.  
> The N24News checkpoint will be considered for release in future updates.

---

## 📂 Datasets & MLLM Image Descriptions

In this work, we use two multimodal classification benchmarks, **N24News** and **Food-101**, and augment them with **MLLM-generated image descriptions**.  
Due to copyright restrictions, **we do not redistribute raw images**.  
Instead, we provide **processed text-side resources**, including dataset splits and MLLM-generated descriptions.

### Food-101

Food-101 is a widely used food image classification dataset.  
We adopt the most commonly used split protocol and filter out image–text pairs with missing or corrupted entries.  
The **final dataset split files** and the corresponding **MLLM-generated image descriptions** for Food-101 are available here:

- `splits/` (train/val/test split files)  
- `descriptions/` (MLLM-generated image descriptions aligned with the split)

👉 Download (text annotations only):  
`https://drive.google.com/drive/folders/1-2XN6tWyW-X7NZ3-r3U5_E5ItT1_urBq`

Users should obtain the original Food-101 images from the official source and then align them with our provided splits and descriptions.

### N24News

N24News is a multimodal news classification dataset containing news articles and associated images.  
We adopt the **original split** provided by the dataset authors.  
Similar to Food-101, we release **only the text-side resources**, including:

- processed train/val/test split files;  
- MLLM-generated image descriptions paired with each news item.

👉 Download (text annotations only):  
`https://drive.google.com/drive/folders/1-PGpaZ4eyb8CVkYl1QJKvnFaQqpL8pi1`

The original N24News data (news articles and images) should be obtained from the official source.  
Our split files and MLLM descriptions can be directly used to reproduce the experimental setup in the paper.

---

## 🧩 Code Availability

We will gradually release the full training and evaluation code for MD-MLLM in future updates.  
The current repository mainly includes:

- example scripts for loading the pretrained checkpoint on the Food-101 dataset;
- basic evaluation and inference scripts based on the released dataset splits and MLLM-generated descriptions.

> 🔔 We will keep updating this repository and progressively release the complete training pipeline, ablation scripts, and more detailed documentation. Stay tuned.
