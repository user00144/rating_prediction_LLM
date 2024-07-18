
# Rating prediction project with Large Language Model Fintuning
___

<div align="center">
  <img width="450" alt="aa" src="https://github.com/user-attachments/assets/78c5b9be-7d84-4790-a345-f1151f1fb100">
</div>


## A Study on User Review text–Based Rating prediction system Using Large Language Model Finetuning
> **Personal Project** , **Mar. 2024 ~ May. 2024**

---



## Software Stacks
![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)




---

## Project Motivation

- Development of Large Language Models make that users to control computers asking in natural language

- Consumer Review, in particular, is made up of mostly natural language information Large Language Model can be good “Feature Extractor”

- In this study, we propose User Review Based Rating Prediction System


---

## Implementation

### 1. Dataset
- **Amazon Reviews Dataset 2023** : from [paper](https://arxiv.org/abs/2403.03952) Hou, Y., Li, J., He, Z., Yan, A., Chen, X., & McAuley, J. “Bridging Language and Items for Retrieval and Recommendation”, 2024.
- This Dataset contains 34 categories, totally 571.54 million reviews
- In this research, used "Movies and TV" dataset, extract 0.5 million review data

### 2. Data preprocessing

- Dataset split
<img width="473" alt="image" src="https://github.com/user-attachments/assets/2f6839f3-a156-4867-b6b6-e4dda0857e98">

- Prompting and Tokenize
<img width="367" alt="image" src="https://github.com/user-attachments/assets/a3fb8036-9b1d-45b6-9612-85882ff19b8b">

### 3. Training

- For training, we used LoRA(Low-Rank Adaptation of Large Language Models) method
- Based model : [declare-lab/flan-alpaca-gpt4-xl](https://huggingface.co/declare-lab/flan-alpaca-gpt4-xl) from huggingface
- Optimizer : AdamW
- Loss function : Cross Entropy Loss (Multiclass)
<img width="473" alt="image" src="https://github.com/user-attachments/assets/efc8da38-63b4-488e-9db1-42f917c1e29c">

### 4. Rating Prediction
 - Merge and Freeze LoRA weight
 - Rating Prediction with test dataset
 - Used metric : RMSE
 - **The Average RMSE value was 0.75626** , This means the predicted rating was about 0.7 points off from the actual rating...

---

## Outputs

- **Publication Conference Paper** in 19th Asia Pacific International Conference on Information Science and Technology (Jun. 2024)
![2](https://github.com/user-attachments/assets/d4e73dcf-bd21-443f-bfbb-595162e91914)
