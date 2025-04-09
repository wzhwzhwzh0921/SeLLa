# Code of Paper *"Enhancing LLM-based Recommendation through Semantic-Aligned Collaborative Knowledge"*



## 📊 Evaluation  
To reproduce the evaluation results of **SeLLa**, follow these steps:  

###  📦  Step 1: Download Pre-trained Model Weights  
Download the dataset-specific weights from the links below and place them in the `./checkpoints` directory:  

| Dataset | Google Drive Link |  
| :--- | :--- |  
| **Amazon-Book** | [checkpoint-656-Book](https://drive.google.com/file/d/1A_Mw4mCJMPqe9aOoMbd5NW54GAx_6lMw/view?usp=sharing)|  
| **ML-1M** | [checkpoint-254-Movie](https://drive.google.com/drive/folders/1TOL_Rohyc0FrJeQZUuaWXCYEHqk3uuCR?usp=sharing) |  

### 🚀 Step 2: Run Evaluation Script  
Execute the following command to evaluate the model:  
```bash
dataset_name="Amazon-Book" # "Amazon-Book or "ML-1M"
model_path="checkpoints/checkpoint-656-Book"  # path_to_ckpt

python eval/evaluate.py \
  --dataset ${dataset_name} \
  --model_path ${model_path} \
  --batch_size 64 \
  --eval_res_path ./log/${dataset_name}_res.json

``` 

### 📝 Expected Output  
The script will compute metrics like AUC, UAUC (the values will match those reported in the paper's Table 3). Full logs will be saved in logs/evaluation_metrics.log, and detailed evaluation results will be stored in JSON format at ./output/${dataset_name}_eval_results.json.


---

## ⚡ Training  
To train **SeLLa** from scratch on your own dataset:  

### ⚙️ Prerequisites  
1. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
2. Preprocess your dataset (e.g., `Amazon-Book` or `ML-1M`) into the `data/` directory.  

### 🎯  Run Training  
```bash  
# stage1
python scripts/train/baseline_train_mf_movie_step1_cl.py

# stage2
bash /workspace/linjh/github/learn/SeLLa/scripts/train/finetune_stage2_book_step1_cycle_two.sh

# stage3
...
```  

### 🏋️  Training Notes  
- Training logs and checkpoints are saved to `./checkpoints`.  
