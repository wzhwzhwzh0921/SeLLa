# Code of Paper *"Enhancing LLM-based Recommendation through Semantic-Aligned Collaborative Knowledge"*



## 📊 Evaluation  
To reproduce the evaluation results of **SeLLa**, follow these steps:  

###  📦  Step 1: Download Datasets and Finetuned Model Weights  
Download datasets and corresponding finetuned model weights from the links below and place them in `eval_data_book/eval_data_movie` and `final_ckpt_path_book/final_ckpt_path_movie`

| Datasets link | Finetuned Model weights link |  
| :--- | :--- |  
| **[Amazon-Book](https://drive.google.com/file/d/13yPDzIDH025CdGttRg3fIhWhO9jbpDHS/view?usp=sharing)** | [checkpoint-656-Book](https://drive.google.com/file/d/1A_Mw4mCJMPqe9aOoMbd5NW54GAx_6lMw/view?usp=sharing)|  
| **[ML-1M](https://drive.google.com/file/d/1dvBXIsixSq7e5FZDAmuwhtCPrwSoiDBv/view?usp=sharing)** | [checkpoint-254-Movie](https://drive.google.com/drive/folders/1TOL_Rohyc0FrJeQZUuaWXCYEHqk3uuCR?usp=sharing) |  

### 🚀 Step 2: Run Evaluation Script  
Execute the following command to evaluate the model:  
```bash
ds_config_path=examples/sft/ds_config_zero3.json

bash examples/sft/finetune_stage2_book_step1_cycle_two.sh ${final_ckpt_path_book} ${train_data_dummy} ${ds_config_path} ${eval_data_book} ${output_dir_book}

bash examples/sft/finetune_stage2_movie_step1_cycle_two.sh ${final_ckpt_path_movie} ${train_data_dummy} ${ds_config_path} ${eval_data_movie} ${output_dir_movie}
``` 

### 📝 Expected Output  
The script will compute metrics like AUC, UAUC (the values will match those reported in the paper's Table 3). 
Detailed evaluation results and full logs d will be stored at `output_dir_book/output_dir_movie`


---

## ⚡ Training  

...
