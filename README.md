# Code of Paper *"Enhancing LLM-based Recommendation through Semantic-Aligned Collaborative Knowledge"*



## 📊 Evaluation  
To reproduce the evaluation results of **SeLLa**, follow these steps:  

###  📦  Step 1: Download Datasets and Finetuned Model Weights  
Download datasets and corresponding finetuned model weights from the links below and place them in `eval_data_book/eval_data_movie` and `final_ckpt_path_book/final_ckpt_path_movie`

| Datasets link | Finetuned Model weights link |  
| :--- | :--- |  
| **[Amazon-Book](https://drive.google.com/file/d/13yPDzIDH025CdGttRg3fIhWhO9jbpDHS/view?usp=sharing)** | [checkpoint-656-Book](https://drive.google.com/drive/folders/1fDO8zweG3T_5y-2leoEnSNpsmBnilUQC?usp=sharing)|  
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

## ⚡Training Steps

### Step 1: Fine-tune the LLM on recommendation instructions

This stage builds the TaLLRec-style instruction data and fine-tunes the LLM with LoRA. The prompt contains a user interaction history and a target item title, and the target answer is `Yes.` or `No.`.

#### 1.1 Generate instruction-tuning JSONL data

For MovieLens-1M:

```bash
cd codes/step1_finetune_llm
python prepare_finetune_data_tallrec_movie.py
```

For Amazon Book:

```bash
cd codes/step1_finetune_llm
python prepare_finetune_data_tallrec_book.py
```

The expected output files are:

```text
data/<dataset>/train_data.jsonl
data/<dataset>/valid_data.jsonl
data/<dataset>/test_data.jsonl
data/<dataset>/test_data_warm.jsonl
data/<dataset>/test_data_cold.jsonl
```

If your script writes to `./data/<dataset>/` while the shell scripts read from `../data/<dataset>/`, make the paths consistent before continuing.

#### 1.2 Run LoRA fine-tuning

Edit `finetunemovie.sh` first:

```bash
MODEL=/path/to/Qwen2-7B
TRAIN_DATA=../data/movie/train_data.jsonl
EVAL_DATA=../data/movie/test_data.jsonl
OUTPUT_DIR=../output/movie
LOG_FILE=../log/movie/finetune_movie_log.out
CUDA_VISIBLE_DEVICES=0
```

Then run:

```bash
cd codes/step1_finetune_llm
bash finetunemovie.sh
```

For Amazon Book, switch the dataset paths:

```bash
TRAIN_DATA=../data/book/train_data.jsonl
EVAL_DATA=../data/book/test_data.jsonl
OUTPUT_DIR=../output/book
LOG_FILE=../log/book/finetune_book_log.out
```

Important training parameters in the provided script:

```text
num_train_epochs              = 1
per_device_train_batch_size   = 8
gradient_accumulation_steps   = 16
learning_rate                 = 8e-5
model_max_length              = 768
use_lora                      = True
q_lora                        = False
deepseed config               = ds_config_zero3.json
```

Main output:

```text
output/<dataset>/checkpoint-*/
```

Use one of these LoRA checkpoints as `lora_path` in Step 2.

---

### Step 2: Train the semantic-aligned collaborative model

This stage has two sub-steps. First, distill item semantic embeddings from the fine-tuned LLM. Second, train the MF collaborative model with both recommendation loss and contrastive alignment loss.

#### 2.1 Distill item semantic embeddings from the fine-tuned LLM

Edit the corresponding script in `codes/step2_train_collab/`:

```python
# llm_embeds_movie.py or llm_embeds_book.py

data_type = "movie"  # or "book"
id2title_path = {
    "movie": "../data/movie/id2title.jsonl",
    "book": "../data/book/id2title.jsonl"
}
model_path = "/path/to/base-llm"
lora_path = "../output/<dataset>/checkpoint-xxx"
model_dim = 3584  # set to the hidden size of the selected base LLM
```

Then run:

```bash
cd codes/step2_train_collab
python llm_embeds_movie.py
# or
python llm_embeds_book.py
```

Main output:

```text
../output/<dataset>/item_embeds.pt
```

The downstream `train_mf.py` loads this tensor as the LLM semantic target. Make sure the save path in `llm_embeds_*.py` exactly matches the load path in `train_mf.py`, for example:

```python
pretrained_weights = torch.load('../output/book/item_embeds.pt')
```

#### 2.2 Train MF with semantic alignment

Edit `train_mf.py`:

```python
data_type = "movie"  # or "book"
data_dirs = {
    "movie": "/path/to/MovieLens-1M/processed/data/",
    "book": "/path/to/AmazonBook/processed/data/"
}
```

Check these hyperparameters in the `__main__` block:

```text
embedding_size = 256
epoch          = 3000
eval_epoch     = 1
patience       = 50 or 100
batch_size     = 512 or 10240, depending on GPU memory
seed           = 2025 or 2026
```

Run:

```bash
cd codes/step2_train_collab
bash train.sh
```

or directly:

```bash
torchrun --standalone --nproc_per_node 1 train_mf.py
```

Main output:

```text
../output/<dataset>/<seed>_<month>_<day>_mf_<dataset>_collab_model.pth
```

This file is the trained collaborative model. It contains:

- user embeddings;
- item embeddings;
- `item_embedding_llm`, initialized from the LLM-distilled item embeddings;
- `trans_1` and `trans_2`, the projection layers used for semantic alignment.

---

### Step 3: Train SeLLa-Rec with collaborative special tokens

This stage injects collaborative knowledge into the LLM through three special tokens:

```text
<User_ID>
<Item_ID>
<Warm_ID>
```

`<User_ID>` and `<Item_ID>` are projected from MF user/item embeddings. `<Warm_ID>` is projected from the LLM-distilled item embedding.

#### 3.1 Generate SeLLa-Rec training JSONL data

Edit `prepare_finetune_data.py`:

```python
data_type = "movie"  # or "book"
data_path = {
    "movie": "/path/to/MovieLens-1M/processed/data/",
    "book": "/path/to/AmazonBook/processed/data/"
}
out_path = {
    "movie": "../data/movie/",
    "book": "../data/book/"
}
```

Run:

```bash
cd codes/step3_train_sella
python prepare_finetune_data.py
```

The generated JSONL records contain:

```json
{
  "type": "chatml",
  "messages": [...],
  "source": 123,
  "source_item": 456,
  "source_history": [12, 34, 56, -1, -1]
}
```

These fields are required by `train_sella_origin.py`:

- `source` -> `user_id`
- `source_item` -> `item_id`
- `source_history` -> `history_id`

#### 3.2 Configure the final SeLLa-Rec trainer

Edit `train_sella_origin.py`:

```python
data_type = "book"  # or "movie"

lora_model_path = {
    "movie": "/path/to/step1/output/movie/checkpoint-xxx/",
    "book": "/path/to/step1/output/book/checkpoint-xxx/"
}

small_model_path = {
    "movie": "/path/to/step2/output/movie/<mf_checkpoint>.pth",
    "book": "/path/to/step2/output/book/<mf_checkpoint>.pth"
}

user_num = {
    "movie": 839,
    "book": 22967
}

item_num = {
    "movie": 3256,
    "book": 34154
}
```

If you want to follow the full SeLLa-Rec alignment setting, initialize the collaborative projection layer from the Step 2 MF projection layers:

```python
model.model.prepare_collm_prompt(
    user_token_id=sep_token_id_li[0],
    item_token_id=sep_token_id_li[1],
    warm_token_id=sep_token_id_li[2],
    small_emb_dim=256,
    projection_model_path=None,
    big_linear_path=None,
    pretrained_with_small=True,
)
```

The uploaded script currently sets `pretrained_with_small=False`, which leaves the collaborative projection randomly initialized. This is useful for ablations, but not the full CL + Projection setting described in the paper.

#### 3.3 Launch final training

Edit `train_sella.sh`:

```bash
MODEL=/path/to/Qwen2-7B
TRAIN_DATA=../data/book/train_data.jsonl
EVAL_DATA=../data/book/test_data.jsonl
OUTPUT_DIR=../output/book2e4/
LOG_DIR=../log/book/sella-mf2.out
CUDA_VISIBLE_DEVICES=0,1
GPUS_PER_NODE=2
```

Then run:

```bash
cd codes/step3_train_sella
bash train_sella.sh
```

Important training parameters in the provided script:

```text
num_train_epochs              = 1
per_device_train_batch_size   = 5
gradient_accumulation_steps   = 30
learning_rate                 = 2e-4
model_max_length              = 768
save_steps                    = 100
eval_steps                    = 100
use_lora                      = True
q_lora                        = False
deepseed config               = ds_config_zero3.json
```

Main outputs:

```text
<OUTPUT_DIR>/checkpoint-*/projection.pth
<OUTPUT_DIR>/checkpoint-*/collab.pth
<OUTPUT_DIR>/checkpoint-*/llm.pth
<OUTPUT_DIR>/projection.pth
<OUTPUT_DIR>/collab.pth
<OUTPUT_DIR>/llm.pth
```

The three saved files correspond to:

| File | Meaning |
|---|---|
| `projection.pth` | projection layer for `<User_ID>` and `<Item_ID>` |
| `collab.pth` | trainable collaborative embeddings/model parameters |
| `llm.pth` | projection layer for `<Warm_ID>` |
