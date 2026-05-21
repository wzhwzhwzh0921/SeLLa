import json
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def get_embds(id2title_path = None,model_path = None,lora_path = None):
    id2title = []

    with open(id2title_path, 'r') as file:
        for line in file:
            id2title.append(json.loads(line))
    print(len(id2title))
    # item_embeds = np.array([])
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map='cuda:0')
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='cuda:0')
    
    
    config = PeftConfig.from_pretrained(lora_path)
    model = PeftModel.from_pretrained(
        model,
        lora_path,
    )

    item_size = len(id2title) + 1
    model_dim = 4096
    item_embeds = np.random.rand(item_size, model_dim)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    for line in id2title:
        try:
            title = line['title']
            inputs = tokenizer(f'Book: {title}', return_tensors="pt", padding=True, truncation=True, max_length=64).to(model.device)
            with torch.no_grad():
                output = model(**inputs, output_hidden_states=True)
            seq_embeds = output.hidden_states[-1][:, -1, :].detach().cpu().numpy()
            item_embeds[int(line['id'])] = seq_embeds
        except Exception as e:
            print(e)
            print('Here we get an error when processing id2title!')
            continue
    
    return item_embeds

data_type = "book"
id2title_path = {
    "movie":"../data/movie/id2title.jsonl",
    "book":"../data/book/id2title80w.jsonl"
}
model_path = "/datas/models/Qwen3-8B" 
# 126/254
lora_path = "/datas/wuxi/Projects/WorkSpace/SellaRec/codes/step1_finetune_llm/output/book/checkpoint-300"
item_embeds = get_embds(id2title_path[data_type],model_path,lora_path)
item_embeds = torch.LongTensor(item_embeds)
torch.save(item_embeds, f'../output/book-qwen3/item_embeds.pt')
