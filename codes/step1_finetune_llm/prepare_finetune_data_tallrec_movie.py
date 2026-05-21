import json

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import os
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

class prepare_for_prompt(Dataset):
    def __init__(self, data, max_len=10):
        # super.__init__()
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_i = self.data[index]
        uid, target, his, label = data_i[0], data_i[1], data_i[2], data_i[3],
        if len(his) < self.max_len:
            his = np.array(his)
        elif len(his) > self.max_len:
            his = np.array(his)[-self.max_len:]
        else:
            his = np.array(his)
        return uid, target, his, label
def write_in_file(data, file_name, path):
    instruct = "A user has given high ratings to the following movies. " \
               "Given the movie list and a target movie's title. " \
               "Please identify whether the user will like the target movie by answering 'Yes.' or 'No.'"
    with open(path+file_name+".jsonl", "w", encoding='utf-8') as f:
        for uid, target, his, label in data:
            # 去除换行符
            his_non = []
            for i in his:
                his_non.append(i.replace('\n', '').replace('\r', ''))
            if len(his_non)<2:
                continue
            if label ==0 :
                answer = "No."
            else:
                answer = "Yes."
            chatml_format = {
                "type": "chatml",
                "messages": [
                    {"role": "system", "content": instruct},
                    {"role": "user", "content": f"MovieList: {his_non}, TargetMovie: {target}"},
                    {"role": "assistant", "content": answer}
                ],
                "source": uid
            }
            f.write(json.dumps(chatml_format) + "\n")
    print(file_name + "Write finish!")
data_dir = "/datas/wuxi/Projects/datas/data/MovieLens-1M/"
path = "../data/movie/"
train_data = pd.read_pickle(data_dir+"train_ood2.pkl")[['uid','title','his_title', 'label']].values
valid_data = pd.read_pickle(data_dir+"valid_ood2.pkl")[['uid','title','his_title', 'label']].values
test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','title','his_title', 'label']].values

# user_num = max(train_data[0].max(),valid_data[0].max(), test_data[0].max()) + 1
# item_num = max(train_data[1].max(),valid_data[1].max(), test_data[1].max()) + 1
test_data_warm_or_cold = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','title','his_title', 'label','not_cold']]
test_data_warm = test_data_warm_or_cold[test_data_warm_or_cold['not_cold'].isin([1])][['uid', 'title', 'his_title', 'label']].values
test_data_cold = test_data_warm_or_cold[test_data_warm_or_cold['not_cold'].isin([0])][['uid', 'title', 'his_title', 'label']].values

train_data = prepare_for_prompt(train_data)
valid_data = prepare_for_prompt(valid_data)
test_data = prepare_for_prompt(test_data)
test_data_warm = prepare_for_prompt(test_data_warm)
test_data_cold = prepare_for_prompt(test_data_cold)
write_in_file(train_data, "train_data", path)
write_in_file(valid_data, "valid_data", path)
write_in_file(test_data, "test_data", path)
write_in_file(test_data_warm, "test_data_warm", path)
write_in_file(test_data_cold, "test_data_cold", path)

