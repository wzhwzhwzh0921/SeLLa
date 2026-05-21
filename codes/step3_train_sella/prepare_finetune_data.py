# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import json

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class prepare_for_prompt(Dataset):
    def __init__(self, data, max_len=10):
        # super.__init__()
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_i = self.data[index]
        #'uid', 'iid', 'title', 'his_title','his', 'label']
        uid, iid, target, his_title, his, label = data_i[0], data_i[1], data_i[2], data_i[3], data_i[4], data_i[5]
        if len(his) < self.max_len:
            his = np.array(his)
        elif len(his) > self.max_len:
            his = np.array(his)[-self.max_len:]
        else:
            his = np.array(his)
        
        if len(his_title) < self.max_len:
            his_title = np.array(his_title)
        elif len(his_title) > self.max_len:
            his_title = np.array(his_title)[-self.max_len:]
        else:
            his_title = np.array(his_title)

        return uid, iid, target, his_title, his, label
def write_in_file(data, file_name, path):

    instruct = "A user has given high ratings to the following movies. " \
               "Given the movielist, the user's feature and a target movie's feature and Title. " \
               "Please Use all available information to identify whether the user will like the target movie by answering 'Yes.' or 'No.'"
    with open(path+file_name+".jsonl", "w") as f:
        for uid, iid, target, his_title, his, label in data:

            # 去除换行符
            his_non = []
            for i in his_title:
                his_non.append(i.replace('\n', '').replace('\r', ''))
            if len(his_non)<2:
                continue
            his = list(his)
            for i in range(len(his)):
                his[i] = int(his[i])
            pad_value = -1
            max_length = 10
            # 进行padding或截断到指定长度
            if len(his) < max_length:
                # 需要padding
                his.extend([pad_value] * (max_length - len(his)))
            if label ==0 :
                answer = "No."
            else:
                answer = "Yes."
            chatml_format = {
                "type": "chatml",
                "messages": [
                    {"role": "system", "content": instruct},
                    {"role": "user", "content": f"The MovieList: {his_non}. The user's feature is <User_ID>. The target movie's feature is <Item_ID>, <Warm_ID>. Targetmovie: {target}."},
                    {"role": "assistant", "content": answer}
                ],
                "source": uid,
                "source_item": iid,
                "source_history": his
            }
            f.write(json.dumps(chatml_format) + "\n")
    print(file_name + "Write finish!")


def write_in_file_book(data, file_name, path):

    instruct = "A user has given high ratings to the following books. " \
               "Given the booklist, the user's feature and a target book's feature and Title. " \
               "Please Use all available information to identify whether the user will like the target book by answering 'Yes.' or 'No.'"
    with open(path+file_name+".jsonl", "w") as f:
        for uid, iid, target, his_title, his, label in data:

            # 去除换行符
            his_non = []
            for i in his_title:
                his_non.append(i.replace('\n', '').replace('\r', ''))
            if len(his_non)<2:
                continue
            his = list(his)
            for i in range(len(his)):
                his[i] = int(his[i])
            pad_value = -1
            max_length = 10
            # 进行padding或截断到指定长度
            if len(his) < max_length:
                # 需要padding
                his.extend([pad_value] * (max_length - len(his)))
            if label ==0 :
                answer = "No."
            else:
                answer = "Yes."
            chatml_format = {
                "type": "chatml",
                "messages": [
                    {"role": "system", "content": instruct},
                    {"role": "user", "content": f"The BookList: {his_non}. The user's feature is <User_ID>. The target book's feature is <Item_ID>, <Warm_ID>. Targetbook: {target}."},
                    {"role": "assistant", "content": answer}
                ],
                "source": uid,
                "source_item": iid,
                "source_history": his
            }
            f.write(json.dumps(chatml_format) + "\n")
    print(file_name + "Write finish!")

data_path = {
    "movie":"/root/autodl-tmp/Projects/CoLLM-QWen2/data/MoiveLens-1M/processed/data/",
    "book":"/datas/wuxi/Projects/datas/data/AmazonBook/data_5core_80w/",
}
out_path = {
    "movie":"../data/movie/",
    "book":"../data/book/",
}

data_type = "book"
data_dir = data_path[data_type]
out_dir = out_path[data_type]


train_data = pd.read_pickle(data_dir+"train_ood2.pkl")[['uid', 'iid', 'title', 'his_title','his', 'label']].values
valid_data = pd.read_pickle(data_dir+"valid_ood2.pkl")[['uid', 'iid', 'title', 'his_title','his', 'label']].values
test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid', 'iid', 'title', 'his_title', 'his', 'label']].values

# user_num = max(train_data[0].max(),valid_data[0].max(), test_data[0].max()) + 1
# item_num = max(train_data[1].max(),valid_data[1].max(), test_data[1].max()) + 1
test_data_warm_or_cold = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid', 'iid', 'title', 'his_title', 'his', 'label', 'not_cold']]
test_data_warm = test_data_warm_or_cold[test_data_warm_or_cold['not_cold'].isin([1])][['uid', 'iid',  'title', 'his_title', 'his', 'label']].values
test_data_cold = test_data_warm_or_cold[test_data_warm_or_cold['not_cold'].isin([0])][['uid', 'iid', 'title', 'his_title', 'his','label']].values
train_data = prepare_for_prompt(train_data)
valid_data = prepare_for_prompt(valid_data)
test_data = prepare_for_prompt(test_data)
test_data_warm = prepare_for_prompt(test_data_warm)
test_data_cold = prepare_for_prompt(test_data_cold)

if data_type == "movie":
    write_in_file(train_data, "train_data", out_dir)
    write_in_file(valid_data, "valid_data", out_dir)
    write_in_file(test_data, "test_data", out_dir)
    write_in_file(test_data_warm, "test_data_warm", out_dir)
    write_in_file(test_data_cold, "test_data_cold", out_dir)
else:
    write_in_file_book(train_data, "train_data", out_dir)
    write_in_file_book(valid_data, "valid_data", out_dir)
    write_in_file_book(test_data, "test_data", out_dir)
    write_in_file_book(test_data_warm, "test_data_warm", out_dir)
    write_in_file_book(test_data_cold, "test_data_cold", out_dir)