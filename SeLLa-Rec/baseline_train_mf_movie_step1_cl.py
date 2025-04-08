import json
import sys

from sklearn.decomposition import PCA
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import numpy as np
import torch.optim
from sklearn.metrics import roc_auc_score
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
import random 

import os
import time

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

class MatrixFactorization(nn.Module):
    # here we does not consider the bais term
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.padding_index = 0
        pretrained_weights = torch.load('***path_to_item_embeds***')
        pretrained_weights = torch.tensor(pretrained_weights).float()
        # pretrained_weights = self.pca(pretrained_weights)
        item_emb_raw = pretrained_weights # 商品嵌入层
        self.trans_1 = nn.Linear(256, 1024, bias=True)
        self.gelu = nn.GELU()
        self.trans_2 = nn.Linear(1024, item_emb_raw.size(1), bias=True)

        self.item_embedding_llm = nn.Embedding.from_pretrained(item_emb_raw,
                                                           freeze=False, padding_idx=0)
        
        self.user_embedding = nn.Embedding(config.user_num, config.embedding_size, padding_idx=self.padding_index)
        self.item_embedding = nn.Embedding(config.item_num, config.embedding_size, padding_idx=self.padding_index)

        print(self.user_embedding.weight.size())
        print("creat MF model, user num:", config.user_num, "item num:", config.item_num)

    def user_encoder(self, users, all_users=None):
        # print("user max:", users.max(), users.min())
        return self.user_embedding(users)

    def item_encoder(self, items, all_items=None):
        # print("items max:", items.max(), items.min())
        return self.item_embedding(items)

    def computer(
            self):  # does not need to compute user reprensentation, directly taking the embedding as user/item representations
        return None, None

    def InfoNCE(self, out_1, out_2):
        out_1, out_2 = F.normalize(out_1, dim=1), F.normalize(out_2, dim=1)

        similarity = torch.exp(torch.mm(out_1, out_2.t()) / 0.2)
        neg = torch.sum(similarity, 1)

        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / 0.2)

        neg = ((0.1 * (pos + neg)) ** 0.01) / 0.01
        pos = -(pos ** 0.01) / 0.01
        loss = pos.mean() + neg.mean()
        return loss

    def forward(self, users, items):
        user_embedding = self.user_embedding(users)
        item_embedding = self.item_embedding(items)
        item_embedding_llm = self.item_embedding_llm(items)
        cl_loss = self.InfoNCE(item_embedding_llm, self.trans_2(self.gelu(self.trans_1(item_embedding))))
        #
        # user_embedding = self.trans_2(self.trans_1(self.user_embedding(users)))
        # item_embedding = self.trans_2(self.trans_1(self.item_embedding(items)))

        matching = torch.mul(user_embedding, item_embedding).sum(dim=-1)
        return matching, cl_loss
def uAUC_me(user, predict, label):
    if not isinstance(predict,np.ndarray):
        predict = np.array(predict)
    if not isinstance(label,np.ndarray):
        label = np.array(label)
    predict = predict.squeeze()
    label = label.squeeze()

    start_time = time.time()
    u, inverse, counts = np.unique(user,return_inverse=True,return_counts=True) # sort in increasing
    index = np.argsort(inverse)
    candidates_dict = {}
    k = 0
    total_num = 0
    only_one_interaction = 0
    computed_u = []
    for u_i in u:
        start_id,end_id = total_num, total_num+counts[k]
        u_i_counts = counts[k]
        index_ui = index[start_id:end_id]
        if u_i_counts ==1:
            only_one_interaction += 1
            total_num += counts[k]
            k += 1
            continue
        # print(index_ui, predict.shape)
        candidates_dict[u_i] = [predict[index_ui], label[index_ui]]
        total_num += counts[k]
        
        k+=1
    print("only one interaction users:",only_one_interaction)
    auc=[]
    only_one_class = 0

    for ui,pre_and_true in candidates_dict.items():
        pre_i,label_i = pre_and_true
        try:
            ui_auc = roc_auc_score(label_i,pre_i)
            auc.append(ui_auc)
            computed_u.append(ui)
        except:
            only_one_class += 1
            # print("only one class")
        
    auc_for_user = np.array(auc)
    print("computed user:", auc_for_user.shape[0], "can not users:", only_one_class)
    uauc = auc_for_user.mean()
    print("uauc for validation Cost:", time.time()-start_time,'uauc:', uauc)
    return uauc, computed_u, auc_for_user


class early_stoper(object):
    def __init__(self,ref_metric='valid_auc', incerase =True,patience=20) -> None:
        self.ref_metric = ref_metric
        self.best_metric = None
        self.increase = incerase
        self.reach_count = 0
        self.patience= patience
        # self.metrics = None
    
    def _registry(self,metrics):
        self.best_metric = metrics

    def update(self, metrics):
        if self.best_metric is None:
            self._registry(metrics)
            return True
        else:
            if self.increase and metrics[self.ref_metric] > self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True
            elif not self.increase and metrics[self.ref_metric] < self.best_metric[self.ref_metric]:
                self.best_metric = metrics
                self.reach_count = 0
                return True 
            else:
                self.reach_count += 1
                return False

    def is_stop(self):
        if self.reach_count>=self.patience:
            return True
        else:
            return False

# set random seed   
def run_a_trail(train_config,log_file=None, save_mode=False,save_file=None,need_train=True,warm_or_cold=None):
    seed = train_config['seed']
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    data_dir = "***path_to_data_dir***"
    # train_data = pd.read_pickle(data_dir + "train_ood2.pkl")[['uid', 'iid', 'title', 'his', 'his_title']].values
    # valid_data = pd.read_pickle(data_dir + "valid_ood2.pkl")[['uid', 'iid', 'title', 'his', 'his_title']].values
    # test_data = pd.read_pickle(data_dir + "test_ood2.pkl")[['uid', 'iid', 'title', 'his', 'his_title']].values
    #
    # user_history = {}

    # #遍历数据
    # for uid, iid, _, his, _ in train_data:
    #     if uid not in user_history:
    #         user_history[uid] = set()
    #     # 将历史交互项目添加到用户历史中
    #     user_history[uid].update(his)
    # user_history_list = {int(uid): [int(item) for item in history] for uid, history in user_history.items()}
    # with open(data_dir + 'u_his.json', 'w') as file:
    #     json.dump(user_history_list, file)
    #
    # from collections import defaultdict
    # def extract_item_info(data):
    #     item_title_dict = defaultdict(str)
    #     for row in data:
    #         _, iid, title, his, his_title = row
    #
    #         # 添加主要商品
    #         item_title_dict[int(iid)] = title
    #
    #         # 添加历史交互商品
    #         for hist_id, hist_title in zip(his, his_title):
    #             if hist_title:  # 确保标题不是空字符串
    #                 item_title_dict[int(hist_id)] = hist_title
    #     return item_title_dict
    #
    # # 从所有数据集中提取商品信息
    # train_items = extract_item_info(train_data)
    # valid_items = extract_item_info(valid_data)
    # test_items = extract_item_info(test_data)
    # train_items.update(test_items)
    # train_items.update(valid_items)
    # print(len(train_items))
    # with open(data_dir + 'id2title.json', 'w') as file:
    #     json.dump(train_items, file)

    train_data = pd.read_pickle(data_dir+"train_ood2.pkl")[['uid','iid','label']].values
    valid_data = pd.read_pickle(data_dir+"valid_ood2.pkl")[['uid','iid','label']].values
    test_data = pd.read_pickle(data_dir+"test_ood2.pkl")[['uid','iid','label']].values
    user_num = max(train_data[:,0].max(), valid_data[:,0].max(), test_data[:,0].max()) + 1
    item_num = max(train_data[:,1].max(), valid_data[:,1].max(), test_data[:,1].max()) + 1

    if warm_or_cold is not None:
        if warm_or_cold == 'warm':
            test_data = pd.read_pickle(data_dir + "test_ood2.pkl")[['uid', 'iid', 'label', 'not_cold']]
            test_data = test_data[test_data['not_cold'].isin([1])][['uid', 'iid', 'label']].values
            print("warm data size:", test_data.shape[0])
            # pass
        else:
            test_data = pd.read_pickle(data_dir + "test_ood2.pkl")[['uid', 'iid',  'label', 'not_cold']]
            test_data = test_data[test_data['not_cold'].isin([0])][['uid', 'iid',  'label']].values
            print("cold data size:", test_data.shape[0])
    

    print("user nums:", user_num, "item nums:", item_num)

    mf_config={
        "user_num": int(user_num),
        "item_num": int(item_num),
        "embedding_size": int(train_config['embedding_size'])
        }
    mf_config = omegaconf.OmegaConf.create(mf_config)

    train_data_loader = DataLoader(train_data, batch_size = train_config['batch_size'], shuffle=True)
    valid_data_loader = DataLoader(valid_data, batch_size = train_config['batch_size'], shuffle=False)
    test_data_loader = DataLoader(test_data, batch_size = train_config['batch_size'], shuffle=False)



    model = MatrixFactorization(mf_config).cuda()

    opt = torch.optim.Adam(model.parameters(),lr=train_config['lr'],weight_decay=train_config['wd'])
    early_stop = early_stoper(ref_metric='valid_auc',incerase=True,patience=train_config['patience'])
    # trainig part
    criterion = nn.BCEWithLogitsLoss()

    if not need_train:
        model.load_state_dict(torch.load(save_file))
        model.eval()
        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(valid_data_loader):
            batch_data = batch_data.cuda()
            ui_matching, _ = model(batch_data[:,0].long(),batch_data[:,1].long())
            users.extend(batch_data[:,0].cpu().numpy())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
        valid_auc = roc_auc_score(label,pre)
        valid_uauc, _, _ = uAUC_me(users, pre, label)
        label = np.array(label)
        pre = np.array(pre)
        thre = 0.1
        pre[pre>=thre] =  1
        pre[pre<thre]  =0
        val_acc = (label==pre).mean()

        pre=[]
        label = []
        users = []
        for batch_id,batch_data in enumerate(test_data_loader):
            batch_data = batch_data.cuda()
            ui_matching, _ = model(batch_data[:,0].long(),batch_data[:,1].long())
            pre.extend(ui_matching.detach().cpu().numpy())
            label.extend(batch_data[:,-1].cpu().numpy())
            users.extend(batch_data[:,0].cpu().numpy())
        test_auc = roc_auc_score(label,pre)
        test_uauc, _, _ = uAUC_me(users, pre, label)

        print("valid_auc:{}, valid_uauc:{}, test_auc:{}, test_uauc:{}, acc: {}".format(valid_auc, valid_uauc, test_auc, test_uauc, val_acc))
        return 
    
    loss_records = {
        "loss_rec": [],  # List to store reconstruction loss for each batch
        "cl_loss": []    # List to store contrastive learning loss for each batch
    }
    for epoch in range(train_config['epoch']):
        model.train()
        
        for bacth_id, batch_data in enumerate(train_data_loader):
            # break
            batch_data = batch_data.cuda()
            ui_matching, cl_loss = model(batch_data[:,0].long(),batch_data[:,1].long())
            loss_rec = criterion(ui_matching,batch_data[:,-1].float())
            # print("loss_rec:", loss_rec)
            # print("loss_cl:", cl_loss)
            # loss = loss_rec + cl_loss
            # loss_records["loss_rec"].append(loss_rec.item())  # Convert tensor to Python float
            # loss_records["cl_loss"].append(cl_loss.item())    # Convert tensor to 
            loss = loss_rec + cl_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        if epoch% train_config['eval_epoch']==0:
            model.eval()
            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(valid_data_loader):
                batch_data = batch_data.cuda()
                ui_matching,_ = model(batch_data[:,0].long(),batch_data[:,1].long())
                users.extend(batch_data[:,0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
            valid_auc = roc_auc_score(label,pre)
            valid_uauc, _, _ = uAUC_me(users, pre, label)

            pre=[]
            label = []
            users = []
            for batch_id,batch_data in enumerate(test_data_loader):
                batch_data = batch_data.cuda()
                ui_matching,_ = model(batch_data[:,0].long(),batch_data[:,1].long())
                users.extend(batch_data[:,0].cpu().numpy())
                pre.extend(ui_matching.detach().cpu().numpy())
                label.extend(batch_data[:,-1].cpu().numpy())
            test_auc = roc_auc_score(label,pre)
            test_uauc, _, _ = uAUC_me(users, pre, label)

            updated = early_stop.update({'valid_auc':valid_auc, 'valid_uauc':valid_uauc,'test_auc':test_auc, 'test_uauc':test_uauc, 'epoch':epoch})
            if updated and save_mode:
                torch.save(model.state_dict(),save_file)


            print("epoch:{}, valid_auc:{}, test_auc:{}, early_count:{}".format(epoch, valid_auc, test_auc, early_stop.reach_count))
            if early_stop.is_stop():
                print("early stop is reached....!")
                # print("best results:", early_stop.best_metric)
                break
            if epoch>500 and early_stop.best_metric[early_stop.ref_metric] < 0.52:
                print("training reaches to 500 epoch but the valid_auc is still less than 0.55")
                break
    print("train_config:", train_config,"\nbest result:",early_stop.best_metric)


    if log_file is not None:
        print("train_config:", train_config, "best result:", early_stop.best_metric, file=log_file)
        log_file.flush()

# with prtrain version:
if __name__=='__main__':
    # lr_ = [1e-1,1e-2,1e-3]
    lr_=[1e-2] #1e-2
    dw_ = [1e-4] #best 1e-2 1e-4
    embedding_size_ = [256]
    seeds = [2025]
    f=None
    for seed in seeds:
        for lr in lr_:
            for wd in dw_:
                for embedding_size in embedding_size_:
                    train_config={
                        'lr': lr,
                        'wd': wd,
                        'embedding_size': embedding_size,
                        "epoch": 3000,
                        "eval_epoch":1,
                        "patience":50,
                        "batch_size":1024,
                        "seed": seed
                    }
                    print(train_config)
                    save_path = f"***path_to_small_model(before trainning with LLM)***"
                    
                    run_a_trail(train_config=train_config, log_file=f, save_mode=True, save_file=save_path,need_train=True,warm_or_cold=None)
    if f is not None:
        f.close()
        