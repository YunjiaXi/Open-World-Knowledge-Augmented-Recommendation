import json
import pickle
import os
import random

import torch
from  torch import nn
import torch.utils.data as Data
import numpy as np

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w') as w:
        json.dump(data, w)


def load_parse_from_json(parse, setting_path):
    with open(setting_path, 'r') as f:
        setting = json.load(f)
    parse_dict = vars(parse)
    for k, v in setting.items():
        parse_dict[k] = v
    return parse


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def load_data(data_path, lm_vec=None, item_vec=None):
    with open(data_path, 'rb') as r:
        hist, itm_fts, usr_fts, lm_idx, lbs = pickle.load(r)
    if lm_vec:
        hist_dens = np.array(lm_vec)[np.array(lm_idx)]
        itm_dens = [item_vec[itm_ft[0]] for itm_ft in itm_fts]
        # data_size = len(hist)
        # hist_dens = np.random.random((data_size, 4096))
        # itm_dens = np.random.random((data_size, 4096))
        hist_dens = torch.from_numpy(hist_dens).long()
        itm_dens = torch.tensor(itm_dens).long()
    hist_spar = torch.tensor(hist).long()
    itm_spar = torch.tensor(itm_fts).long()
    usr_fts = torch.tensor(usr_fts).long()
    lbs = torch.tensor(lbs).long()
    if lm_vec:
        data_set = Data.TensorDataset(hist_spar, hist_dens, itm_spar, itm_dens, usr_fts, lbs)
    else:
        data_set = Data.TensorDataset(hist_spar, itm_spar, usr_fts, lbs)
    return data_set


def load_train_and_test(train_path, test_path, lm_vec_path=None):
    if lm_vec_path:
        with open(lm_vec_path, 'rb') as r:
            train_hist_vec, test_hist_vec, item_vec_dict = pickle.load(r)
    else:
        train_hist_vec, test_hist_vec, item_vec_dict = None, None, None
    train_set = load_data(train_path, train_hist_vec, item_vec_dict)
    test_set = load_data(test_path, test_hist_vec, item_vec_dict)
    return train_set, test_set


def correct_title(title):
    title = title.strip()
    spl_list = title.split(',')
    # spl_list = [word.strip() for word in spl_list]
    last_word = spl_list[-1].strip().lower()
    if last_word == 'the' or last_word == 'a':
        tmp = ','.join(spl_list[:-1])
        title = spl_list[-1].strip() + ' ' + tmp
    return title


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight)

def str2list(s):
    return [int(i.strip()) for i in s.strip().split(',')]
# def obtain_verbalizer_ids(verbalizer, tokenizer):
#     verbalizer_ids = tokenizer.convert_tokens_to_id(list(verbalizer.keys()))
#     index2ids = {1: verbalizer_ids[i] for i in range(len(verbalizer_ids))}
#     return verbalizer_ids, index2ids


def get_paragraph_representation(outputs, mask, pooler='cls', dim=1):
    last_hidden = outputs.last_hidden_state
    hidden_states = outputs.hidden_states

    # Apply different poolers

    if pooler == 'cls':
        # There is a linear+activation layer after CLS representation
        return outputs.pooler_output.cpu()  # chatglm不能用，用于bert
    elif pooler == 'cls_before_pooler':
        return last_hidden[:, 0].cpu()
    elif pooler == "avg":
        return ((last_hidden * mask.unsqueeze(-1)).sum(dim) / mask.sum(dim).unsqueeze(-1)).cpu()
    elif pooler == "avg_first_last":
        first_hidden = hidden_states[1]
        last_hidden = hidden_states[-1]
        pooled_result = ((first_hidden + last_hidden) / 2.0 * mask.unsqueeze(-1)).sum(dim) / mask.sum(dim).unsqueeze(-1)
        return pooled_result.cpu()
    elif pooler == "avg_top2":
        second_last_hidden = hidden_states[-2]
        last_hidden = hidden_states[-1]
        pooled_result = ((last_hidden + second_last_hidden) / 2.0 * mask.unsqueeze(-1)).sum(dim) / mask.sum(dim).unsqueeze(-1)
        return pooled_result.cpu()
    elif pooler == 'len_last':  # 根据padding方式last方式也不一样
        lens = mask.unsqueeze(-1).sum(dim)
        # index = torch.arange(last_hidden.shape[0])
        # print(index)
        pooled_result = [last_hidden[i, lens[i] - 1, :] for i in range(last_hidden.shape[0])]
        pooled_result = torch.concat(pooled_result, dim=0)
        return pooled_result.cpu()
    elif pooler == 'last':
        if dim == 0:
            return last_hidden[-1, :, :]
        else:
            return last_hidden[:, -1, :]
    elif pooler == 'wavg':
        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden.size())
            .float().to(last_hidden.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            mask
            .unsqueeze(-1)
            .expand(last_hidden.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden * input_mask_expanded * weights, dim=dim)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=dim)

        pooled_result = sum_embeddings / sum_mask
        return pooled_result.cpu()
    else:
        raise NotImplementedError

