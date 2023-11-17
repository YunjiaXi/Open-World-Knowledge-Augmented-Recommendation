import json
import os
import pickle
from datetime import date
import random
from collections import defaultdict
import csv
from pre_utils import load_json, save_json, save_pickle, GENDER_MAPPING, \
    AGE_MAPPING, OCCUPATION_MAPPING

rerank_item_from_hist = 4
rerank_hist_len = 10
rerank_list_len = 10
ctr_hist_len = 10


def generate_ctr_data(sequence_data, lm_hist_idx, uid_set):
    # print(list(lm_hist_idx.values())[:10])
    full_data = []
    total_label = []
    for uid in uid_set:
        start_idx = lm_hist_idx[str(uid)]
        item_seq, rating_seq = sequence_data[str(uid)]
        for idx in range(start_idx, len(item_seq)):
            label = 1 if rating_seq[idx] > rating_threshold else 0
            full_data.append([uid, idx, label])
            total_label.append(label)
    print('user num', len(uid_set), 'data num', len(full_data), 'pos ratio',
          sum(total_label) / len(total_label))
    print(full_data[:5])
    return full_data


def generate_rerank_data(sequence_data, lm_hist_idx, uid_set, item_set):
    full_data = []
    for uid in uid_set:
        start_idx = lm_hist_idx[str(uid)]
        item_seq, rating_seq = sequence_data[str(uid)]
        idx = start_idx
        seq_len = len(item_seq)
        while idx < seq_len:
            end_idx = min(idx + rerank_item_from_hist, seq_len)
            chosen_iid = item_seq[idx:end_idx]
            neg_sample_num = rerank_list_len - len(chosen_iid)
            neg_sample = random.sample(item_set, neg_sample_num)
            candidates = chosen_iid + neg_sample
            chosen_rating = rating_seq[idx:end_idx]
            candidate_lbs = [1 if rating > rating_threshold else 0 for rating in
                             chosen_rating] + [0 for _ in range(neg_sample_num)]
            list_zip = list(zip(candidates, candidate_lbs))
            random.shuffle(list_zip)
            candidates[:], candidate_lbs[:] = zip(*list_zip)
            full_data.append([uid, idx, candidates, candidate_lbs])
            idx = end_idx
    print('user num', len(uid_set), 'data num', len(full_data))
    print(full_data[:5])
    return full_data


def generate_hist_prompt(sequence_data, item2attribute, datamap, lm_hist_idx, dataset_name):
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2user = datamap['id2user']
    if dataset_name == 'ml-1m':
        user2attribute = datamap['user2attribute']
    hist_prompts = {}
    print('item2attribute', list(item2attribute.items())[:10])
    for uid, item_rating in sequence_data.items():
        user = id2user[uid]
        item_seq, rating_seq = item_rating
        cur_idx = lm_hist_idx[uid]
        hist_item_seq = item_seq[:cur_idx]
        hist_rating_seq = rating_seq[:cur_idx]
        history_texts = []
        for iid, rating in zip(hist_item_seq, hist_rating_seq):
            tmp = '"{}", {} stars; '.format(itemid2title[str(iid)], int(rating))
            history_texts.append(tmp)
        if dataset_name == 'amz':
            # prompt = 'Analyze user\'s preferences on product (consider factors like genre, functionality, quality, ' \
            #          'price, design, reputation. Provide clear explanations based on ' \
            #          'relevant details from the user\'s product viewing history and other pertinent factors.'
            # hist_prompts[uid] = 'Given user\'s product rating history: ' + ''.join(history_texts) + prompt
            prompt = 'Analyze user\'s preferences on books about factors like genre, author, writing style, theme, ' \
                     'setting, length and complexity, time period, literary quality, critical acclaim (Provide ' \
                     'clear explanations based on relevant details from the user\'s book viewing history and other ' \
                     'pertinent factors.'
            hist_prompts[user] = 'Given user\'s book rating history: ' + ''.join(history_texts) + prompt
        elif dataset_name == 'ml-1m':
            gender, age, occupation = user2attribute[uid]
            user_text = 'Given a {} user who is aged {} and {}, this user\'s movie viewing history over time' \
                        ' is listed below. '.format(GENDER_MAPPING[gender], AGE_MAPPING[age],
                                                    OCCUPATION_MAPPING[occupation])
            question = 'Analyze user\'s preferences on movies (consider factors like genre, director/actors, time ' \
                       'period/country, character, plot/theme, mood/tone, critical acclaim/award, production quality, ' \
                       'and soundtrack). Provide clear explanations based on relevant details from the user\'s movie ' \
                       'viewing history and other pertinent factors.'
            hist_prompts[user] = user_text + ''.join(history_texts) + question
        else:
            raise NotImplementedError
    print('data num', len(hist_prompts))
    print(list(hist_prompts.items())[0])
    return hist_prompts


def generate_item_prompt(item2attribute, datamap, dataset_name):
    itemid2title = datamap['itemid2title']
    attrid2name = datamap['id2attribute']
    id2item = datamap['id2item']
    item_prompts = {}
    for iid, title in itemid2title.items():
        item = id2item[iid]
        if dataset_name == 'amz':
            brand, cate = item2attribute[str(iid)]
            brand_name = attrid2name[str(brand)]
            # cate_name = attrid2name[cate]
            item_prompts[item] = 'Introduce book {}, which is from brand {} and describe its attributes including but' \
                                ' not limited to genre, author, writing style, theme, setting, length and complexity, ' \
                                'time period, literary quality, critical acclaim.'.format(title, brand_name)
            # item_prompts[iid] = 'Introduce product {}, which is from brand {} and describe its attributes (including but' \
            #                     ' not limited to genre, functionality, quality, price, design, reputation).'.format(title, brand_name)
        elif dataset_name == 'ml-1m':
            item_prompts[item] = 'Introduce movie {} and describe its attributes (including but not limited to genre, ' \
                                'director/cast, country, character, plot/theme, mood/tone, critical ' \
                                'acclaim/award, production quality, and soundtrack).'.format(title)
        else:
            raise NotImplementedError
    print('data num', len(item_prompts))
    print(list(item_prompts.items())[0])
    return item_prompts


if __name__ == '__main__':
    random.seed(12345)
    DATA_DIR = '../data/'
    # DATA_SET_NAME = 'amz'
    DATA_SET_NAME = 'ml-1m'
    if DATA_SET_NAME == 'ml-1m':
        rating_threshold = 3
    else:
        rating_threshold = 4
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')
    SEQUENCE_PATH = os.path.join(PROCESSED_DIR, 'sequential_data.json')
    ITEM2ATTRIBUTE_PATH = os.path.join(PROCESSED_DIR, 'item2attributes.json')
    DATAMAP_PATH = os.path.join(PROCESSED_DIR, 'datamaps.json')
    SPLIT_PATH = os.path.join(PROCESSED_DIR, 'train_test_split.json')

    sequence_data = load_json(SEQUENCE_PATH)
    train_test_split = load_json(SPLIT_PATH)
    item2attribute = load_json(ITEM2ATTRIBUTE_PATH)
    item_set = list(map(int, item2attribute.keys()))
    print('final loading data')
    # print(list(item2attribute.keys())[:10])

    print('generating ctr train dataset')
    train_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                  train_test_split['train'])
    print('generating ctr test dataset')
    test_ctr = generate_ctr_data(sequence_data, train_test_split['lm_hist_idx'],
                                 train_test_split['test'])
    print('save ctr data')
    save_pickle(train_ctr, PROCESSED_DIR + '/ctr.train')
    save_pickle(test_ctr, PROCESSED_DIR + '/ctr.test')
    train_ctr, test_ctr = None, None

    print('generating reranking train dataset')
    train_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'],
                                        train_test_split['train'], item_set)
    print('generating reranking test dataset')
    test_rerank = generate_rerank_data(sequence_data, train_test_split['lm_hist_idx'],
                                       train_test_split['test'], item_set)
    print('save reranking data')
    save_pickle(train_rerank, PROCESSED_DIR + '/rerank.train')
    save_pickle(test_rerank, PROCESSED_DIR + '/rerank.test')
    train_rerank, test_rerank = None, None

    datamap = load_json(DATAMAP_PATH)

    statis = {
        'rerank_list_len': rerank_list_len,
        'attribute_ft_num': datamap['attribute_ft_num'],
        'rating_threshold': rating_threshold,
        'item_num': len(datamap['id2item']),
        'attribute_num': len(datamap['id2attribute']),
        'rating_num': 5,
        'dense_dim': 0,
    }
    save_json(statis, PROCESSED_DIR + '/stat.json')

    print('generating item prompt')
    item_prompt = generate_item_prompt(item2attribute, datamap, DATA_SET_NAME)
    print('generating history prompt')
    hist_prompt = generate_hist_prompt(sequence_data, item2attribute, datamap,
                                       train_test_split['lm_hist_idx'], DATA_SET_NAME)
    print('save prompt data')
    save_json(item_prompt, PROCESSED_DIR + '/prompt.item')
    save_json(hist_prompt, PROCESSED_DIR + '/prompt.hist')
    item_prompt, hist_prompt = None, None

