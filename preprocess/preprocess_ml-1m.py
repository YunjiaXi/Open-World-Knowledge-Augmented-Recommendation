'''
split train/test by user IDs, train: test= 9: 1
RS history: recent rated 10 items (pos & neg), ID & attributes & rating
LM history: one lm history for each user(max_len=30, item ID, attributes, rating)
attribute: category
rating >= 4 as positive, rating < 4 as negative, no negative sampling
'''
import os
import random
import tqdm
import html
from collections import defaultdict
import numpy as np
import json
from datetime import date
from string import ascii_letters, digits, punctuation, whitespace
from pre_utils import set_seed, parse, add_comma, save_json, correct_title

lm_hist_max = 30
train_ratio = 0.9
rating_score = 0.0  # rating score smaller than this score would be deleted
# user 60-core item 40-core
user_core = 0
item_core = 0
attribute_core = 0


def ml_1m(data_file, rating_score):
    datas = []
    all_num, wo_rating_num = 0, 0
    with open(data_file, 'r') as r:
        lines = r.readlines()
        for row in lines:
            uid, iid, rating, timestamp = row.split('::')
            timestamp = date.fromtimestamp(float(timestamp))
            all_num += 1
            if int(rating) < rating_score:
                wo_rating_num += 1
                continue
            datas.append((uid, iid, timestamp, int(rating)))
    print('total rating', all_num, 'wo rating or filter rating num', wo_rating_num, wo_rating_num / all_num)
    return datas


def ml_1m_user(data_file):
    user_info = {}
    with open(data_file, 'r') as r:
        lines = r.readlines()
        for row in lines:
            uid, gender, age, occupation, _ = row.split('::')
            user_info[uid] = [gender, int(age), int(occupation)]
    return user_info


def ml_1m_meta(meta_file, data_maps):  # return the metadata of products
    datas = {}
    item_asins = set(data_maps['item2id'].keys())
    with open(meta_file, 'r', encoding='latin-1') as r:
        lines = r.readlines()
        for row in lines:
            iid, title, cates = row.split('::')
            if iid not in item_asins:
                continue
            cate_list = cates.strip().split('|')
            for i in range(len(cate_list)):
                cate_list[i] = cate_list[i].split(' ')[-1]
            new_info = {'categories': [cate_list[0]],
                        'title': correct_title(title)}
            datas[iid] = new_info
    meta_set = set(datas.keys())
    return datas, item_asins.difference(meta_set)


# categories and brand is all attribute
def get_attribute_ml(meta_infos, datamaps, user_infos):
    attributes = defaultdict(int)
    for iid, info in meta_infos.items():
        for cate in info['categories']:
            # attributes[cates[1].strip()] += 1
            attributes[cate] += 1

    print(f'before delete, attribute num:{len(attributes)}')
    new_meta = {}
    for iid, info in meta_infos.items():
        new_meta[iid] = []
        for cate in info['categories']:
            new_meta[iid].append(cate)
        if len(new_meta[iid]) > 2:
            print(new_meta[iid], info['categories'])
    # mapping
    attribute2id = {}
    id2attribute = {}
    attributeid2num = defaultdict(int)
    attribute_id = 1
    items2attributes = {}
    attribute_lens = []
    itemid2title = {}

    for iid, attributes in new_meta.items():
        item_id = datamaps['item2id'][iid]
        items2attributes[item_id] = []
        for attribute in attributes:
            if attribute not in attribute2id:
                attribute2id[attribute] = attribute_id
                id2attribute[attribute_id] = attribute
                attribute_id += 1
            attributeid2num[attribute2id[attribute]] += 1
            items2attributes[item_id].append(attribute2id[attribute])
        attribute_lens.append(len(items2attributes[item_id]))
        itemid2title[item_id] = meta_infos[iid]['title']


    print(f'before delete, attribute num:{len(attribute2id)}')
    print(f'attributes len, Min:{np.min(attribute_lens)}, Max:{np.max(attribute_lens)}, '
          f'Avg.:{np.mean(attribute_lens):.4f}')
    # update datamap
    datamaps['attribute2id'] = attribute2id
    datamaps['id2attribute'] = id2attribute
    datamaps['user2attribute'] = user_infos
    datamaps['attributeid2num'] = attributeid2num
    datamaps['itemid2title'] = itemid2title
    datamaps['attribute_ft_num'] = 1

    return len(attribute2id), np.mean(attribute_lens), datamaps, items2attributes


def get_interaction(datas):  # return a dict, key is user and value is a list of items
    user_seq = {}
    for data in datas:
        user, item, time, rating = data
        if user in user_seq:
            user_seq[user].append((item, time, rating))
        else:
            user_seq[user] = []
            user_seq[user].append((item, time, rating))

    for user, item_time in user_seq.items():
        item_time.sort(key=lambda x: x[1])  # sorting by time
        items = []
        for t in item_time:
            items.append([t[0], t[2]])
        user_seq[user] = items
    return user_seq


# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    rating_count = defaultdict(int)
    for user, items in user_items.items():
        for item, rating in items:
            user_count[user] += 1
            item_count[item] += 1
            rating_count[rating] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, rating_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, rating_count, False
    return user_count, item_count, rating_count, True  # guarantee Kcore


# filter K-core
def filter_Kcore(user_items, user_core, item_core):
    user_count, item_count, rating_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:  # delete user
                user_items.pop(user)
            else:
                for item, rating in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove([item, rating])
        user_count, item_count, rating_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items


def id_map(user_items):  # user_items dict
    user2id = {}  # raw 2 uid
    item2id = {}  # raw 2 iid
    id2user = {}  # uid 2 raw
    id2item = {}  # iid 2 raw
    user_id = 1
    item_id = 1
    final_data = {}
    lm_hist_idx = {}
    random_user_list = list(user_items.keys())
    random.shuffle(random_user_list)
    for user in random_user_list:
        items = user_items[user]
        if user not in user2id:
            user2id[user] = user_id
            id2user[user_id] = user
            user_id += 1
        iids = []  # item id lists
        ratings = []
        for item, rating in items:
            if item not in item2id:
                item2id[item] = item_id
                id2item[item_id] = item
                item_id += 1
            iids.append(item2id[item])
            ratings.append(rating)
        uid = user2id[user]
        lm_hist_idx[uid] = min((len(iids) + 1) // 2, lm_hist_max)
        final_data[uid] = [iids, ratings]
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
    }
    return final_data, user_id - 1, item_id - 1, data_maps, lm_hist_idx


def update_data(user_items, item_diff, id2item):
    new_data = {}
    lm_hist_idx = {}
    for user, user_data in user_items.items():
        iids, ratings = user_data
        new_idds, new_ratings = [], []
        for id, rating in zip(iids, ratings):
            if id2item[id] not in item_diff:
                new_idds.append(id)
                new_ratings.append(rating)
        new_data[user] = [new_idds, new_ratings]
        lm_hist_idx[user] = min((len(iids) + 1) // 2, lm_hist_max)
        # item_num += len(new_idds)
    item_num = len(id2item) - len(item_diff)
    return new_data, item_num, lm_hist_idx


def preprocess(data_file, meta_file, user_file, processed_dir, data_type='ml-1m'):
    assert data_type in {'ml-1m'}

    datas = ml_1m(data_file, rating_score=rating_score)

    user_items = get_interaction(datas)
    print(f'{data_file} Raw data has been processed! Lower than {rating_score} are deleted!')
    # raw_id user: [item1, item2, item3...]
    if item_core > 0 or user_core > 0:
        user_items = filter_Kcore(user_items, user_core=user_core, item_core=item_core)
        print(f'User {user_core}-core complete! Item {item_core}-core complete!')

    user_count, item_count, rating_count, _ = check_Kcore(user_items, user_core=user_core,
                                                          item_core=item_core)  ## user_count: number of interaction for each user
    user_items, user_num, item_num, data_maps, lm_hist_idx = id_map(user_items)

    print('get meta infos')
    meta_infos, item_diff = ml_1m_meta(meta_file, data_maps)
    user_datas = ml_1m_user(user_file)
    if item_diff:
        print('diff num', len(item_diff))
        user_items, item_num, lm_hist_idx = update_data(user_items, item_diff, data_maps['id2item'])
    else:
        print('no different item num')
    '''
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
        'lm_hist_idx': lm_hist_idx
    }
    '''
    user_count_list = list(user_count.values())
    user_avg, user_min, user_max = np.mean(user_count_list), np.min(user_count_list), np.max(user_count_list)
    item_count_list = list(item_count.values())
    item_avg, item_min, item_max = np.mean(item_count_list), np.min(item_count_list), np.max(item_count_list)
    # rating_count_list = list(rating_count.values())
    interact_num = np.sum([x for x in user_count_list])
    sparsity = (1 - interact_num / (user_num * item_num)) * 100

    show_info = f'Total User: {user_num}, Avg User: {user_avg:.4f}, Min Len: {user_min}, Max Len: {user_max}\n' + \
                f'Total Item: {item_num}, Avg Item: {item_avg:.4f}, Min Inter: {item_min}, Max Inter: {item_max}\n' + \
                f'Iteraction Num: {interact_num}, Sparsity: {sparsity:.2f}%'
    print(show_info)
    print(rating_count, (rating_count[4] + rating_count[5]) / sum(list(rating_count.values())))
    print((rating_count[5]) / sum(list(rating_count.values())))

    # train/test split
    user_set = list(user_items.keys())
    random.shuffle(user_set)
    train_user = user_set[:int(len(user_set) * train_ratio)]
    test_user = user_set[int(len(user_set) * train_ratio):]
    train_test_split = {
        'train': train_user,
        'test': test_user,
        'lm_hist_idx': lm_hist_idx
    }
    print('user items sample:', user_items[user_set[0]])

    print('Begin extracting meta infos...')

    attribute_num, avg_attribute, datamaps, item2attributes = get_attribute_ml(meta_infos, data_maps, user_datas)
    for uid, items in user_items.items():
        item, rating = items
        for i in item:
            if i not in item2attributes:
                print('not in', i)

    sample_item = list(datamaps['itemid2title'].items())[:20]
    print('itemid2title sample')
    for itemid, title in sample_item:
        cate = item2attributes[itemid]
        print('Title:', title)
        print('Category:', datamaps['id2attribute'][cate[0]])


    print(f'{meta_file} & {add_comma(user_num)}& {add_comma(item_num)} & {user_avg:.1f}'
          f'& {item_avg:.1f}& {add_comma(interact_num)}& {sparsity:.2f}\%&{add_comma(attribute_num)}&'
          f'{avg_attribute:.1f} \\')

    # -------------- Save Data ---------------
    save_data_file = processed_dir + '/sequential_data.json'  # interaction sequence between user and item
    item2attributes_file = processed_dir + '/item2attributes.json'  # item and corresponding attributes
    datamaps_file = processed_dir + '/datamaps.json'  # datamap
    split_file = processed_dir + '/train_test_split.json'  # train/test splitting
    '''
    data_maps = {
        'user2id': user2id,
        'item2id': item2id,
        'id2user': id2user,
        'id2item': id2item,
        datamaps['attribute2id'] = attribute2id
        datamaps['id2attribute'] = id2attribute
        datamaps['attributeid2num'] = attributeid2num
        datamaps['itemid2title'] = itemid2title
    }
    '''

    save_json(user_items, save_data_file)
    save_json(item2attributes, item2attributes_file)
    save_json(datamaps, datamaps_file)
    save_json(train_test_split, split_file)


if __name__ == '__main__':
    set_seed(1234)
    DATA_DIR = '../data/'
    DATA_SET_NAME = 'ml-1m'
    DATA_FILE = os.path.join(DATA_DIR, DATA_SET_NAME + '/raw_data/ratings.dat')
    META_FILE = os.path.join(DATA_DIR, DATA_SET_NAME + '/raw_data/movies.dat')
    USER_FILE = os.path.join(DATA_DIR, DATA_SET_NAME + '/raw_data/users.dat')
    PROCESSED_DIR = os.path.join(DATA_DIR, DATA_SET_NAME, 'proc_data')

    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    preprocess(DATA_FILE, META_FILE, USER_FILE, PROCESSED_DIR, data_type='ml-1m')

