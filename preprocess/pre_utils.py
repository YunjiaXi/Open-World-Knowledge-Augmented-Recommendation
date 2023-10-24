import os
import random
import numpy as np
import json
import pickle
import gzip


AGE_MAPPING = {
    1: "under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+"
}

GENDER_MAPPING = {'F': 'female', 'M': 'male'}

OCCUPATION_MAPPING = {
    0: "in an unknown occupation",
    1: "an academic/educator",
    2: "an artist",
    3: "in clerical/admin department",
    4: "a college/grad student",
    5: "a customer service staff",
    6: "a doctor/health care",
    7: "an executive/managerial",
    8: "a farmer",
    9: "a homemaker",
    10: "a K-12 student",
    11: "a lawyer",
    12: "a programmer",
    13: "retired",
    14: "in sales/marketing department",
    15: "a scientist",
    16: "self-employed",
    17: "a technician/engineer",
    18: "a tradesman/craftsman",
    19: "unemployed",
    20: "a writer",
}


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w') as w:
        json.dump(data, w)
    # json_str = json.dumps(data)
    # with open(file_path, 'w') as out:
    #     out.write(json_str)


def ReadLineFromFile(path):
    lines = []
    with open(path, 'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def add_comma(num):
    # 1000000 -> 1,000,000
    str_num = str(num)
    res_num = ''
    for i in range(len(str_num)):
        res_num += str_num[i]
        if (len(str_num) - i - 1) % 3 == 0:
            res_num += ','
    return res_num[:-1]

def correct_title(title):
    try:
        pure_title, time = title.strip().split('(')
    except:
        return title
    spl_list = pure_title.strip().split(',')
    # spl_list = [word.strip() for word in spl_list]
    last_word = spl_list[-1].strip().lower()
    if last_word == 'the' or last_word == 'a':
        tmp = ','.join(spl_list[:-1])
        title = spl_list[-1].strip() + ' ' + tmp + ' (' + time
    return title
