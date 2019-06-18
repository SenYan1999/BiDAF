import os
import json
from tqdm import tqdm
import ujson as json
import numpy as np
import spacy
import config
from utils import SQuADData
import torch

def tokenize(sentents):
    sents = nlp(sentents)
    return [token.text for token in sents]


def get_span(tokens, context):
    current = 0
    span = []
    for token in tokens:
        current = context.find(token, current)
        span.append((current, current + len(token)))
        current += len(token)
    return span


def get_data(file):
    with open(file) as f:
        raw_data = json.load(f)

    total = 0
    raw_data = raw_data['data']
    examples = []
    examples_eval = {}
    for data in tqdm(raw_data):
        for para in data['paragraphs']:
            context = para['context'].replace('"', "'").replace('``', '"')
            context_tokens = tokenize(context)
            context_span = get_span(context_tokens, context)
            context_chars = [list(token) for token in context_tokens]
            for qa in para['qas']:
                total += 1
                question = qa['question'].replace('"', "'").replace('``', '"')
                question_tokens = tokenize(question)
                question_chars = [list(token) for token in question_tokens]
                answer_texts = []
                for ans in qa['answers']:
                    ans_text = ans['text']
                    answer_texts.append(ans_text)
                    ans_start = ans['answer_start']
                    ans_end = ans_start + len(ans_text)
                    ans_span = []
                    for idx, span in enumerate(context_span):
                        if not (ans_start >= span[1] or ans_end <= span[0]):
                            ans_span.append(idx)
                    y1s = ans_span[0]
                    y2s = ans_span[-1]
                    example = {'context_tokens': context_tokens, 'context_chars': context_chars,'question_tokens':question_tokens,
                               'question_chars': question_chars,'y1s': y1s, 'y2s': y2s, 'uuid': total}
                    examples.append(example)
                    examples_eval[str(total)] = {'context': context, 'question': question,
                                                 'answer': answer_texts, 'context_span': context_span}
                    break
    return (examples, examples_eval)


def get_embedding(emb_file):
    embedding_dict = {}
    with open(emb_file) as f:
        for line in tqdm(f):
            array = line.split(' ')
            word = array[0]
            vector = list(map(float, array[1: ]))
            embedding_dict[word] = vector

    token2idx = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx['<pad>'] = 0
    token2idx['<unk>'] = 1
    embedding_dict['<pad>'] = [0. for _ in range(300)]
    embedding_dict['<unk>'] = [np.random.normal(scale=0.1) for _ in range(300)]
    idx2embed = {idx: embedding_dict[token] for token, idx in token2idx.items()}
    embedding_mat = [idx2embed[idx] for idx in range(len(idx2embed))]
    return embedding_mat, token2idx


def get_char2idx():
    char2idx = {}
    all_chars = '`1234567890-=qwertyuiop[]\\asdfghjkl;\'zxcvbnm,./~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:"ZXCVBNM<>?'
    for i, c in enumerate(all_chars):
        char2idx[c] = i + 2
    char2idx['<pad>'] = 0
    char2idx['<unk>'] = 1
    return char2idx


def get_input(example, word2idx, char2idx):
    para_limit = config.para_limit
    ques_limit = config.ques_limit
    char_limit = config.char_limit

    context_ids = np.zeros([para_limit], dtype=np.int32)
    context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
    question_idx = np.zeros([ques_limit], dtype=np.int32)
    question_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx:
                return word2idx[each]
        return 1

    def _get_char(char):
        if char in char2idx:
            return char2idx[char]
        return 1

    for i, token in enumerate(example['context_tokens']):
        if i == para_limit:
            break
        context_ids[i] = _get_word(token)
    for i, token in enumerate(example['context_chars']):
        if i == para_limit:
            break
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idx[i][j] = _get_char(char)
    for i, token in enumerate(example['question_tokens']):
        if i == ques_limit:
            break
        question_idx[i] = _get_word(token)
    for i, token in enumerate(example['question_chars']):
        if i == ques_limit:
            break
        for j, char in enumerate(token):
            if j == char_limit:
                break
            question_char_idx[i][j] = _get_char(char)
    return context_ids, context_char_idx, question_idx, question_char_idx


def get_final_data(data, word2idx, char2idx):
    context_tokens, context_chars, question_tokens, question_chars = [], [], [], []
    y1s, y2s = [], []
    ids = []
    for example in tqdm(data):
        context_token, context_char, question_token, question_char = get_input(example, word2idx, char2idx)
        context_tokens.append(context_token)
        context_chars.append(context_char)
        question_tokens.append(question_token)
        question_chars.append(question_char)
        y1s.append(example['y1s'] if example['y1s'] < 400 else 399)
        y2s.append(example['y2s'] if example['y2s'] < 400 else 399)
        ids.append(example['uuid'])
    return (torch.tensor(context_tokens), torch.tensor(context_chars), torch.tensor(question_tokens),
            torch.tensor(question_chars), torch.tensor(y1s), torch.tensor(y2s), torch.tensor(ids))

def save(obj, file):
    with open(file, 'w') as f:
        json.dump(obj, f)


def load(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data


def save_input(data, dir):
    torch.save(data[0], os.path.join(dir, 'context_tokens.pt'))
    torch.save(data[1], os.path.join(dir, 'context_chars.pt'))
    torch.save(data[2], os.path.join(dir, 'question_tokens.pt'))
    torch.save(data[3], os.path.join(dir, 'question_chars.pt'))
    torch.save(data[4], os.path.join(dir, 'y1s.pt'))
    torch.save(data[5], os.path.join(dir, 'y2s.pt'))
    torch.save(data[6], os.path.join(dir, 'ids.pt'))

if __name__ == '__main__':
    data_path = '../data/squad/'
    embed_path = '../data/glove/'
    nlp = spacy.blank('en')
    data_dir = './pre_data'

    print('Begin converting raw data....')
    data_train = get_data(os.path.join(data_path, 'train-v2.0.json'))
    print('Done!!')
    print('Saving raw data to %s' % data_dir)
    save(data_train, os.path.join(data_dir, 'data_train_pre.json'))
    print('Done!!!')

    print('Begin converting raw data....')
    data_dev = get_data(os.path.join(data_path, 'dev-v2.0.json'))
    print('Done!!')
    print('Saving raw data to %s' % data_dir)
    save(data_dev, os.path.join(data_dir, 'data_dev_pre.json'))
    print('Done!!!')

    # print('Begin converting embedding...')
    # embed_file = os.path.join(embed_path, 'glove.6B.300d.txt')
    # print(embed_file)
    # embedding = get_embedding(embed_file)
    # save(embedding, os.path.join(data_dir, 'embed_pre.json'))
    # print('Done!!!')

    print('Begin converting raw data to input data')
    word2idx = load('pre_data/embed_pre.json')[1]
    # data_train = load('pre_data/data_train_pre.json')
    # data_dev = load('pre_data/data_dev_pre.json')
    # word2idx = embedding[1]
    char2idx = get_char2idx()
    final_train = get_final_data(data_train[0], word2idx, char2idx)
    save_input(final_train, 'pre_data/input/train')
    final_dev = get_final_data(data_dev[0], word2idx, char2idx)
    save_input(final_dev, 'pre_data/input/dev')
    print('Done!!')

    print('Get SQuAD Dataset')
    train_dataset = SQuADData(os.path.join(data_dir, 'data_train.json'))
    dev_dataset = SQuADData(os.path.join(data_dir, 'data_dev.json'))
    save(train_dataset, os.path.join(data_dir, 'train_dataset.json'))
    save(dev_dataset, os.path.join(data_dir, 'dev_dataset.json'))
    print('Done!')
