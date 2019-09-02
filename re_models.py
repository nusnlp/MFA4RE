import sys
import os
import numpy as np
import random

from collections import OrderedDict
import collections
import json
import pickle
import math
import datetime
from tqdm import tqdm
from recordclass import recordclass

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.backends.cudnn.deterministic = True


def custom_print(*msg):
    for i in range(0, len(msg)):
        if i == len(msg) - 1:
            print(msg[i])
            logger.write(str(msg[i]) + '\n')
        else:
            print(msg[i], ' ', end='')
            logger.write(str(msg[i]))


def load_word_embedding(embed_file, vocab):
    custom_print('vocab length:', len(vocab))
    # custom_print('entity vocab length', len(entity_vocab))
    embed_vocab = OrderedDict()
    embed_matrix = list()
    embed_vocab['<PAD>'] = 0
    embed_matrix.append(np.zeros(word_embed_dim, dtype=np.float32))
    embed_vocab['<UNK>'] = 1
    embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
    word_idx = 2

    with open(embed_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < word_embed_dim + 1:
                continue
            word = parts[0]
            if word in vocab and vocab[word] >= word_density:
                    vec = [np.float32(val) for val in parts[1:]]
                    embed_matrix.append(vec)
                    embed_vocab[word] = word_idx
                    word_idx += 1

    custom_print('embed vocab length:', len(embed_vocab))

    for word in vocab:
        if word not in embed_vocab and vocab[word] >= word_density:
            # custom_print(word)
            embed_matrix.append(np.random.uniform(-0.25, 0.25, word_embed_dim))
            embed_vocab[word] = word_idx
            word_idx += 1

    custom_print('embed vocab length:', len(embed_vocab))
    return embed_vocab, np.array(embed_matrix, dtype=np.float32)


def build_vocab(train, dev, test, vocab_file, embedding_file):
    vocab = OrderedDict()

    for d in train + dev + test:
        for word in d.Text.split():
            word = word.strip()
            if len(word) > 0:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

    embed_vocab, embed_matrix = load_word_embedding(embedding_file, vocab)
    output = open(vocab_file, 'wb')
    pickle.dump(embed_vocab, output)
    output.close()
    return embed_vocab, embed_matrix


def load_vocab(vocab_file):
    with open(vocab_file, 'rb') as f:
        embed_vocab = pickle.load(f)
    return embed_vocab


def get_sample(uid, Id, sent, dep_data, arg1, arg2, arg1_start, arg1_end, arg2_start, arg2_end, rel_name):
    sent = sent.strip()
    words = sent.split()
    norm_words = words
    norm_words.append('<PAD>')
    norm_words.append('<PAD>')
    norm_words.append('<PAD>')
    norm_words.append('<PAD>')
    norm_words.append('<PAD>')

    arg1_mask = list()
    arg2_mask = list()
    piece1_mask = list()
    piece2_mask = list()
    piece3_mask = list()
    words_mask = list()

    for i in range(0, len(norm_words)):
        arg1_mask.append(0)
        arg2_mask.append(0)
        piece1_mask.append(0)
        piece2_mask.append(0)
        piece3_mask.append(0)
        words_mask.append(0)

    for i in range(0, len(words)):
        words_mask[i] = 1
    first_cut_idx = arg1_end
    second_cut_idx = arg2_end
    if arg2_start < arg1_start:
        first_cut_idx, second_cut_idx = second_cut_idx, first_cut_idx

    for i in range(0, first_cut_idx + 1):
        piece1_mask[i] = 1
    for i in range(first_cut_idx, second_cut_idx+1):
        piece2_mask[i] = 1
    for i in range(second_cut_idx, len(words)):
        piece3_mask[i] = 1

    entity_indicator = list()
    for ind in range(0, len(words)):
        entity_indicator.append(1)
    for ind in range(arg1_start, arg1_end+1):
        entity_indicator[ind] = 2
    for ind in range(arg2_start, arg2_end+1):
        entity_indicator[ind] = 3

    arg1_dep_dist = get_dep_dist(dep_data, arg1)
    arg2_dep_dist = get_dep_dist(dep_data, arg1)
    arg_dep_dist = list()

    for i in range(0, len(arg1_dep_dist)):
        arg1_dep_dist[i] += 1
        arg2_dep_dist[i] += 1
        arg_dep_dist.append(int(math.floor((arg1_dep_dist[i] + arg2_dep_dist[i]) / 2)))

    for i in range(0, len(arg1_dep_dist)):
        if arg1_start <= i <= arg1_end:
            arg1_dep_dist[i] = dep_win_size_rel
            arg_dep_dist[i] = dep_win_size_rel

        if arg2_start <= i <= arg2_end:
            arg2_dep_dist[i] = dep_win_size_rel
            arg_dep_dist[i] = dep_win_size_rel

    arg1_head_dist_lst = list()
    arg2_head_dist_lst = list()
    for ind in range(0, len(norm_words)):
        dist = arg1_start - ind
        if dist >= 0:
            dist += 1
            dist = min(dist, max_word_arg_head_dist)
        else:
            dist *= -1
            dist = min(dist, max_word_arg_head_dist)
            dist += max_word_arg_head_dist
        arg1_head_dist_lst.append(dist)
        dist = arg2_start - ind
        if dist >= 0:
            dist += 1
            dist = min(dist, max_word_arg_head_dist)
        else:
            dist *= -1
            dist = min(dist, max_word_arg_head_dist)
            dist += max_word_arg_head_dist
        arg2_head_dist_lst.append(dist)

    for ind in range(arg1_start, arg1_end + 1):
        arg1_head_dist_lst[ind] = 1
    for ind in range(arg2_start, arg2_end + 1):
        arg2_head_dist_lst[ind] = 1

    arg1_ctx_start = max(0, arg1_start - ctx_len)
    arg1_ctx_end = min(len(words)-1, arg1_end + ctx_len)

    arg2_ctx_start = max(0, arg2_start - ctx_len)
    arg2_ctx_end = min(len(words) - 1, arg2_end + ctx_len)

    for i in range(arg1_ctx_start, arg1_ctx_end + 1):
        arg1_mask[i] = 1
    for i in range(arg2_ctx_start, arg2_ctx_end + 1):
        arg2_mask[i] = 1

    sample = QASample(UID=uid, Id=Id, Len=len(norm_words), Text=sent, Arg1=arg1, Arg2=arg2,
                      Words=norm_words, Words_Mask=words_mask, WordsEntIndicator=entity_indicator,
                      WordsArg1Dist=arg1_head_dist_lst, WordsArg1DepDist=arg1_dep_dist,
                      WordsArg2Dist=arg2_head_dist_lst, WordsArg2DepDist=arg2_dep_dist,
                      WordsDepDist=arg_dep_dist, Arg1_Mask=arg1_mask, Arg2_Mask=arg2_mask,
                      Piece1Mask=piece1_mask, Piece2Mask=piece2_mask, Piece3Mask=piece3_mask,
                      RelationName=rel_name)
    return sample


def get_dep_dist(dep_data, ent):
    for e in dep_data:
        if e['text'] == ent:
            return e['dep_dist']
    return []


def get_data(lines, dep_lines, is_training_data=False):
    custom_print(len(lines))
    samples = []
    uid = 1
    cnt = 0
    out_of_len = 0
    dep_error = 0
    for i in range(0, len(lines)):
        line = lines[i].strip()
        dep_line = dep_lines[i].strip()
        data = json.loads(line)
        sent = data['sentText']
        if is_training_data and len(sent.split()) > max_sent_len:
            out_of_len += 1
            continue
        dep_data = json.loads(dep_line)
        for rel_mention in data['relationMentions']:
            arg1 = rel_mention['arg1Text']
            arg2 = rel_mention['arg2Text']
            rel_name = rel_mention['relationName']

            if is_training_data and rel_name not in relation_cls_label_map:
                continue
            if rel_name not in relation_cls_label_map:
                cnt += 1

            sample = get_sample(uid, data['sentId'], sent, dep_data, arg1, arg2,
                                int(rel_mention['arg1StartIndex']), int(rel_mention['arg1EndIndex']),
                                int(rel_mention['arg2StartIndex']), int(rel_mention['arg2EndIndex']),
                                rel_name)
            valid = False
            for dist in sample.WordsDepDist:
                if 1 <= dist <= dep_win_size_rel:
                    valid = True
                    break
            if valid:
                samples.append(sample)
                uid += 1
            else:
                dep_error += 1
    print(dep_error)
    custom_print(cnt)
    if is_training_data:
        custom_print(out_of_len)
    return samples


def read_data(file_path, dep_file, is_training_data=False):
    reader = open(file_path)
    lines = reader.readlines()
    reader.close()

    reader = open(dep_file)
    dep_lines = reader.readlines()
    reader.close()

    # lines = lines[0:min(1000, len(lines))]
    # dep_lines = dep_lines[0:min(1000, len(dep_lines))]

    data = get_data(lines, dep_lines, is_training_data)
    return data


def get_threshold(data, preds):
    max_f1 = -1.0
    best_th = -1.0
    cur_th = 0.0
    while cur_th < 1.0:
        pred_pos, gt_pos, correct_pos = get_F1(data, preds, th=cur_th)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        cur_f1 = (2 * p * r) / (p + r + 1e-8)
        if cur_f1 > max_f1:
            max_f1 = cur_f1
            best_th = cur_th
        cur_th += 0.01
    return best_th


def get_F1(data, preds, th=0.0):
    gt_pos = 0
    pred_pos = 0
    correct_pos = 0
    for i in range(0, len(data)):
        org_rel_name = data[i].RelationName
        pred_val = np.argmax(preds[i])
        pred_rel_name = list(relation_cls_label_map)[pred_val]
        if org_rel_name not in ignore_rel_list:
            gt_pos += 1
        if pred_rel_name not in ignore_rel_list and np.max(preds[i]) > th:
            pred_pos += 1
        if org_rel_name == pred_rel_name and org_rel_name not in ignore_rel_list and np.max(preds[i]) > th:
            correct_pos += 1
    return pred_pos, gt_pos, correct_pos


def write_PR_curve(data, preds, file_name):
    cur_th = 0.0
    writer = open(file_name, 'w')
    writer.write('Threshold,Prec.,Rec.,F1\n')
    while cur_th < 1.001:
        pred_pos, gt_pos, correct_pos = get_F1(data, preds, th=cur_th)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        cur_f1 = (2 * p * r) / (p + r + 1e-8)
        writer.write(','.join([str(cur_th)[:6], str(p)[:6], str(r)[:6], str(cur_f1)[:6]]) + '\n')
        cur_th += 0.001


def pr_curve(infile, outfile):
    reader = open(infile)
    lines = reader.readlines()
    reader.close()
    pr_lst = list()
    for line in lines[1:]:
        parts = line.strip().split(',')
        p = float(parts[1])
        r = float(parts[2])
        pr_lst.append((p, r))
    eps = 0.01
    standard_pr_lst = []

    r = 0.0
    while r <= 1.0:
        p = 0
        min_diff = 10.0
        for i in range(0, len(pr_lst)):
            if abs(pr_lst[i][1] - r) < eps and abs(pr_lst[i][1] - r) < min_diff:
                min_diff = abs(pr_lst[i][1] - r)
                p = pr_lst[i][0]
        if p != 0.0:
            standard_pr_lst.append((p, r))
        r += 0.01
    writer = open(outfile, 'w')
    writer.write('Rec.,Prec.\n')
    for pair in standard_pr_lst:
        writer.write(','.join([str(pair[1])[:6], str(pair[0])]) + '\n')
    writer.close()


def cal_auc(infile, min_p, min_r):
    reader = open(infile)
    lines = reader.readlines()
    reader.close()
    pr_lst = []
    for line in lines[1:]:
        parts = line.strip().split(',')
        p = float(parts[1])
        r = float(parts[0])
        if p >= min_p and r >= min_r:
            pr_lst.append((p, r))
    auc = 0.0
    for i in range(0, len(pr_lst) - 1):
        auc += 0.5 * (pr_lst[i][0] + pr_lst[i+1][0]) * (pr_lst[i+1][1] - pr_lst[i][1])
    custom_print('AUC:', round(auc, 3))


def write_pred_file(data, preds, out_file, threshold=0.0):
    # custom_print('writing relation output...')
    # custom_print(len(data))
    # custom_print(len(preds))
    writer = open(out_file, 'w')
    out_dct = OrderedDict()
    sent_dct = OrderedDict()
    for i in range(0, len(data)):
        pred_val = np.argmax(preds[i])
        pred_score = np.max(preds[i])
        rm = OrderedDict()
        rm['arg1Text'] = data[i].Arg1
        # rm['arg1Start'] = data[i].Arg1Start
        rm['arg2Text'] = data[i].Arg2
        # rm['arg2Start'] = data[i].Arg2Start
        rm['predictedRelationName'] = list(relation_cls_label_map)[pred_val]
        if pred_score <= threshold:
            rm['predictedRelationName'] = 'None'
        rm['confidence'] = str(np.max(preds[i]))
        rm['relationName'] = data[i].RelationName
        sentId = data[i].Id
        sent_dct[sentId] = data[i].Text
        if sentId not in out_dct:
            out_dct[sentId] = [rm]
        else:
            out_dct[sentId].append(rm)

    # custom_print(len(out_dct))
    for sentId in out_dct:
        data = OrderedDict()
        data['sentId'] = sentId
        data['sentText'] = sent_dct[sentId]
        data['predictedRelationMentions'] = out_dct[sentId]
        writer.write(json.dumps(data) + '\n')
    writer.close()


def shuffle_data(data):
    custom_print(len(data))
    data.sort(key=lambda x: x.Len)
    num_batch = int(len(data) / batch_size)
    rand_idx = random.sample(range(num_batch), num_batch)
    new_data = []
    for idx in rand_idx:
        new_data += data[batch_size * idx: batch_size * (idx + 1)]
    if len(new_data) < len(data):
        new_data += data[num_batch * batch_size:]
    return new_data


def get_class_label_map(rel_file):
    cls_label_map = collections.OrderedDict()
    label_cls_map = collections.OrderedDict()
    reader = open(rel_file)
    lines = reader.readlines()
    reader.close()

    label = 0
    for line in lines:
        line = line.strip()
        cls_label_map[line] = label
        label_cls_map[label] = line
        label += 1
    return cls_label_map, label_cls_map


def get_rel_counts(data):
    rel_cnt = OrderedDict()
    for rel_name in relation_cls_label_map:
        rel_cnt[rel_name] = 0
    for d in data:
        rel_cnt[d.RelationName] += 1

    return rel_cnt


def get_max_len(sample_batch):
    max_len = len(sample_batch[0].Words)
    for idx in range(1, len(sample_batch)):
        if len(sample_batch[idx].Words) > max_len:
            max_len = len(sample_batch[idx].Words)
    return max_len


def get_ent_indicator_seq(indicator, max_len):
    indicator_seq = list()
    for ind in range(0, len(indicator)):
        indicator_seq.append(indicator[ind])
    pad_len = max_len - len(indicator)
    for i in range(0, pad_len):
        indicator_seq.append(0)
    return indicator_seq


def get_distance_seq(dist, max_len):
    head_dist_seq = list()
    for ind in range(0, len(dist)):
        head_dist_seq.append(dist[ind])
    pad_len = max_len - len(dist)
    for i in range(0, pad_len):
        head_dist_seq.append(0)
    return head_dist_seq


def get_dep_distance(dist, max_len, win_size):
    mask_seq = list()
    dist_seq = list()
    for i in range(0, len(dist)):
        if use_dep_dist:
            cur_win = win_size + 1
            if 1 <= dist[i] <= win_size:
                cur_win = dist[i]
                mask_seq.append(0)
            else:
                mask_seq.append(1)
            dist_seq.append(float(1 / math.pow(2, cur_win - 1)))
        else:
            dist_seq.append(1.0)
            mask_seq.append(0)
    pad_len = max_len - len(dist)
    for i in range(0, pad_len):
        mask_seq.append(1)
        dist_seq.append(0.0)
    return dist_seq, mask_seq


def get_words_index_seq(path_words, max_len):
    path_seq = list()
    for word in path_words:
        if word in word_vocab:
            path_seq.append(word_vocab[word])
        else:
            path_seq.append(word_vocab['<UNK>'])
    pad_len = max_len - len(path_words)
    for i in range(0, pad_len):
        path_seq.append(word_vocab['<PAD>'])
    return path_seq


def get_padded_mask(mask, max_len):
    mask_seq = list()
    for i in range(0, len(mask)):
        mask_seq.append(mask[i])
    pad_len = max_len - len(mask)
    for i in range(0, pad_len):
        mask_seq.append(0)
    return mask_seq


def get_batch_data(cur_samples, is_training=False):
    """
    Returns the training samples and labels as numpy array
    """
    max_len = get_max_len(cur_samples)

    words_list = list()
    words_mask_list = list()
    words_arg1_dist_list = list()
    words_arg2_dist_list = list()
    words_ent_ind_list = list()

    arg1_list = list()
    arg2_list = list()

    arg1_mask_list = list()
    arg2_mask_list = list()
    piece1mask_list = list()
    piece2mask_list = list()
    piece3mask_list = list()

    arg1_dep_dist_list = list()
    arg1_dep_dist_mask = list()
    arg2_dep_dist_list = list()
    arg2_dep_dist_mask = list()
    arg_dep_dist_list = list()
    arg_dep_dist_mask = list()

    rel_labels_list = list()

    for sample in cur_samples:
        words_list.append(get_words_index_seq(sample.Words, max_len))
        words_arg1_dist_list.append(get_distance_seq(sample.WordsArg1Dist, max_len))
        words_arg2_dist_list.append(get_distance_seq(sample.WordsArg2Dist, max_len))
        words_ent_ind_list.append(get_ent_indicator_seq(sample.WordsEntIndicator, max_len))

        arg1_mask_list.append(get_padded_mask(sample.Arg1_Mask, max_len))
        arg2_mask_list.append(get_padded_mask(sample.Arg2_Mask, max_len))

        dep_dist, dep_dist_mask = get_dep_distance(sample.WordsArg1DepDist, max_len, dep_win_size_rel)
        arg1_dep_dist_list.append(dep_dist)
        arg1_dep_dist_mask.append(dep_dist_mask)

        dep_dist, dep_dist_mask = get_dep_distance(sample.WordsArg2DepDist, max_len, dep_win_size_rel)
        arg2_dep_dist_list.append(dep_dist)
        arg2_dep_dist_mask.append(dep_dist_mask)

        dep_dist, dep_dist_mask = get_dep_distance(sample.WordsDepDist, max_len, dep_win_size_rel)
        arg_dep_dist_list.append(dep_dist)
        arg_dep_dist_mask.append(dep_dist_mask)

        piece1mask_list.append(get_padded_mask(sample.Piece1Mask, max_len))
        piece2mask_list.append(get_padded_mask(sample.Piece2Mask, max_len))
        piece3mask_list.append(get_padded_mask(sample.Piece3Mask, max_len))
        words_mask_list.append(get_padded_mask(sample.Words_Mask, max_len))

        arg1_list.append(get_words_index_seq([sample.Arg1.split()[-1]], 1))
        arg2_list.append(get_words_index_seq([sample.Arg2.split()[-1]], 1))

        if is_training:
            rel_labels_list.append(relation_cls_label_map[sample.RelationName])

    return max_len, \
            {'words_input': np.array(words_list, dtype=np.float32),
             'words_mask_input': np.array(words_mask_list),
             'words_arg1_dist_input': np.array(words_arg1_dist_list),
             'words_arg2_dist_input': np.array(words_arg2_dist_list),
             'words_ent_indicator_input': np.array(words_ent_ind_list),
             'arg1_dep_dist': np.array(arg1_dep_dist_list),
             'arg2_dep_dist': np.array(arg2_dep_dist_list),
             'arg_dep_dist': np.array(arg_dep_dist_list),
             'arg1_dep_dist_mask': np.array(arg1_dep_dist_mask),
             'arg2_dep_dist_mask': np.array(arg2_dep_dist_mask),
             'arg_dep_dist_mask': np.array(arg_dep_dist_mask),
             'arg1_mask_input': np.array(arg1_mask_list),
             'arg2_mask_input': np.array(arg2_mask_list),
             'piece1mask_input': np.array(piece1mask_list),
             'piece2mask_input': np.array(piece2mask_list),
             'piece3mask_input': np.array(piece3mask_list),
             'arg1':np.array(arg1_list),
             'arg2': np.array(arg2_list)}, \
            {'relation_label': np.array(rel_labels_list, dtype=np.int32)}


# Models


class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.attn = nn.Parameter(torch.randn(input_dim, output_dim))
        stdv = 1. / math.sqrt(self.attn.size()[-1])
        self.attn.data.uniform_(-stdv, stdv)

    def forward(self, input, query, dep_dist, dep_mask, eps=1e-8):
        query = query.unsqueeze(2)
        sim = torch.matmul(input, self.attn)
        score = torch.bmm(sim, query).squeeze()
        score = torch.tanh(score)
        score = torch.exp(score)
        score = score * dep_dist

        if softmax_type == 0:
            score.data.masked_fill_(dep_mask.data, 0.0)
            score_sum = torch.sum(score, -1)
            score_sum = score_sum.unsqueeze(1)
            score /= score_sum
        else:
            score.data.masked_fill_(dep_mask.data, -float('inf'))
            score = nn.Softmax(dim=-1)(score)

        return score.unsqueeze(2)


class CNN_Layer(nn.Module):
    def __init__(self, input_dim, num_filter, kernel_size):
        super(CNN_Layer, self).__init__()
        self.tri_conv = nn.Conv1d(input_dim, num_filter, kernel_size, padding=1)

    def forward(self, input, mask):
        mask = mask.unsqueeze(2)
        input = torch.mul(input, mask)
        input = input.permute(0, 2, 1)
        output = self.tri_conv(input)
        output = F.tanh(torch.max(output, 2)[0])
        return output


class PCNN_Layer(nn.Module):
    def __init__(self, input_dim, num_filter):
        super(PCNN_Layer, self).__init__()
        self.tri_conv = nn.Conv1d(input_dim, num_filter, 3, padding=1)
        self.conv_non_linear = nn.Tanh()

    def forward(self, input, mask, p1mask, p2mask, p3mask):
        x1_mask = p1mask.byte()
        mask_copy = x1_mask.clone()
        x1_mask[mask_copy == 0] = 1
        x1_mask[mask_copy == 1] = 0

        x2_mask = p2mask.byte()
        mask_copy = x2_mask.clone()
        x2_mask[mask_copy == 0] = 1
        x2_mask[mask_copy == 1] = 0

        x3_mask = p3mask.byte()
        mask_copy = x3_mask.clone()
        x3_mask[mask_copy == 0] = 1
        x3_mask[mask_copy == 1] = 0

        mask = mask.unsqueeze(2)
        input = torch.mul(input, mask)
        input = input.permute(0, 2, 1)
        output = self.tri_conv(input)

        x1_mask = x1_mask.unsqueeze(1)  # .repeat(1, 1, self.input_dim)
        x2_mask = x2_mask.unsqueeze(1)  # .repeat(1, 1, self.input_dim)
        x3_mask = x3_mask.unsqueeze(1)  # .repeat(1, 1, self.input_dim)

        piece1 = output.clone()
        piece1.data.masked_fill_(x1_mask.data, -float('inf'))
        piece1 = self.conv_non_linear(torch.max(piece1, 2)[0])

        piece2 = output.clone()
        piece2.data.masked_fill_(x2_mask.data, -float('inf'))
        piece2 = self.conv_non_linear(torch.max(piece2, 2)[0])

        piece3 = output.clone()
        piece3.data.masked_fill_(x3_mask.data, -float('inf'))
        piece3 = self.conv_non_linear(torch.max(piece3, 2)[0])

        return torch.cat((piece1, piece2, piece3), 1)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.drop_rate = drop_out
        self.num_filter = conv_filter_cnt
        self.class_count = len(relation_cls_label_map)
        self.input_dim = word_embed_dim + 2 * distance_embed_dim

        self.word_embeddings = nn.Embedding(len(word_vocab), word_embed_dim, padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))
        self.word_arg1_head_distance_embed = nn.Embedding(dist_vocab_size, distance_embed_dim, padding_idx=0)
        self.word_arg2_head_distance_embed = nn.Embedding(dist_vocab_size, distance_embed_dim, padding_idx=0)

        self.cnn_layer = CNN_Layer(self.input_dim, self.num_filter, 3)
        self.dense = nn.Linear(self.num_filter, self.class_count)
        self.dropout = nn.Dropout(self.drop_rate)
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward(self, words_seq, words_mask, words_arg1_dist_seq, words_arg2_dist_seq, is_training=False):
        word_embeds = self.word_embeddings(words_seq)
        dist1_embeds = self.word_arg1_head_distance_embed(words_arg1_dist_seq)
        dist2_embeds = self.word_arg2_head_distance_embed(words_arg2_dist_seq)

        if apply_embed_dropout:
            word_embeds = self.dropout(word_embeds)
            dist1_embeds = self.dropout(dist1_embeds)
            dist2_embeds = self.dropout(dist2_embeds)

        words_input = torch.cat((word_embeds, dist1_embeds, dist2_embeds), 2)
        words_output = self.cnn_layer(words_input, words_mask)
        words_output = self.dropout(words_output)
        probs = self.dense(words_output)
        if is_training:
            probs = self.logsoftmax(probs)
        else:
            probs = F.softmax(probs)
        return probs


class PCNN(nn.Module):
    def __init__(self):
        super(PCNN, self).__init__()
        self.drop_rate = drop_out
        self.num_filter = conv_filter_cnt
        self.class_count = len(relation_cls_label_map)
        self.input_dim = word_embed_dim + 2 * distance_embed_dim   # + ent_indicator_embed_dim

        self.word_embeddings = nn.Embedding(len(word_vocab), word_embed_dim, padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))

        self.word_arg1_head_distance_embed = nn.Embedding(dist_vocab_size, distance_embed_dim, padding_idx=0)
        self.word_arg2_head_distance_embed = nn.Embedding(dist_vocab_size, distance_embed_dim, padding_idx=0)

        self.pcnn_layer = PCNN_Layer(self.input_dim, self.num_filter)

        self.dense = nn.Linear(3 * self.num_filter, self.class_count)
        self.dropout = nn.Dropout(self.drop_rate)
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward(self, words_seq, words_mask, words_arg1_dist_seq, words_arg2_dist_seq, piece1_mask, piece2_mask,
                piece3_mask, is_training=False):
        word_embeds = self.word_embeddings(words_seq)
        dist1_embeds = self.word_arg1_head_distance_embed(words_arg1_dist_seq)
        dist2_embeds = self.word_arg2_head_distance_embed(words_arg2_dist_seq)
        if apply_embed_dropout:
            word_embeds = self.dropout(word_embeds)
            dist1_embeds = self.dropout(dist1_embeds)
            dist2_embeds = self.dropout(dist2_embeds)
        pcnn_input = torch.cat((word_embeds, dist1_embeds, dist2_embeds), 2)
        pcnn_output = self.pcnn_layer(pcnn_input, words_mask, piece1_mask, piece2_mask, piece3_mask)

        rel_probs = self.dense(self.dropout(pcnn_output))
        if is_training:
            rel_probs = self.logsoftmax(rel_probs)
        else:
            rel_probs = F.softmax(rel_probs)
        return rel_probs


class EA(nn.Module):
    def __init__(self):
        super(EA, self).__init__()
        self.drop_rate = drop_out
        self.num_filter = conv_filter_cnt
        self.class_count = len(relation_cls_label_map)
        self.input_dim = word_embed_dim + 2 * distance_embed_dim   # + ent_indicator_embed_dim
        self.hidden_dim = 2 * self.input_dim

        self.word_embeddings = nn.Embedding(len(word_vocab), word_embed_dim, padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))

        self.word_arg1_head_distance_embed = nn.Embedding(dist_vocab_size, distance_embed_dim, padding_idx=0)
        self.word_arg2_head_distance_embed = nn.Embedding(dist_vocab_size, distance_embed_dim, padding_idx=0)

        self.ent1_attn_a = nn.Linear(self.input_dim + word_embed_dim, self.input_dim + word_embed_dim, bias=False)
        self.ent1_attn_r = nn.Linear(self.input_dim + word_embed_dim, 1, bias=False)

        self.ent2_attn_a = nn.Linear(self.input_dim + word_embed_dim, self.input_dim + word_embed_dim, bias=False)
        self.ent2_attn_r = nn.Linear(self.input_dim + word_embed_dim, 1, bias=False)

        self.cnn = CNN_Layer(self.input_dim, self.num_filter, 3)

        self.dense = nn.Linear(self.num_filter + 2 * self.input_dim, self.class_count)
        self.dropout = nn.Dropout(self.drop_rate)
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward(self,
                words_seq, words_mask, words_arg1_dist_seq, words_arg2_dist_seq,
                piece1_mask, piece2_mask, piece3_mask, arg1, arg2, is_training=False):
        word_embeds = self.word_embeddings(words_seq)
        time_steps = word_embeds.size()[1]
        dist1_embeds = self.word_arg1_head_distance_embed(words_arg1_dist_seq)
        dist2_embeds = self.word_arg2_head_distance_embed(words_arg2_dist_seq)
        arg1_embeds = self.word_embeddings(arg1)
        arg2_embeds = self.word_embeddings(arg2)

        if apply_embed_dropout:
            word_embeds = self.dropout(word_embeds)
            dist1_embeds = self.dropout(dist1_embeds)
            dist2_embeds = self.dropout(dist2_embeds)
            arg1_embeds = self.dropout(arg1_embeds)
            arg2_embeds = self.dropout(arg2_embeds)

        arg1_embeds = arg1_embeds.squeeze().unsqueeze(1).repeat(1, time_steps, 1)
        arg2_embeds = arg2_embeds.squeeze().unsqueeze(1).repeat(1, time_steps, 1)

        x_mask = words_mask.byte()
        mask_copy = x_mask.clone()
        x_mask[mask_copy == 0] = 1
        x_mask[mask_copy == 1] = 0

        input = torch.cat((word_embeds, dist1_embeds, dist2_embeds), 2)
        cnn_output = self.cnn(input, words_mask)

        ent1_attn_input = torch.cat((input, arg1_embeds), 2)
        ent1_attn = F.tanh(self.ent1_attn_a(ent1_attn_input))
        ent1_attn = self.ent1_attn_r(ent1_attn).squeeze()
        ent1_attn.data.masked_fill_(x_mask.data, -float('inf'))
        ent1_attn = nn.Softmax()(ent1_attn).unsqueeze(1)
        ent1_attn_vecs = torch.bmm(ent1_attn, input).squeeze()

        ent2_attn_input = torch.cat((input, arg2_embeds), 2)
        ent2_attn = F.tanh(self.ent2_attn_a(ent2_attn_input))
        ent2_attn = self.ent2_attn_r(ent2_attn).squeeze()
        ent2_attn.data.masked_fill_(x_mask.data, -float('inf'))
        ent2_attn = nn.Softmax()(ent2_attn).unsqueeze(1)
        ent2_attn_vecs = torch.bmm(ent2_attn, input).squeeze()

        rel_probs = self.dense(self.dropout(torch.cat((cnn_output, ent1_attn_vecs, ent2_attn_vecs), 1)))
        if is_training:
            rel_probs = self.logsoftmax(rel_probs)
        else:
            rel_probs = F.softmax(rel_probs)
        return rel_probs


class BGWA(nn.Module):
    def __init__(self):
        super(BGWA, self).__init__()
        self.drop_rate = drop_out
        self.num_filter = conv_filter_cnt
        self.class_count = len(relation_cls_label_map)
        self.input_dim = word_embed_dim + 2 * distance_embed_dim   # + ent_indicator_embed_dim
        self.hidden_dim = 2 * self.input_dim

        self.word_embeddings = nn.Embedding(len(word_vocab), word_embed_dim, padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))

        self.word_arg1_head_distance_embed = nn.Embedding(dist_vocab_size, distance_embed_dim, padding_idx=0)
        self.word_arg2_head_distance_embed = nn.Embedding(dist_vocab_size, distance_embed_dim, padding_idx=0)

        self.gru = nn.GRU(self.input_dim, int(self.hidden_dim / lstm_direction), 1, batch_first=True,
                            dropout=self.drop_rate, bidirectional=True)

        self.word_attn_a = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.word_attn_r = nn.Linear(self.hidden_dim, 1, bias=False)

        self.wa_pcnn = PCNN_Layer(self.hidden_dim, self.num_filter)

        self.dense = nn.Linear(3 * self.num_filter, self.class_count)
        self.dropout = nn.Dropout(self.drop_rate)
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward(self,
                words_seq, words_mask, words_arg1_dist_seq, words_arg2_dist_seq,
                piece1_mask, piece2_mask, piece3_mask, is_training=False):
        word_embeds = self.word_embeddings(words_seq)
        batch_len = word_embeds.size()[0]
        dist1_embeds = self.word_arg1_head_distance_embed(words_arg1_dist_seq)
        dist2_embeds = self.word_arg2_head_distance_embed(words_arg2_dist_seq)

        if apply_embed_dropout:
            word_embeds = self.dropout(word_embeds)
            dist1_embeds = self.dropout(dist1_embeds)
            dist2_embeds = self.dropout(dist2_embeds)

        x_mask = words_mask.byte()
        mask_copy = x_mask.clone()
        x_mask[mask_copy == 0] = 1
        x_mask[mask_copy == 1] = 0

        gru_input = torch.cat((word_embeds, dist1_embeds, dist2_embeds), 2)
        h0 = autograd.Variable(torch.FloatTensor(torch.zeros(lstm_direction, batch_len,
                                                             int(self.hidden_dim / lstm_direction))))
        h0 = h0.cuda()

        gru_output, hc = self.gru(gru_input, h0)
        word_attn = F.tanh(self.word_attn_a(gru_output))
        word_attn = self.word_attn_r(word_attn).squeeze()
        word_attn.data.masked_fill_(x_mask.data, -float('inf'))
        word_attn = nn.Softmax()(word_attn).unsqueeze(2)
        attn_vecs = torch.mul(gru_output, word_attn)
        wa_output = self.wa_pcnn(attn_vecs, words_mask, piece1_mask, piece2_mask, piece3_mask)

        rel_probs = self.dense(self.dropout(wa_output))
        if is_training:
            rel_probs = self.logsoftmax(rel_probs)
        else:
            rel_probs = F.softmax(rel_probs)
        return rel_probs


class MFA(nn.Module):
    def __init__(self):
        super(MFA, self).__init__()
        self.drop_rate = drop_out
        self.num_filter = conv_filter_cnt
        self.class_count = len(relation_cls_label_map)
        self.input_dim = word_embed_dim + ent_indicator_embed_dim

        self.hidden_dim = 2 * self.input_dim

        self.word_embeddings = nn.Embedding(len(word_vocab), word_embed_dim, padding_idx=0)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(word_embed_matrix))

        self.ent_ind_embeddings = nn.Embedding(4, ent_indicator_embed_dim, padding_idx=0)
        self.word_arg1_head_distance_embed = nn.Embedding(dist_vocab_size, distance_embed_dim, padding_idx=0)
        self.word_arg2_head_distance_embed = nn.Embedding(dist_vocab_size, distance_embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(self.input_dim, int(self.hidden_dim/lstm_direction), 1, batch_first=True,
                            bidirectional=True)
        self.cnn = CNN_Layer(self.hidden_dim + 2 * distance_embed_dim, self.num_filter, 3)
        self.arg1_cnn = CNN_Layer(self.input_dim, self.num_filter, 3)
        self.arg2_cnn = CNN_Layer(self.input_dim, self.num_filter, 3)

        if mfc > 0:
            self.att1_e1 = Attention(self.hidden_dim, self.num_filter)
            self.att1_e2 = Attention(self.hidden_dim, self.num_filter)

            if mfc > 1:
                self.att2_e1 = Attention(self.hidden_dim, self.num_filter)
                self.att2_e2 = Attention(self.hidden_dim, self.num_filter)

            if mfc > 2:
                self.att3_e1 = Attention(self.hidden_dim, self.num_filter)
                self.att3_e2 = Attention(self.hidden_dim, self.num_filter)

            if mfc > 3:
                self.att4_e1 = Attention(self.hidden_dim, self.num_filter)
                self.att4_e2 = Attention(self.hidden_dim, self.num_filter)

            if mfc > 4:
                self.att5_e1 = Attention(self.hidden_dim, self.num_filter)
                self.att5_e2 = Attention(self.hidden_dim, self.num_filter)

        self.dense = nn.Linear(3 * self.num_filter + 2 * mfc * self.hidden_dim, self.class_count)
        self.dropout = nn.Dropout(self.drop_rate)
        self.logsoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()

    def forward(self,
                words_seq, words_mask, words_ent_ind_seq, words_arg1_dist_seq, words_arg2_dist_seq, arg_dep_dist,
                arg1_dep_dist, arg2_dep_dist, arg1_mask, arg2_mask, arg_dep_mask, arg1_dep_mask, arg2_dep_mask,
                is_training=False):
        word_embeds = self.word_embeddings(words_seq)
        batch_len = word_embeds.size()[0]
        indicator_embeds = self.ent_ind_embeddings(words_ent_ind_seq)
        dist1_embeds = self.word_arg1_head_distance_embed(words_arg1_dist_seq)
        dist2_embeds = self.word_arg2_head_distance_embed(words_arg2_dist_seq)

        if apply_embed_dropout:
            word_embeds = self.dropout(word_embeds)
            indicator_embeds = self.dropout(indicator_embeds)
            dist1_embeds = self.dropout(dist1_embeds)
            dist2_embeds = self.dropout(dist2_embeds)

        words_input = torch.cat((word_embeds, indicator_embeds), 2)

        h0 = autograd.Variable(torch.FloatTensor(torch.zeros(lstm_direction, batch_len,
                                                             int(self.hidden_dim/lstm_direction))))
        h0 = h0.cuda()
        c0 = autograd.Variable(torch.FloatTensor(torch.zeros(lstm_direction, batch_len,
                                                             int(self.hidden_dim/lstm_direction))))
        c0 = c0.cuda()
        words_output, hc = self.lstm(words_input, (h0, c0))

        cnn_input = torch.cat((words_output, dist1_embeds, dist2_embeds), 2)
        cnn_output = self.cnn(cnn_input, words_mask)

        arg_input = words_input
        arg1 = self.arg1_cnn(arg_input, arg1_mask)
        arg2 = self.arg2_cnn(arg_input, arg2_mask)
        feature = torch.cat((cnn_output, arg1, arg2), 1)

        if mfc > 0:
            att1_scores = self.att1_e1(words_output, arg1, arg1_dep_dist, arg_dep_mask)
            att1_scores = att1_scores.squeeze().unsqueeze(1)
            att1_out = torch.bmm(att1_scores, words_output).squeeze()
            feature = torch.cat((feature, att1_out), -1)

            att1_scores = self.att1_e2(words_output, arg2, arg2_dep_dist, arg_dep_mask)
            att1_scores = att1_scores.squeeze().unsqueeze(1)
            att1_out = torch.bmm(att1_scores, words_output).squeeze()
            feature = torch.cat((feature, att1_out), -1)

        if mfc > 1:
            att2_scores = self.att2_e1(words_output, arg1, arg1_dep_dist, arg_dep_mask)
            att2_scores = att2_scores.squeeze().unsqueeze(1)
            att2_out = torch.bmm(att2_scores, words_output).squeeze()
            feature = torch.cat((feature, att2_out), -1)

            att2_scores = self.att2_e2(words_output, arg2, arg2_dep_dist, arg_dep_mask)
            att2_scores = att2_scores.squeeze().unsqueeze(1)
            att2_out = torch.bmm(att2_scores, words_output).squeeze()
            feature = torch.cat((feature, att2_out), -1)

        if mfc > 2:
            att3_scores = self.att3_e1(words_output, arg1, arg1_dep_dist, arg_dep_mask)
            att3_scores = att3_scores.squeeze().unsqueeze(1)
            att3_out = torch.bmm(att3_scores, words_output).squeeze()
            feature = torch.cat((feature, att3_out), -1)

            att3_scores = self.att3_e2(words_output, arg2, arg2_dep_dist, arg_dep_mask)
            att3_scores = att3_scores.squeeze().unsqueeze(1)
            att3_out = torch.bmm(att3_scores, words_output).squeeze()
            feature = torch.cat((feature, att3_out), -1)

        if mfc > 3:
            att4_scores = self.att4_e1(words_output, arg1, arg1_dep_dist, arg_dep_mask)
            att4_scores = att4_scores.squeeze().unsqueeze(1)
            att4_out = torch.bmm(att4_scores, words_output).squeeze()
            feature = torch.cat((feature, att4_out), -1)

            att4_scores = self.att4_e2(words_output, arg2, arg2_dep_dist, arg_dep_mask)
            att4_scores = att4_scores.squeeze().unsqueeze(1)
            att4_out = torch.bmm(att4_scores, words_output).squeeze()
            feature = torch.cat((feature, att4_out), -1)

        if mfc > 4:
            att5_scores = self.att5_e1(words_output, arg1, arg1_dep_dist, arg_dep_mask)
            att5_scores = att5_scores.squeeze().unsqueeze(1)
            att5_out = torch.bmm(att5_scores, words_output).squeeze()
            feature = torch.cat((feature, att5_out), -1)

            att5_scores = self.att5_e2(words_output, arg2, arg2_dep_dist, arg_dep_mask)
            att5_scores = att5_scores.squeeze().unsqueeze(1)
            att5_out = torch.bmm(att5_scores, words_output).squeeze()
            feature = torch.cat((feature, att5_out), -1)

        probs = self.dense(self.dropout(feature))
        if is_training:
            probs = self.logsoftmax(probs)
        else:
            probs = self.softmax(probs)
        return probs


def get_model(model_id):
    if model_id == 1:
        return CNN()
    if model_id == 2:
        return PCNN()
    if model_id == 3:
        return EA()
    if model_id == 4:
        return BGWA()
    if model_id == 5:
        return MFA()


def predict(samples, model, model_id):
    pred_batch_size = batch_size
    batch_count = math.ceil(len(samples) / pred_batch_size)
    move_last_batch = False
    if len(samples) - pred_batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    preds = list()
    model.eval()
    np.random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    random.seed(random_seed)
    for batch_idx in tqdm(range(0, batch_count)):
        batch_start = batch_idx * pred_batch_size
        batch_end = min(len(samples), batch_start + pred_batch_size)
        if batch_idx == batch_count - 1 and move_last_batch:
            batch_end = len(samples)

        cur_batch = samples[batch_start:batch_end]
        cur_seq_len, cur_samples_input, cur_samples_target = get_batch_data(cur_batch)

        words_seq = torch.from_numpy(cur_samples_input['words_input'].astype('long'))
        words_mask = torch.from_numpy(cur_samples_input['words_mask_input'].astype('float32'))
        words_arg1_dist_seq = torch.from_numpy(cur_samples_input['words_arg1_dist_input'].astype('long'))
        words_arg2_dist_seq = torch.from_numpy(cur_samples_input['words_arg2_dist_input'].astype('long'))
        words_ent_ind_seq = torch.from_numpy(cur_samples_input['words_ent_indicator_input'].astype('long'))

        arg_dep_dist = torch.from_numpy(cur_samples_input['arg_dep_dist'].astype('float32'))
        arg_dep_dist_mask = torch.from_numpy(cur_samples_input['arg_dep_dist_mask'].astype('uint8'))
        arg1_dep_dist = torch.from_numpy(cur_samples_input['arg1_dep_dist'].astype('float32'))
        arg1_dep_dist_mask = torch.from_numpy(cur_samples_input['arg1_dep_dist_mask'].astype('uint8'))
        arg2_dep_dist = torch.from_numpy(cur_samples_input['arg2_dep_dist'].astype('float32'))
        arg2_dep_dist_mask = torch.from_numpy(cur_samples_input['arg2_dep_dist_mask'].astype('uint8'))

        arg1 = torch.from_numpy(cur_samples_input['arg1'].astype('long'))
        arg2 = torch.from_numpy(cur_samples_input['arg2'].astype('long'))
        arg1_mask = torch.from_numpy(cur_samples_input['arg1_mask_input'].astype('float32'))
        arg2_mask = torch.from_numpy(cur_samples_input['arg2_mask_input'].astype('float32'))

        piece1mask_seq = torch.from_numpy(cur_samples_input['piece1mask_input'].astype('float32'))
        piece2mask_seq = torch.from_numpy(cur_samples_input['piece2mask_input'].astype('float32'))
        piece3mask_seq = torch.from_numpy(cur_samples_input['piece3mask_input'].astype('float32'))

        if torch.cuda.is_available():
            words_seq = words_seq.cuda()
            words_mask = words_mask.cuda()
            words_arg1_dist_seq = words_arg1_dist_seq.cuda()
            words_arg2_dist_seq = words_arg2_dist_seq.cuda()
            words_ent_ind_seq = words_ent_ind_seq.cuda()

            arg_dep_dist = arg_dep_dist.cuda()
            arg_dep_dist_mask = arg_dep_dist_mask.cuda()
            arg1_dep_dist = arg1_dep_dist.cuda()
            arg1_dep_dist_mask = arg1_dep_dist_mask.cuda()
            arg2_dep_dist = arg2_dep_dist.cuda()
            arg2_dep_dist_mask = arg2_dep_dist_mask.cuda()

            arg1 = arg1.cuda()
            arg2 = arg2.cuda()
            arg1_mask = arg1_mask.cuda()
            arg2_mask = arg2_mask.cuda()

            piece1mask_seq = piece1mask_seq.cuda()
            piece2mask_seq = piece2mask_seq.cuda()
            piece3mask_seq = piece3mask_seq.cuda()

        words_seq = autograd.Variable(words_seq)
        words_mask = autograd.Variable(words_mask)
        words_arg1_dist_seq = autograd.Variable(words_arg1_dist_seq)
        words_arg2_dist_seq = autograd.Variable(words_arg2_dist_seq)
        words_ent_ind_seq = autograd.Variable(words_ent_ind_seq)

        arg1 = autograd.Variable(arg1)
        arg2 = autograd.Variable(arg2)
        arg1_mask = autograd.Variable(arg1_mask)
        arg2_mask = autograd.Variable(arg2_mask)

        arg_dep_dist = autograd.Variable(arg_dep_dist)
        arg_dep_dist_mask = autograd.Variable(arg_dep_dist_mask)
        arg1_dep_dist = autograd.Variable(arg1_dep_dist)
        arg1_dep_dist_mask = autograd.Variable(arg1_dep_dist_mask)
        arg2_dep_dist = autograd.Variable(arg2_dep_dist)
        arg2_dep_dist_mask = autograd.Variable(arg2_dep_dist_mask)

        piece1mask_seq = autograd.Variable(piece1mask_seq)
        piece2mask_seq = autograd.Variable(piece2mask_seq)
        piece3mask_seq = autograd.Variable(piece3mask_seq)

        if model_id in [1]:
            outputs = model(words_seq, words_mask, words_arg1_dist_seq, words_arg2_dist_seq)
        elif model_id == 2:
            outputs = model(words_seq, words_mask, words_arg1_dist_seq, words_arg2_dist_seq,
                            piece1mask_seq, piece2mask_seq, piece3mask_seq)
        elif model_id in [3]:
            outputs = model(words_seq, words_mask, words_arg1_dist_seq, words_arg2_dist_seq,
                            piece1mask_seq, piece2mask_seq, piece3mask_seq, arg1, arg2)
        elif model_id in [4]:
            outputs = model(words_seq, words_mask, words_arg1_dist_seq, words_arg2_dist_seq,
                            piece1mask_seq, piece2mask_seq, piece3mask_seq)
        elif model_id in [5]:
            outputs = model(words_seq, words_mask, words_ent_ind_seq, words_arg1_dist_seq, words_arg2_dist_seq,
                            arg_dep_dist, arg1_dep_dist, arg2_dep_dist, arg1_mask, arg2_mask, arg_dep_dist_mask,
                            arg1_dep_dist_mask, arg2_dep_dist_mask)

        preds += list(outputs.data.cpu().numpy())

    return preds


def torch_train(model_id, train_samples, dev_samples, test_samples, best_model_file):
    train_size = len(train_samples)
    batch_count = int(math.ceil(train_size/batch_size))
    move_last_batch = False
    if len(train_samples) - batch_size * (batch_count - 1) == 1:
        move_last_batch = True
        batch_count -= 1
    custom_print(batch_count)
    model = get_model(model_id)

    custom_print(model)
    if torch.cuda.is_available():
        model.cuda()

    rel_loss_func = nn.NLLLoss()
    optimizer = optim.Adagrad(model.parameters())
    custom_print(optimizer)

    best_dev_acc = -1.0
    best_epoch_idx = -1
    best_epoch_seed = -1
    for epoch_idx in range(0, num_epoch):
        model.train()
        custom_print('Epoch:', epoch_idx + 1)
        cur_seed = random_seed + epoch_idx + 1
        np.random.seed(cur_seed)
        torch.cuda.manual_seed(cur_seed)
        random.seed(cur_seed)
        cur_shuffled_train_data = shuffle_data(train_samples)
        start_time = datetime.datetime.now()
        train_loss_val = 0.0
        for batch_idx in tqdm(range(0, batch_count)):
            batch_start = batch_idx * batch_size
            batch_end = min(len(cur_shuffled_train_data), batch_start + batch_size)
            if batch_idx == batch_count - 1 and move_last_batch:
                batch_end = len(cur_shuffled_train_data)

            cur_batch = cur_shuffled_train_data[batch_start:batch_end]
            cur_seq_len, cur_samples_input, cur_samples_target = get_batch_data(cur_batch, True)

            words_seq = torch.from_numpy(cur_samples_input['words_input'].astype('long'))
            words_mask = torch.from_numpy(cur_samples_input['words_mask_input'].astype('float32'))
            words_arg1_dist_seq = torch.from_numpy(cur_samples_input['words_arg1_dist_input'].astype('long'))
            words_arg2_dist_seq = torch.from_numpy(cur_samples_input['words_arg2_dist_input'].astype('long'))
            words_ent_ind_seq = torch.from_numpy(cur_samples_input['words_ent_indicator_input'].astype('long'))

            arg_dep_dist = torch.from_numpy(cur_samples_input['arg_dep_dist'].astype('float32'))
            arg_dep_dist_mask = torch.from_numpy(cur_samples_input['arg_dep_dist_mask'].astype('uint8'))
            arg1_dep_dist = torch.from_numpy(cur_samples_input['arg1_dep_dist'].astype('float32'))
            arg1_dep_dist_mask = torch.from_numpy(cur_samples_input['arg1_dep_dist_mask'].astype('uint8'))
            arg2_dep_dist = torch.from_numpy(cur_samples_input['arg2_dep_dist'].astype('float32'))
            arg2_dep_dist_mask = torch.from_numpy(cur_samples_input['arg2_dep_dist_mask'].astype('uint8'))

            arg1 = torch.from_numpy(cur_samples_input['arg1'].astype('long'))
            arg2 = torch.from_numpy(cur_samples_input['arg2'].astype('long'))
            arg1_mask = torch.from_numpy(cur_samples_input['arg1_mask_input'].astype('float32'))
            arg2_mask = torch.from_numpy(cur_samples_input['arg2_mask_input'].astype('float32'))

            piece1mask_seq = torch.from_numpy(cur_samples_input['piece1mask_input'].astype('float32'))
            piece2mask_seq = torch.from_numpy(cur_samples_input['piece2mask_input'].astype('float32'))
            piece3mask_seq = torch.from_numpy(cur_samples_input['piece3mask_input'].astype('float32'))

            relation_label_target = torch.from_numpy(cur_samples_target['relation_label'].astype('long'))

            if torch.cuda.is_available():
                words_seq = words_seq.cuda()
                words_mask = words_mask.cuda()
                words_arg1_dist_seq = words_arg1_dist_seq.cuda()
                words_arg2_dist_seq = words_arg2_dist_seq.cuda()
                words_ent_ind_seq = words_ent_ind_seq.cuda()

                arg_dep_dist = arg_dep_dist.cuda()
                arg_dep_dist_mask = arg_dep_dist_mask.cuda()
                arg1_dep_dist = arg1_dep_dist.cuda()
                arg1_dep_dist_mask = arg1_dep_dist_mask.cuda()
                arg2_dep_dist = arg2_dep_dist.cuda()
                arg2_dep_dist_mask = arg2_dep_dist_mask.cuda()

                arg1 = arg1.cuda()
                arg2 = arg2.cuda()
                arg1_mask = arg1_mask.cuda()
                arg2_mask = arg2_mask.cuda()

                piece1mask_seq = piece1mask_seq.cuda()
                piece2mask_seq = piece2mask_seq.cuda()
                piece3mask_seq = piece3mask_seq.cuda()

                relation_label_target = relation_label_target.cuda()

            words_seq = autograd.Variable(words_seq)
            words_mask = autograd.Variable(words_mask)
            words_arg1_dist_seq = autograd.Variable(words_arg1_dist_seq)
            words_arg2_dist_seq = autograd.Variable(words_arg2_dist_seq)
            words_ent_ind_seq = autograd.Variable(words_ent_ind_seq)

            arg1 = autograd.Variable(arg1)
            arg2 = autograd.Variable(arg2)
            arg1_mask = autograd.Variable(arg1_mask)
            arg2_mask = autograd.Variable(arg2_mask)

            arg_dep_dist = autograd.Variable(arg_dep_dist)
            arg_dep_dist_mask = autograd.Variable(arg_dep_dist_mask)
            arg1_dep_dist = autograd.Variable(arg1_dep_dist)
            arg1_dep_dist_mask = autograd.Variable(arg1_dep_dist_mask)
            arg2_dep_dist = autograd.Variable(arg2_dep_dist)
            arg2_dep_dist_mask = autograd.Variable(arg2_dep_dist_mask)

            piece1mask_seq = autograd.Variable(piece1mask_seq)
            piece2mask_seq = autograd.Variable(piece2mask_seq)
            piece3mask_seq = autograd.Variable(piece3mask_seq)

            relation_label_target = autograd.Variable(relation_label_target)

            if model_id in [1]:
                outputs = model(words_seq, words_mask, words_arg1_dist_seq, words_arg2_dist_seq, True)
            elif model_id == 2:
                outputs = model(words_seq, words_mask, words_arg1_dist_seq, words_arg2_dist_seq,
                                piece1mask_seq, piece2mask_seq, piece3mask_seq, True)
            elif model_id in [3]:
                outputs = model(words_seq, words_mask, words_arg1_dist_seq, words_arg2_dist_seq,
                                piece1mask_seq, piece2mask_seq, piece3mask_seq, arg1, arg2, True)
            elif model_id in [4]:
                outputs = model(words_seq, words_mask, words_arg1_dist_seq, words_arg2_dist_seq,
                                piece1mask_seq, piece2mask_seq, piece3mask_seq, True)
            elif model_id in [5]:
                outputs = model(words_seq, words_mask, words_ent_ind_seq, words_arg1_dist_seq, words_arg2_dist_seq,
                                arg_dep_dist, arg1_dep_dist, arg2_dep_dist, arg1_mask, arg2_mask, arg_dep_dist_mask,
                                arg1_dep_dist_mask, arg2_dep_dist_mask, True)

            loss = rel_loss_func(outputs, relation_label_target)
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), 10.0)
            optimizer.step()
            train_loss_val += loss.data[0]

        train_loss_val /= batch_count
        end_time = datetime.datetime.now()
        custom_print('Training Loss:', train_loss_val)
        custom_print('Time:', end_time - start_time)

        custom_print('\nDev Results\n')
        torch.cuda.manual_seed(random_seed)
        dev_preds = predict(dev_samples, model, model_id)

        pred_pos, gt_pos, correct_pos = get_F1(dev_samples, dev_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        dev_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('F1:', round(dev_acc, 3))

        if dev_acc >= best_dev_acc:
            best_epoch_idx = epoch_idx + 1
            best_epoch_seed = cur_seed
            custom_print('model saved......')
            best_dev_acc = dev_acc
            torch.save(model.state_dict(), best_model_file)

        custom_print('\nTest Results\n')
        torch.cuda.manual_seed(random_seed)
        test_preds = predict(test_samples, model, model_id)

        pred_pos, gt_pos, correct_pos = get_F1(test_samples, test_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        test_acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('F1:', round(test_acc, 3))

        custom_print('\n\n')
        if epoch_idx + 1 - best_epoch_idx >= early_stop_cnt:
            break

    custom_print('*******')
    custom_print('Best Epoch:', best_epoch_idx)
    custom_print('Best Epoch Seed:', best_epoch_seed)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
    random_seed = 1023
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    src_data_folder = sys.argv[2]
    trg_data_folder = sys.argv[3]
    model_name = int(sys.argv[4])
    job_mode = sys.argv[5]
    mfc = 1
    if model_name == 5:
        mfc = int(sys.argv[6])
        assert 0<= mfc <= 5

    if not os.path.exists(trg_data_folder):
        os.mkdir(trg_data_folder)

    batch_size = 50
    num_epoch = 50
    max_sent_len = 100
    dep_win_size_rel = 5
    use_dep_dist = True
    dim_size = 50
    softmax_type = 0
    word_embed_dim = 50
    conv_filter_cnt = 230
    distance_embed_dim = 5
    ent_indicator_embed_dim = 10
    apply_embed_dropout = False
    embedding_file = os.path.join(src_data_folder, 'w2v.txt')

    word_density = 10
    drop_out = 0.5
    ctx_len = 5
    early_stop_cnt = 3
    lstm_direction = 2
    ignore_rel_list = ['None', 'NA', 'Other']
    is_type_available = False

    relation_cls_label_map, rel_label_cls_map = get_class_label_map(os.path.join(src_data_folder, 'relations.txt'))
    max_word_arg_head_dist = 30
    dist_vocab_size = 2 * max_word_arg_head_dist + 1

    QASample = recordclass("QASample", "UID Id Len Text Arg1 Arg2 Words Words_Mask WordsArg1Dist WordsArg2Dist "
                                       "WordsEntIndicator WordsArg1DepDist WordsArg2DepDist WordsDepDist "
                                       "Arg1_Mask Arg2_Mask Piece1Mask Piece2Mask Piece3Mask RelationName")

    # train a model
    if job_mode == 'train':
        logger = open(os.path.join(trg_data_folder, 'training.log'), 'w')
        custom_print(sys.argv)
        custom_print(dep_win_size_rel)
        custom_print(ctx_len)
        custom_print(drop_out)
        custom_print('loading data......')
        best_model_file_name = os.path.join(trg_data_folder, 'model.h5py')
        out_file_name = os.path.join(trg_data_folder, 'dev-relation-out.txt')
        train_file = os.path.join(src_data_folder, 'train.json')
        dev_file = os.path.join(src_data_folder, 'dev.json')
        test_file = os.path.join(src_data_folder, 'test.json')

        train_dep_file = os.path.join(src_data_folder, 'train.dep')
        dev_dep_file = os.path.join(src_data_folder, 'dev.dep')
        test_dep_file = os.path.join(src_data_folder, 'test.dep')

        train_data = read_data(train_file, train_dep_file, is_training_data=True)
        dev_data = read_data(dev_file, dev_dep_file)
        test_data = read_data(test_file, test_dep_file)

        custom_print('Training data size:', len(train_data))
        custom_print('Development data size:', len(dev_data))
        custom_print('Test data size:', len(test_data))

        custom_print("preparing vocabulary......")
        vocab_file_name = os.path.join(trg_data_folder, 'vocab.pkl')
        all_data = train_data + dev_data + test_data
        word_vocab, word_embed_matrix = build_vocab(train_data, dev_data, test_data, vocab_file_name, embedding_file)

        custom_print('vocab size:', len(word_vocab))

        custom_print("Training started......")
        torch_train(model_name, train_data, dev_data, test_data, best_model_file_name)
        logger.close()

    if job_mode == 'test':
        logger = open(os.path.join(trg_data_folder, 'test.log'), 'w')
        custom_print(sys.argv)
        custom_print("loading word vectors......")
        vocab_file_name = os.path.join(trg_data_folder, 'vocab.pkl')
        word_vocab = load_vocab(vocab_file_name)

        word_embed_matrix = np.zeros((len(word_vocab), word_embed_dim), dtype=np.float32)
        custom_print('vocab size:', len(word_vocab))

        custom_print('seed:', random_seed)
        model_file = os.path.join(trg_data_folder, 'model.h5py')

        best_model = get_model(model_name)
        custom_print(best_model)
        if torch.cuda.is_available():
            best_model.cuda()
        best_model.load_state_dict(torch.load(model_file))

        # prediction on dev data

        dev_file = os.path.join(src_data_folder, 'dev.json')
        dev_dep_file = os.path.join(src_data_folder, 'dev.dep')
        dev_data = read_data(dev_file, dev_dep_file)
        custom_print('Dev data size:', len(dev_data))
        torch.cuda.manual_seed(random_seed)
        dev_preds = predict(dev_data, best_model, model_name)

        custom_print('\nDev Results')
        pred_pos, gt_pos, correct_pos = get_F1(dev_data, dev_preds)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('P:', round(p, 3))
        custom_print('R:', round(r, 3))
        custom_print('F1 Before Thresholding:', round(acc, 3))

        threshold = get_threshold(dev_data, dev_preds)
        custom_print('\nThreshold:', round(threshold, 3))
        print()
        pred_pos, gt_pos, correct_pos = get_F1(dev_data, dev_preds, threshold)
        custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
        p = float(correct_pos) / (pred_pos + 1e-8)
        r = float(correct_pos) / (gt_pos + 1e-8)
        acc = (2 * p * r) / (p + r + 1e-8)
        custom_print('P:', round(p, 3))
        custom_print('R:', round(r, 3))
        custom_print('F1 After Thresholding:', round(acc, 3))

        test_files = ['test']
        for file_name in test_files:
            custom_print('\n\n\nTest Results:', file_name)
            test_input_file = os.path.join(src_data_folder, file_name + '.json')
            test_dep_file = os.path.join(src_data_folder, file_name + '.dep')
            test_data = read_data(test_input_file, test_dep_file)
            out_file_name = os.path.join(trg_data_folder, file_name + '-output.json')

            custom_print('Test data size:', len(test_data))
            torch.cuda.manual_seed(random_seed)
            test_preds = predict(test_data, best_model, model_name)

            pred_pos, gt_pos, correct_pos = get_F1(test_data, test_preds)
            custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
            p = float(correct_pos) / (pred_pos + 1e-8)
            r = float(correct_pos) / (gt_pos + 1e-8)
            test_acc = (2 * p * r) / (p + r + 1e-8)
            custom_print('F1 Before Thresholding:', round(test_acc, 3))
            print()

            pred_pos, gt_pos, correct_pos = get_F1(test_data, test_preds, threshold)
            custom_print(pred_pos, '\t', gt_pos, '\t', correct_pos)
            p = float(correct_pos) / (pred_pos + 1e-8)
            r = float(correct_pos) / (gt_pos + 1e-8)
            test_acc = (2 * p * r) / (p + r + 1e-8)
            custom_print('P:', round(p, 3))
            custom_print('R:', round(r, 3))
            custom_print('F1 After Thresholding:', round(test_acc, 3))
            print()
            # write_pred_file(test_data, test_preds, out_file_name, threshold)
            # write_PR_curve(test_data, test_preds, os.path.join(trg_data_folder, file_name + '_pr_data.csv'))
            # pr_curve(os.path.join(trg_data_folder, file_name + '_pr_data.csv'),
            #          os.path.join(trg_data_folder, file_name + '_pr_curve.csv'))

        logger.close()
