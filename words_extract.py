#!/usr/bin/env python
# -*- coding:utf-8 -*-
import os
import fire
import pypinyin
from pypinyin import lazy_pinyin


def txt_read(file):
    with open(file, 'r', encoding='utf-8') as f:
        ff = f.readlines()
    ff = [x.strip() for x in ff if x not in [""]]
    ff = [x for x in ff if x not in [""]]
    ff = [x.split("---------------")[-1] if "---------------" in x else x for x in ff]
    return ff


def get_sentence(path='train_corpus'):
    bath_path = os.getcwd()
    file_path = os.path.join(bath_path, path)
    file_list = os.listdir(file_path)
    all_sentence = []
    for file in file_list:
        if file.endswith(".txt"):
            all_sentence.extend(txt_read(file=os.path.join(file_path, file)))
    return all_sentence


def transform_han(value):
    num_han = {0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七', 8: '八', 9: '九'}
    value = ''.join(x for x in value if x in "0123456789")
    value = [num_han.get(int(x)) for x in value]
    return value


def get_words(all_sentence=''):
    words_list = []
    sentence_len = []
    sentence_pinyin = []
    for sentence in all_sentence:
        sentence_len.append(len(sentence))
        sentence_pinyin.append(' '.join(lazy_pinyin(x, style=pypinyin.TONE2)[0] for x in list(sentence)))
        sentence = list(sentence)
        words_list.extend(sentence)
    sentence_pinyin = [len(x) for x in sentence_pinyin]
    print('max sentence len:', max(sentence_len))
    print('max char len:', max(sentence_pinyin))
    words_list = list(set(words_list))
    num_list = [x for x in words_list if x in "1234567890"]
    num_han = transform_han(value=''.join(x for x in num_list))
    not_num_list = [x for x in num_list if x not in "1234567890"]
    words_list.extend(num_han)
    if len(not_num_list) > 0:
        print("words list include number:%s" % (' '.join(x for x in not_num_list)))
    words_list = [x.strip() for x in words_list]
    words_list = [x for x in words_list if
                  x not in "1234567890!abcdefghigklmnopqrstuvwxyzABCDEFGHIGKLMNOPQRSTUVWUVWXYZ@#$%^&*(" \
                           ")！@#￥……&×（）？?：;“”‘’.。,，,:.!@#$%^&*()?/`~'"]
    pinyin_no_tone = [lazy_pinyin(x)[0] for x in words_list]
    pinyin_tone = [lazy_pinyin(x, style=pypinyin.TONE)[0] for x in words_list]

    return words_list, pinyin_no_tone, pinyin_tone


def words_not_sentence(sentence="我是中国人", words_list=None, pinyin_no_tone=None, pinyin_tone=None):
    sentence = list(sentence)
    sentence = [x.strip() for x in sentence]
    sentence = [x for x in sentence if
                x not in "1234567890!abcdefghigklmnopqrstuvwxyzABCDEFGHIGKLMNOPQRSTUVWUVWXYZ@#$%^&*(" \
                         ")！@#￥……&×（）？?：:;“”‘’.。'，,"]
    num = "十百千万亿零一二三四五六七八九"
    num = list(num)
    not_include_words = []
    not_include_pinyin_no_tone = []
    not_include_pinyin_tone = []
    if words_list:
        for words in sentence:
            if words not in words_list:
                not_include_words.append(words)
    if len(not_include_words) > 0:
        print("句子:%s：\n训练词典中不包含下列词语:\n%s" % (''.join(x for x in sentence), ' '.join(x for x in not_include_words)))
    else:
        print("所有词语都包含在训练词典中")
    if pinyin_no_tone:
        sentence_pinyin_no_tone = [lazy_pinyin(x)[0] for x in sentence]
        for words_pin in sentence_pinyin_no_tone:
            if words_pin not in pinyin_no_tone:
                not_include_pinyin_no_tone.append(words_pin)
    if len(not_include_pinyin_no_tone) > 0:
        print("句子:%s：\n训练词典中不包含下列音:\n%s" % (
            ''.join(x for x in sentence), ' '.join(x for x in not_include_pinyin_no_tone)))
    if pinyin_tone:
        sentence_pinyin_tone = [lazy_pinyin(x, style=pypinyin.TONE)[0] for x in sentence]
        for piny in sentence_pinyin_tone:
            if piny not in pinyin_tone:
                not_include_pinyin_tone.append(piny)
    if len(list(set(not_include_pinyin_tone))) > 0:
        result = False
        print("句子:%s：\n训练词典中不包含下列音（带音调）:\n%s" % (
            ''.join(x for x in sentence), ' '.join(x for x in not_include_pinyin_tone)))
    else:
        result = True
    # -------------十百千万亿零一二三四五六七八九------------------------------------------------------------
    num = [lazy_pinyin(x, style=pypinyin.TONE)[0] for x in num]
    not_include_num = []
    for nu in num:
        if nu not in pinyin_tone:
            not_include_num.append(nu)
    if len(not_include_num) > 0:
        print("十百千万亿零一二三四五六七八九 没有包含的有：%s" % ' '.join(x for x in not_include_num))
    else:
        print("十百千万亿零一二三四五六七八九 所有音调都在训练集合中")

    return result, list(set(not_include_pinyin_tone))


def check_char(sentence="是这样，因为我是刚来的话务员，回答不了您太多问题，有关具体细节的问题稍后让我们贷款专员给您回电话再详细解答。"):
    all_sentence = get_sentence()
    words_list, pinyin_no_tone, pinyin_tone = get_words(all_sentence=all_sentence)
    result, not_include_tone = words_not_sentence(sentence=sentence, words_list=words_list, pinyin_no_tone=pinyin_no_tone, pinyin_tone=pinyin_tone)
    return result, not_include_tone


if __name__ == '__main__':
    fire.Fire(check_char)
