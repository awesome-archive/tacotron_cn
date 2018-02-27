# -*- coding: utf-8 -*-
from utils import atc
from pypinyin import lazy_pinyin, load_phrases_dict
import pypinyin
import json

# origin
# "."
PUNCTUATION = ['、', '“', '”', '；', '：', '（', "）", ":", ";", ",", "?", "!", "\"", "\'", "(", ")"]
PUNCTUATION1 = r'，、。？！;,?!'  # 断句分隔符
PUNCTUATION2 = r'“”；：（）×"\':()*#'  # 其它符号
'''
alpha_pronuce = {"A": "ei", "B": "bii", "C": "cii", "D": "dii", "E": "ii", "F": "ef", "G": "jii", "H": "eich",
                 "I": "ai", "J": "jei", "K": "kei", "L": "el", "M": "em", "N": "en",
                 "O": "eo", "P": "pii", "Q": "kiu", "R": "aa", "S": "es", "T": "tii", "U": "iu", "V": "vii",
                 "W": "dabliu", "X": "eks", "Y": "wia", "Z": "zii"}
'''
alpha_pronuce = {"A": "ei ", "B": "bii ", "C": "sii ", "D": "dii ", "E": "ii ", "F": "ef ", "G": "dji ", "H": "eich ",
                 "I": "ai ", "J": "jei ", "K": "kei ", "L": "el ", "M": "em ", "N": "en ",
                 "O": "eo ", "P": "pii ", "Q": "kiu ", "R": "aa ", "S": "es ", "T": "tii ", "U": "iu ", "V": "vii ",
                 "W": "dabliu ", "X ": "eiks ", "Y": "wai ", "Z": "zii "}

# PUNCTUATION2 = r'“”（）×"\'()*#'  # 其它符号
# load_phrases_dict({u'360': [[u'jú'], [u'zǐ']]})

def json_load():
    with open("user_dict/fault-tolerant_word.json", "r") as rf:
        data = json.load(rf)
    return data


usr_phrase = json_load()
load_phrases_dict(usr_phrase)


def text2pinyin(syllables):
    temp = []
    for syllable in syllables:
        for p in PUNCTUATION:
            syllable = syllable.replace(p, "")
        # print(syllable)
        # if syllable.isdigit():
        try:
            syllable = atc.num2chinese(syllable)
            # print("sy:", syllable)
            new_sounds = lazy_pinyin(syllable, style=pypinyin.TONE2)
            print("pinyin:" + str(new_sounds))
            for e in new_sounds:
                temp.append(e)
        except:
            syllable = syllable.replace(".", "")
            for p in PUNCTUATION:
                syllable = syllable.replace(p, "")
            temp.append(syllable)
    return temp


def ch2p(speech):
    if type(speech) == str:
        # print('拼音转换: ', speech)
        syllables = lazy_pinyin(speech, style=pypinyin.TONE2)
        # print('---------1 ', speech, '----------')
        syllables = text2pinyin(syllables)
        text = ' '.join(syllables)
        ''''''
        for alpha, pronuce in alpha_pronuce.items():
            text = text.replace(alpha, pronuce)
        text = text.replace("  "," ")
        text = text.replace("  ", " ")

        return text
    else:
        print("input format error")


def num2han(value):
    num_han = {0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七', 8: '八', 9: '九'}
    value = ''.join(x for x in value if x in "0123456789")
    value = ''.join(num_han.get(int(x)) for x in value)
    return value


def num2phone(value):
    num_han = {0: '零', 1: '妖', 2: '二', 3: '三', 4: '四', 5: '五', 6: '六', 7: '七', 8: '八', 9: '九'}
    value = ''.join(x for x in value if x in "0123456789")
    value = ''.join(num_han.get(int(x)) for x in value)
    return value


if __name__ == "__main__":
    print(num2han(value="010194567898"))
    print(ch2p(num2phone("010194567898")))
    print(atc.num2chinese("3418.91"))
    print(atc.num2chinese("2418.91", twoalt=True))
    print(ch2p("月A 三ABW六零"))
