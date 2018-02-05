import pypinyin
import traceback


from util.atc import num2chinese

PUNCTUATION = ['，', '、', '。', '？', '！', '“', '”', '；', '：', '（', "）", ":", ";", ",", ".", "?", "!", "\"", "\'", "(",
               ")"]


def num2chinese_txt(txt):
    r = ''
    num = ''
    for ch in txt:
        if ch.isdigit():
            num += ch
        else:
            if num:
                # num = '2802'
                num = num2chinese(num)
                print('num after num2chinese:', num)
                r += num
            r += ch
            num = ''
    if num:
        num = num2chinese(num)
        print('num after num2chinese:', num)
        r += num
    return r


def text2pinyin(txt, style=pypinyin.TONE2):
    temp = []
    txt = txt.replace('，', '').replace('！', '')
    txt = num2chinese_txt(txt)

    for ch in txt:
        try:
            if ch.isdigit():
                assert False  # never here

            new_sounds = pypinyin.lazy_pinyin(ch, style=style)
            for e in new_sounds:
                # print('eeeee:', e)
                tone_tail = False
                if tone_tail:
                    e1 = ''
                    t = None
                    for c in e:
                        if not c.isdigit():
                            e1 += c
                        else:
                            t = c
                    if t is not None:
                        e1 += t
                    # print('e1:', e1)
                else:
                    e1 = e
                temp.append(e1)
        except:
            traceback.print_exc()
            for p in PUNCTUATION:
                ch = ch.replace(p, "")
            temp.append(ch)

    pinyin = ' '.join(temp)

    print('text2pinyin:', txt, '-->', pinyin)
    return pinyin


def test1():
    txt = '如果款项2802元仍未结清，后续部门会联系你'
    pinyin = text2pinyin(txt, style=pypinyin.TONE2)
    # pinyin = text2pinyin(txt, style=pypinyin.NORMAL)
    print(pinyin)


def test2():
    txt = '2018如果款项2802元仍未结清，后续部门会联系你1982'
    print(num2chinese_txt(txt))


if __name__ == '__main__':
    test1()
