#!/usr/bin/env python
#-*- coding:utf-8 -*-
# -*- coding: utf-8 -*-
# demo_server.py
import argparse
import falcon
from hparams import hparams, hparams_debug_string
import io
import json
import os
import re
import contextlib
import wave
from synthesizer_pb_ckpt import Synthesizer
import traceback
from pydub import AudioSegment
import uuid

# from util.pinyin_util import text2pinyin
from chinese2pinyin import ch2p, num2han, num2phone, PUNCTUATION1, PUNCTUATION2
from words_extract import check_char
from logger.logger import create_logger
from utils.timem import now_str
from utils.wav_util import vol_gain, trim_noise  # , trim_len
from vad_detect import vad_check_wav, vad_check
# special_num = ["01053965700", "360"]

MAX_SEN_LEN = 40

html_body = '''<html><title>Demo</title>
<style>
body {padding: 16px; font-family: sans-serif; font-size: 14px; color: #444}
input {font-size: 14px; padding: 8px 12px; outline: none; border: 1px solid #ddd}
input:focus {box-shadow: 0 1px 2px rgba(0,0,0,.15)}
p {padding: 12px}
button {background: #28d; padding: 9px 14px; margin-left: 8px; border: none; outline: none;
        color: #fff; font-size: 14px; border-radius: 4px; cursor: pointer;}
button:hover {box-shadow: 0 1px 2px rgba(0,0,0,.15); opacity: 0.9;}
button:active {background: #29f;}
button[disabled] {opacity: 0.4; cursor: default}
</style>
<body>
<form>
  <input id="text" type="text" size="40" placeholder="Enter Text">
  <button id="button" name="synthesize">Speak</button>
</form>
<p id="message"></p>
<audio id="audio" controls autoplay hidden></audio>
<script>
function q(selector) {return document.querySelector(selector)}
q('#text').focus()
q('#button').addEventListener('click', function(e) {
  text = q('#text').value.trim()
  if (text) {
    q('#message').textContent = 'Synthesizing...'
    q('#button').disabled = true
    q('#audio').hidden = true
    synthesize(text)
  }
  e.preventDefault()
  return false
})
function synthesize(text) {
  url = '/synthesize'
  // url = '/synthesize?text=' + encodeURIComponent(text)
  console.log('feching ' + url)
  fetch(url, {
    method: 'POST',
    body: JSON.stringify({text: text}),
    cache: 'no-cache'
  })
    .then(function(res) {
      if (!res.ok) throw Error(response.statusText)
      return res.blob()
    }).then(function(blob) {
      q('#message').textContent = ''
      q('#button').disabled = false
      q('#audio').src = URL.createObjectURL(blob)
      q('#audio').hidden = false
    }).catch(function(err) {
      q('#message').textContent = 'Error: ' + err.message
      q('#button').disabled = false
    })
}
</script></body></html>
'''


def replace_sign(txt, sign, func):
    if sign in txt:
        old_num = txt.split(sign)[-1].split(")")[0]
        num_txt = func(old_num)
        old_txt = sign + old_num + ")"
        txt = txt.replace(old_txt, num_txt)
    return txt


def process_special_num(txt):
    signs = [('(&amp;', num2han), ('(&', num2han), ('(#', num2phone)]
    for sign, func in signs:
        txt = replace_sign(txt, sign, func)
    return txt


# noinspection PyMethodMayBeStatic
class UIResource:
    def __init__(self):
        pass

    # noinspection PyUnusedLocal
    def on_get(self, req, res):
        res.content_type = 'text/html'
        res.body = html_body


def process_txt(org_txt):
    # app_logger.info('txt type ' + str(type(org_txt)) + str(len(org_txt)) + str(org_txt))
    app_logger.info('got text: ' + str(org_txt))
    if isinstance(org_txt, list):
        txt = ','.join(org_txt)
    else:
        txt = str(org_txt)

    txt = process_special_num(txt)
    app_logger.info("after process_special_num: " + str(txt))

    for ch in PUNCTUATION2:
        txt = str(txt).replace(ch, '')
    for ch in PUNCTUATION1:
        txt = str(txt).replace(ch, ',')
    txt = re.sub(r'[a-zA-Z]', '', txt)
    txt = txt.strip()
    txt = txt.strip(',')
    app_logger.info('new txt: ' + str(txt))

    # for num in special_num:
    #     if num in txt:
    #         txt = txt.replace(num, num2han(num))
    return txt


# noinspection PyMethodMayBeStatic
class SynthesisResource:
    def __init__(self):
        self.speed_wav = False
        self.vol = False
        self.use_trim_noise = True
        self.use_trim_len = True
        self.use_len_limit = True

    def on_post(self, req, res):
        # print('req.params:', req.params)
        # get_txt = req.params.get('text')
        # print('in on_post req.params:', req.params)

        try:
            get_txt = ''
            if req.content_length:
                s = str(req.stream.read().decode())
                data = json.loads(s)
                app_logger.info('on_post data:' + str(data))
                get_txt = data.get('text', '')
                self.speed_wav = data.get('speed_wav', False)
                self.vol = data.get('vol', False)
                self.use_trim_noise = data.get('use_trim_noise', False)
                self.use_trim_len = data.get('use_trim_len', True)
                self.use_len_limit = data.get('use_len_limit', True)

            if not get_txt:
                raise falcon.HTTPBadRequest()

            app_logger.info('in on_post get_txt:'+str(get_txt))

            # if '请问是' in get_txt:
            #    raise Exception('test one sen fail')

            use_check_char = False
            syn_result = None
            not_include_tone = None
            if use_check_char:
                syn_result, not_include_tone = check_char(sentence=get_txt)

            if not use_check_char or syn_result:
                sen, max_sen_len = self.judge_sentence_len(txt=get_txt)

                if self.use_len_limit and max_sen_len > MAX_SEN_LEN:
                    app_logger.info("the length of synthesize sentence has exceed max length %s of synthesise char"
                                    % max_sen_len)

                    res.status = '500 the len sentence exceed max synthesis length'
                else:
                    self.tts_synthesize(get_txt, res)
            else:
                app_logger.info("the synthesize has not include char: " + str(not_include_tone))
                res.status = '500 check_char fail'

                # raise Exception('test exception')
        except Exception as e:
            app_logger.info('on_post fail!')
            exc_info = traceback.format_exc()
            app_logger.info(exc_info)
            res.status = '500 exception %s' % str(e)
        app_logger.info("res.status: %s" % str(res.status))

    # noinspection PyPep8,PyBroadException
    # , char_len = None
    def handle_wav(self, wav_file_path):
        ret = ''
        try:
            app_logger.info('in handle_wav vol:' + str(self.vol))
            app_logger.info('in handle_wav use_trim_len:' + str(self.use_trim_len))
            if self.use_trim_noise:
                wav_file_path = trim_noise(wav_file_path, app_logger)

            if False:
                pass
                #wav_file_path, _ = vad_check_wav(wav_path=wav_file_path)
                # wav_file_path = trim_len(wav_file_path, char_len=char_len, logger=app_logger)

            path8k = wav_file_path.replace(".wav", "_8k.wav")
            cmd = 'sox %s -r 8000 %s' % (wav_file_path, path8k)
            os.system(cmd)

            ret = path8k
            if self.speed_wav:
                path_spd = path8k.replace(".wav", "_spd.wav")
                if os.path.isfile(path_spd):
                    os.remove(path_spd)

                bin_path = 'soundstretch'
                cmd_speed = '%s %s %s -tempo=42.5 > /dev/null 2>&1' % (bin_path, path8k, path_spd)
                os.system(cmd_speed)
                if os.path.isfile(path_spd):
                    ret = path_spd
                else:
                    app_logger.info(str(path_spd) + ' not exists')
                    ret = path8k
            else:
                ret = path8k

            if self.vol:
                new_ret = vol_gain(ret)
                if new_ret is not None:
                    ret = new_ret

        except:
            app_logger.info('handle_wav fail')
            traceback.print_exc()

        if os.path.isfile(ret):
            return ret
        else:
            return None

    # noinspection PyMethodMayBeStatic
    def judge_sentence_len(self, txt):
        for ch in PUNCTUATION1:
            txt = str(txt)
            txt = txt.replace(ch, ',')
        txt = txt.split(',')
        sentence_len = []
        for line in txt:
            sentence_len.append([line, len(line)])
        sentence_len = sorted(sentence_len, key=lambda i: -i[1])
        sentence, _ = zip(*sentence_len)
        max_len = len(sentence[0])
        return sentence, max_len

    def tts_synthesize(self, get_txt, res):
        txt = process_txt(get_txt)
        split_sentence = txt.split(',')
        result = b''#AudioSegment.silent(duration=50)

        import datetime
        start_time = datetime.datetime.now()
        app_logger.info('synthesizing ...')

        inputs = []
        for x in split_sentence:
            inputs.append(ch2p(x))
            app_logger.info('to pinyin: ' + str([len(j) for j in inputs]) + ' ' + str(inputs))
        out = synthesizer.synthesize(inputs)
        for wav in out:
            _, wav = vad_check_wav(wav_path=wav)
            result += wav

        uuid_str = str(uuid.uuid1()).replace('-', '')
        tmp_fn = os.path.realpath('./tmp/%s.wav' % uuid_str)
        tmp_path = os.path.dirname(tmp_fn)
        app_logger.info('tmp_path: ' + str(tmp_path))

        #result.export(tmp_fn, format='wav')
        self.write_wave_vad(wav_path=tmp_fn, audio=result, sample_rate=16000)

        app_logger.info('self.vol: ' + str(self.vol))

        total_len = sum([len(j) for j in split_sentence])
        app_logger.info('total_len: ' + str(total_len))

        new_fn = self.handle_wav(tmp_fn)  # , char_len=total_len
        app_logger.info('new_fn after handle_wav: ' + str(new_fn))

        fp = open(new_fn, 'rb')
        data = fp.read()
        fp.close()

        aud = io.BytesIO(data)
        res.data = aud.read()

        end_time = datetime.datetime.now()
        d = (end_time - start_time)
        used_time_ms = d.total_seconds() * 1000
        app_logger.info('used_time_ms:' + str(used_time_ms))

        res.content_type = 'audio/wav'

        rm_wav = True
        if rm_wav:
            os.system('rm -f %s/%s*.wav' % (tmp_path, uuid_str))

    def write_wave_vad(self, wav_path, audio, sample_rate):
        """Writes a .wav file.

        Takes path, PCM audio data, and sample rate.
        """
        with contextlib.closing(wave.open(wav_path, 'wb')) as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)

class Test:
  def on_get(self, req, res):
    res.body = 'success'
    res.status = falcon.HTTP_200

synthesizer = Synthesizer()
api = falcon.API()
api.add_route('/synthesize', SynthesisResource())
api.add_route('/', UIResource())
api.add_route('/test',Test())

if not os.path.exists('./tmp'):
    os.makedirs('./tmp')

if __name__ == '__main__':
    from wsgiref import simple_server

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
    parser.add_argument('--port', type=int, default=9000)
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()

    log_path = "./log_tts"
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_fn = os.path.join(log_path, "tts_app_%d_%s.log" % (args.port, now_str(sep=False)))
    app_logger = create_logger("app.tts", loglevel=20, fn=log_fn)
    print('log_fn:', log_fn)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    hparams.parse(args.hparams)

    app_logger.info('-------------- log_path:' + str(hparams_debug_string()))
    synthesizer.load(args.checkpoint)
    app_logger.info('Serving on port ' + str(args.port))
    simple_server.make_server('0.0.0.0', args.port, api).serve_forever()
else:
    synthesizer.load(os.environ['CHECKPOINT'])

"""
def test():
  synthesizer.load('../FlaskWebv2/logs-tacotron/model.ckpt-52000')

  txt = '你是'
  print('got text:', txt)
  if not txt:
    raise falcon.HTTPBadRequest()
  data = synthesizer.synthesize(txt)
  data = data.read()
  # data = FileIter(BytesIO_Object.read())
  print(type(data), len(data))
"""

"""
def test1():
    txt = '好，如果款项仍未结清，后续部门会联“系你，请保持手机畅通，再见！'
    pinyin = text2pinyin(txt, style=pypinyin.TONE2)
    # pinyin = text2pinyin(txt, style=pypinyin.NORMAL)
    print(pinyin)

    pinyin = ch2p('欠20.32元')
    print(pinyin)


def test2():
    txt = '好，如果款！项54234元仍未!结清\"，后续部”门abd会?联系你，请保持？手机APP畅通，再见！'
    print(process_txt(txt))


if __name__ == '__main__':
    test2()
"""


