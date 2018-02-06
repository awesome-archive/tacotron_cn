# wav_util.py

import os
import subprocess

from util.path_util import main_filename, get_path
from util.cmd_util import call_cmd


# noinspection PyPep8,PyBroadException
def get_stat(fn):
    try:
        cmd = 'sox %s -n stat -v' % fn
        s = subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT)
        s = s.decode('utf-8')
        s = s.strip()
        v = float(s)
    except:
        print(fn, ' v=err')
        v = 'err'
    return v


def vol_gain(fn):
    mfn = main_filename(fn)
    v = get_stat(fn)
    if v != 'err':
        dst_fn = os.path.join(get_path(fn), mfn + '_vol.wav')
        call_cmd('sox -v %s %s %s' % (v, fn, dst_fn))
        return dst_fn
    else:
        return None


def trim_noise(fn, logger):
    try:
        logger.info('fn:' + fn)
        mfn = main_filename(fn)
        new_fn = os.path.join(get_path(fn), mfn + '_trim_noise.wav')
        cmd = 'sox -V3 %s %s silence 1 0.1 0.1%% 1 0.1 0.1%% : newfile : restart' % (fn, new_fn)
        logger.info('in trim_noise:' + cmd)
        call_cmd(cmd)
        logger.info('new_fn:' + new_fn)

        mfn = main_filename(new_fn)
        new_fn = os.path.join(get_path(fn), mfn + '001.wav')
        logger.info('new_fn 1:' + new_fn)
        return new_fn
    except Exception as e:
        raise e
