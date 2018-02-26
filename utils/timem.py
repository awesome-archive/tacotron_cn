import datetime

'''
from timem import *

s = ts()
s,_ = te(s, 'step1')
s,_ = te(s, 'step2')
'''


def ts(info=None, logger=None, disp_time=False):
    start_time = datetime.datetime.now()
    if info is not None:
        if logger is None:
            if disp_time:
                print(info, str(start_time))
            else:
                print(info)
        else:
            logger.info(info)
    return start_time


def te(s, info='', logger=None, disp=True, disp_time=False, tail=''):
    info = info.strip()
    # if len(info) == 0:
    #    raise

    end_time = datetime.datetime.now()
    d = (end_time - s)
    used_time_ms = d.total_seconds() * 1000
    if disp_time:
        r = '%s %s-%s used time: %.2f ms' % (info, str(s), str(end_time), used_time_ms)
    else:
        r = '%s used time: %.2f ms' % (info, used_time_ms)

    r += tail

    if disp:
        if logger is None:
            print(r)
        else:
            logger.info(r)
    return end_time, used_time_ms


def now_str(sep=True):
    fmt = "%Y-%m-%d %H:%M:%S" if sep else "%Y%m%d%H%M%S"
    n = datetime.datetime.now()
    return n.strftime(format=fmt)
