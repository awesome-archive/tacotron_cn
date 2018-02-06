import logging


def create_logger(name, loglevel=logging.INFO, fn=None):
    fmt = '%(asctime)s %(name)s %(levelname)s[%(filename)s:%(lineno)d] %(message)s'
    datefmt = None  # '%y-%m-%d %H:%M:%S'

    # set up logging to file
    '''logging.basicConfig(
        filename=fn,
        level=loglevel,
        format=fmt,
        datefmt=datefmt
    )'''

    logger = logging.getLogger(name)
    logger.handlers = []

    logger.setLevel(loglevel)

    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if fn:
        print('new app_log file:', fn)
        fh = logging.FileHandler(fn, mode='w')
        fh.setLevel(loglevel)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
