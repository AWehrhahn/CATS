import logging

def init(log):
    logging.basicConfig(filename='exoSpectro.log', level=logging.DEBUG, filemode='w')
    log.isInit = True

def log(level, msg):
    """ log/print stuff """

    if not hasattr(log, 'isInit'):
        init(log)

    logging.info(msg)
    print('   ' * level + '-', msg)
