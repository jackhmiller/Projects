import logging
import datetime as dt
import os

folder = dt.datetime.now().strftime("%y-%m-%d-%H_%M")
path = "../Auction_model/Log/{}".format(folder)
os.mkdir(path)

#log_day = {'DoW': dt.datetime.today().strftime('%A')}
logger = logging.getLogger('Bond Auction')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')
file_handler = logging.FileHandler('{}/log.log'.format(path))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
