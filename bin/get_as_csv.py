import sys
import argparse
import logging
import datetime as dt
import json
import itertools
import numpy as np

from mlfdb import mlfdb
from lib import io as _io

def main():
    """
    Get data from db and save it as csv
    """ 
    a = mlfdb.mlfdb()
    io = _io.IO()
   
    starttime, endtime = io.get_dates(options)

    logging.info('Loading classification dataset from db')
    if starttime is not None and endtime is not None:
        logging.info('Using time range {} - {}'.format(starttime.strftime('%Y-%m-%d'), endtime.strftime('%Y-%m-%d')))        
    
    metadata, header, data = a.get_rows(options.dataset,
                                        starttime=starttime,
                                        endtime=endtime,
                                        rowtype=options.type)

    logging.info('Header is: \n {} \n'.format(','.join(header)))
    
    logging.info('Shape of metadata: {}'.format(np.array(metadata).shape))
    logging.info('Sample of metadata: \n {} \n'.format(np.array(metadata)))
    
    logging.info('Shape of data {}'.format(data.shape))
    logging.info('Sample of data: \n {} \n '.format(data))


    # TODO saving as csv
    
    # Serialize model to disc
    # logging.info('Serializing dataset to disc: {}'.format(options.save_path))
    
    # csv = dataset.as_csv()
    # with open(options.save_path, "w") as f: 
    #    f.write(csv)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--starttime', type=str, help='Start time of the classification data interval')
    parser.add_argument('--endtime', type=str, help='End time of the classification data interval')
    parser.add_argument('--save_path', type=str, default=None, help='Dataset save path and filename')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name')
    parser.add_argument('--type', type=str, default='feature', help='feature/label')
    parser.add_argument('--logging_level',
                        type=str,
                        default='INFO',
                        help='options: DEBUG,INFO,WARNING,ERROR,CRITICAL')

    options = parser.parse_args()
    
    debug=False

    logging_level = {'DEBUG':logging.DEBUG,
                     'INFO':logging.INFO,
                     'WARNING':logging.WARNING,
                     'ERROR':logging.ERROR,
                     'CRITICAL':logging.CRITICAL}
    logging.basicConfig(format=("[%(levelname)s] %(asctime)s %(filename)s:%(funcName)s:%(lineno)s %(message)s"), level=logging_level[options.logging_level])

    main()
