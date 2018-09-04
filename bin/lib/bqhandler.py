# -*- coding: utf-8 -*-
"""
BigQuery handler
"""
import sys
import re
import logging
import os
import numpy as np
import pandas as pd
import datetime
from collections import defaultdict
from configparser import ConfigParser
from google.cloud import bigquery

class BQHandler(object):
    """
    Handle database connection and provide
    methods to insert and read storm objects to database
    """

    def __init__(self, debug=False, training=False, config_filename = None):
        self._connect()

        self.parameters = None
        self.locations = None
        self.order = None
        self.batch_size = None

    def set_params(self,
                   starttime,
                   endtime,
                   batch_size=None,
                   loc_col='loc_name',
                   project=None,
                   dataset=None,
                   table=None,
                   parameters=['*'],
                   locations=None,
                   order=None):
        """
        Set params to be used for get_batch function
        """
        self.starttime = starttime
        self.endtime = endtime
        self.batch_size = batch_size
        self.loc_col = loc_col

        if project is None:
            self.project = self.tables['project']
        else:
            self.project = project

        if dataset is None:
            self.dataset = self.tables['dataset']
        else:
            self.dataset = dataset

        if table is None:
            self.table = self.tables['feature_table']
        else:
            self.table = table

        self.parameters = parameters
        self.locations = locations
        self.batch_num = 0
        self.order = order

    def get_batch_num(self):
        """
        Return curretn batch number
        """
        return self.batch_num

    def get_batch(self):
        """
        Get next batch
        """
        logging.info('Batch {}...'.format(self.batch_num))
        offset = self.batch_num * self.batch_size
        sql = self._format_query(offset)
        self.batch_num += 1
        logging.debug(sql)
        return self._query(sql)

    def get_rows(self,
                 starttime=None,
                 endtime=None,
                 loc_col=None,
                 project=None,
                 dataset=None,
                 table=None,
                 parameters=['*'],
                 locations=None,
                 order=None):
        """
        Get all feature rows from given dataset. All arguments can be given here or in set_params method

        dataset_name : str
                       dataset name
        starttime : DateTime
                    start time of rows ( data fetched from ]starttime, endtime] )
        endtime : DateTime
                    end time of rows ( data fetched from ]starttime, endtime] )
        parameters : list
                     list of parameters to fetch. If omited all distinct parameters from the first 100 rows are fetched
        locations : list or None
                    location list. If None, * is used
        order : list or None
                Order by columns. If None, order by is not used. ORDER BY don't work for large tables in bq.

        returns : pd dataframe
        """

        if starttime is not None: self.starttime = starttime
        if endtime is not None: self.endtime = endtime
        if loc_col is not None: self.loc_col = loc_col
        if project is not None: self.project = project
        if dataset is not None: self.dataset = dataset
        if table is not None: self.table = table
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = ['*']
        if locations is not None: self.locations = locations
        if order is not None: self.order = order

        # If batch size is set, query data in batches
        if self.batch_size is not None:
            data = pd.DataFrame()
            while True:
                batch = self.get_batch()
                if len(batch) < 1:
                    logging.debug('Found {} rows'.format(len(data)))
                    return data
                data = pd.concat([data, batch])
        # Else do everything once
        else:
            sql = self._format_query()
            logging.debug(sql)
            return self._query(sql)


    def dataset_to_table(self, df, dataset, table):
        """
        Save given dataframe to table

        df      : DataFrame
                  DataFrame to save
        dataset : str
                  dataset name
        table   : str
                  table name
        """

        self._connect()
        dataset_ref = self.client.dataset(dataset)
        table_ref = dataset_ref.table(table)
        job = self.client.load_table_from_dataframe(df, table_ref)
        job.result()
        assert job.state == 'DONE'

    def execute(self, statement):
        """
        Execute single SQL statement in
        a proper manner
        """

        self._connect()
        query_job = self.client.query(statement)
        result = query_job.result()
        return result

    def _query(self, sql):
        """
        Execute query and return results

        sql str sql to execute

        return list of sets
        """
        self._connect()
        result = self.client.query(sql).result()
        return result.to_dataframe()

    def _format_query(self, offset=None):
        timeformat = '%Y-%m-%d %H:%M:%S'
        sql = '''
        SELECT {params} FROM `{project}.{dataset}.{table}`
        WHERE time >= TIMESTAMP("{starttime}")
        AND time < TIMESTAMP("{endtime}") '''.format(
            params=','.join(self.parameters),
            starttime=self.starttime.strftime(timeformat),
            endtime=self.endtime.strftime(timeformat),
            project=self.project,
            dataset=self.dataset,
            table=self.table
            )

        if self.locations is not None:
            sql += ' AND {loc_col} in ({locations})'.format(loc_col=self.loc_col,
            locations='"'+'","'.join(self.locations)+'"')

        if self.order is not None:
            sql += ' ORDER BY {}'.format(','.join(self.order))

        if offset is not None:
            sql += ' LIMIT {limit} OFFSET {offset}'.format(limit=self.batch_size, offset=offset)

        return sql

    def _connect(self):
        """ Create connection if needed """
        self.client = bigquery.Client()
        return self.client
