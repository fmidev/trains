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
        self.reason_code_table = None
        self.where = None
        self.only_winters = False
        self.reason_codes_exclude = None
        self.reason_codes_include = None

    def set_params(self,
                   batch_size=None,
                   loc_col=None,
                   project=None,
                   dataset=None,
                   table=None,
                   parameters=None,
                   locations=None,
                   where=None,
                   order=None,
                   only_winters=None,
                   reason_code_table=None,
                   reason_codes_exclude=None,
                   reason_codes_include=None):
        """
        Set params to be used for get_batch function
        """
        #self.starttime = starttime
        #self.endtime = endtime
        if self.batch_size is not None:
            self.batch_size = batch_size

        if loc_col is not None:
            self.loc_col = loc_col

        if project is not None:
            self.project = project

        if dataset is not None:
            self.dataset = dataset

        if table is not None:
            self.table = table

        if only_winters is not None:
            self.only_winters = only_winters

        if where is not None:
            self.where = where

        if parameters is not None:
            self.parameters = parameters

        if locations is not None:
            self.locations = locations

        if order is not None:
            self.order = order

        if reason_code_table is not None:
            self.reason_code_table = reason_code_table
            self.reason_codes_exclude = reason_codes_exclude
            self.reason_codes_include = reason_codes_include

        self.batch_num = 0


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

    def get_rows(self, *args, **kwargs):
                 # starttime=None,
                 # endtime=None,
                 # loc_col=None,
                 # project=None,
                 # dataset=None,
                 # table=None,
                 # parameters=['*'],
                 # locations=None,
                 # where=None,
                 # order=None,
                 # only_winters=False):
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

        # if starttime is not None: self.starttime = starttime
        # if endtime is not None: self.endtime = endtime
        # if loc_col is not None: self.loc_col = loc_col
        # if project is not None: self.project = project
        # if dataset is not None: self.dataset = dataset
        # if table is not None: self.table = table
        # if parameters is not None:
        #     self.parameters = parameters
        # else:
        #     self.parameters = ['*']
        # if locations is not None: self.locations = locations
        # if order is not None: self.order = order
        # self.only_winters = only_winters
        # self.where = where

        self.set_params(**kwargs)
        self.starttime, self.endtime = args

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

    def nparray_to_table(self, values, columns, project, dataset, table):
        """
        Save given numpy array to table

        values  : lst
                  list of Numpy arrays
        columns : lst
                  list of lists of column names
        dataset : str
                  dataset name
        table   : str
                  table name
        """

        table = table.replace('-','_')

        dfs = []
        i = 0
        for data in values:
            dfs.append(pd.DataFrame(data, columns=columns[i]))
            i += 1
        df = pd.concat(dfs, axis=1)

        self.delete_table(project, dataset, table)

        self.dataset_to_table(df, dataset, table)

    def delete_table(self, project, dataset, table):
        """
        Delete big query table

        dataset : str
                  dataset name
        table   : str
                  table name
        """
        self._connect()
        table_id = '{project}.{dataset}.{table}'.format(project=project,
                                                        dataset=dataset,
                                                        table=table)
        self.client.delete_table(table_id, not_found_ok=True)
        logging.info("Deleted table '{}'.".format(table_id))

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
        """
        Format query based on options

        TODO Exclude: T1,P2,R1,R3,V4,K6
        spec. include I1,I2
        """
        timeformat = '%Y-%m-%d %H:%M:%S'

        parameters = parameters = ['a.{}'.format(param) for param in self.parameters]
        from_clause = '`{project}.{dataset}.{table}` a'.format(project=self.project,
                                                               dataset=self.dataset,
                                                               table=self.table)
        if self.reason_code_table is not None:
            parameters += ['count']
            from_clause = '''
            `{project}.{dataset}.{table}` a
            LEFT JOIN `{project}.{dataset}.{reason_table}` b
            ON (a.train_type=b.train_type AND a.trainstation=b.end_station AND a.time=b.time)
            '''.format(project=self.project,
                       dataset=self.dataset,
                       table=self.table,
                       reason_table=self.reason_code_table)

        sql = '''
        SELECT {params} FROM {from_clause}
        WHERE a.time >= TIMESTAMP("{starttime}")
        AND a.time < TIMESTAMP("{endtime}") '''.format(
            from_clause = from_clause,
            params=','.join(parameters),
            starttime=self.starttime.strftime(timeformat),
            endtime=self.endtime.strftime(timeformat)
            )

        if self.reason_codes_exclude is not None:
            sql += ' AND (b.code IS NULL OR b.code NOT IN ("{}"))'.format('","'.join(self.reason_codes_exclude))
        elif self.reason_codes_include is not None:
            sql += ' AND (b.code IN ("{}"))'.format('","'.join(self.reason_codes_include))
        elif self.reason_code_table is not None:
            raise Exception('If reason_code_table is set reason_codes_exclude or reason_codes_include has to be also set.')

        if self.locations is not None:
            sql += ' AND a.{loc_col} in ({locations})'.format(loc_col=self.loc_col,
            locations='"'+'","'.join(self.locations)+'"')

        if self.only_winters:
            sql += ' AND EXTRACT(MONTH FROM a.time) IN (1,2,3,4,11,12)'

        if self.where is not None:
            for col, value in self.where.items():
                sql +=' AND {}={}'.format(col, value)

        if self.order is not None:
            sql += ' ORDER BY a.{}'.format(','.join(self.order))

        if offset is not None:
            sql += ' LIMIT {limit} OFFSET {offset}'.format(limit=self.batch_size, offset=offset)

        return sql

    def _connect(self):
        """ Create connection if needed """
        self.client = bigquery.Client()
        return self.client
