# -*- coding: utf-8 -*-
"""
BigQuery handler
"""
import sys
import re
import logging
import os
import numpy as np
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

        if config_filename is None:
            config_filename = os.path.dirname(os.path.abspath(__file__))+'/../../cnf/big_query.ini'
        self.config_filename = config_filename
        _ = self._config(self.config_filename)

        self._connect()

    def _connect(self):
        """ Create connection if needed """
        #params = self._config(self.config_filename)
        self.client = bigquery.Client()
        return self.client

    def _config(self, config_filename):
        parser = ConfigParser()
        parser.read(config_filename)

        if parser.has_section('tables'):
            params = parser.items('tables')
            tables = {}
            for param in params:
                tables[param[0]] = param[1]
            self.tables = tables
        else:
            raise Exception('Section {0} not found in the {1} file'.format('tables', self.config_filename))

        return tables

    def get_rows(self,
                 starttime,
                 endtime,
                 loc_col='loc_name',
                 project=None,
                 dataset=None,
                 table=None,
                 parameters=['*'],
                 locations=None):
        """
        Get all feature rows from given dataset

        dataset_name : str
                       dataset name
        starttime : DateTime
                    start time of rows ( data fetched from ]starttime, endtime] )
        endtime : DateTime
                    end time of rows ( data fetched from ]starttime, endtime] )
        parameters : list
                     list of parameters to fetch. If omited all distinct parameters from the first 100 rows are fetched

        returns : xx
        """

        if project is None: project = self.tables['project']
        if dataset is None: dataset = self.tables['dataset']
        if table is None: table = self.tables['feature_table']

        timeformat = '%Y-%m-%d %H:%M:%S'

        sql = '''
        SELECT {params} FROM `{project}.{dataset}.{table}`
        WHERE time >= TIMESTAMP("{starttime}") AND time < TIMESTAMP("{endtime}") '''.format(params=','.join(parameters),
                   starttime=starttime.strftime(timeformat),
                   endtime=endtime.strftime(timeformat),
                   project=project,
                   dataset=dataset,
                   table=table)

        if locations is not None:
            sql += ' AND {loc_col} in ({locations})'.format(loc_col=loc_col,
                                                            locations='"'+'","'.join(locations)+'"')
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
