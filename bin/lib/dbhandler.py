# -*- coding: utf-8 -*-
"""
Database handler
"""
import sys
import re
import psycopg2
import logging
import os
import numpy as np
import datetime
from collections import defaultdict
from configparser import ConfigParser

class DBHandler(object):
    """
    Handle database connection and provide
    methods to insert and read storm objects to database
    """

    def __init__(self, debug=False, training=False, config_filename = None):

        if config_filename is None:
            config_filename = os.path.dirname(os.path.abspath(__file__))+'/../../cnf/pred_db.ini'
        self.config_filename = config_filename

        self._connect()

    def _connect(self):
        """ Create connection if needed """
        params = self._config(self.config_filename)
        self.conn = psycopg2.connect(**params)
        return self.conn

    def _config(self, config_filename, section='gemini'):
        parser = ConfigParser()
        parser.read(config_filename)

        db = {}
        if parser.has_section(section):
            params = parser.items(section)
            for param in params:
                db[param[0]] = param[1]
        else:
            raise Exception('Section {0} not found in the {1} file'.format(section, self.config_filename))

        if parser.has_section('tables'):
            params = parser.items('tables')
            tables = {}
            for param in params:
                tables[param[0]] = param[1]
            self.tables = tables
        else:
            raise Exception('Section {0} not found in the {1} file'.format('tables', self.config_filename))

        return db

    def insert_forecasts(self, forecast):
        """
        Insert forecasts to the db

        forecast : dict
                   forecast in format {name: [(station_id, timestamp, forecast_value, analysist_time), ...], ...}
        """

        for name, time_list in forecast.items():

            loc_id = self.get_location_by_name(name)

            sql = "INSERT INTO {schema}.{table} (station_id, timestamp, forecast_value, analysistime) VALUES ".format(schema = self.tables['schema'], table=self.tables['for_table'])

            first = True
            for values in time_list:

                if not first: sql += ', '
                else: first = False

                sql += "({station_id}, TO_TIMESTAMP({timestamp}), {value}, TO_TIMESTAMP({analysistime})) """.format(station_id=loc_id, timestamp=int(values[0]), value=values[3], analysistime=int(values[4]))

            try:
                logging.debug(sql)
                self.execute(sql)
            except psycopg2.IntegrityError as e:
                logging.error(e)

    def clean_forecasts(self, tolerance):
        """
        Clean old values from db

        tolerance : timedelta
                    how old times (from current time) are cleand
        """

        now = datetime.datetime.now()
        remove_from = now - tolerance

        sql = "DELETE FROM {schema}.{table} WHERE analysistime < '{t}' AND timestamp < '{t}'".format(schema=self.tables['schema'], table=self.tables['for_table'], t=remove_from)

        logging.debug(sql)

        self.execute(sql)

    def insert_stations(self, locations):
        """
        Add locations to the db

        locations : dict
                    location information in following format: [{name: {'lat':xx, 'lon':xx}, ...]
        """
        logging.info('Adding {} locations to db...'.format(len(locations)))

        ids = []
        sql = "INSERT INTO {schema}.{table} (station_name, geom) VALUES ".format(schema=self.tables['schema'], table=self.tables['loc_table'])
        first = True
        for loc, latlon in locations.items():
            if not first:
                sql = sql+', '
            else:
                first = False
            sql = sql + "('{name}', ST_GeomFromText('POINT({lon} {lat})', 4326))".format(name=loc, lat=latlon['lat'], lon=latlon['lon'])
        self.execute(sql)

    def get_locations_by_name(self, names):
        """
        Find location ids by names

        names : list
                list of location names
        """
        sql = "SELECT id, station_name FROM {}.{} WHERE station_name IN ({})".format(self.tables['schema'], self.tables['loc_table'], '\''+'\',\''.join(names)+'\'')

        return self._query(sql)

    def get_location_by_name(self, name):
        """
        Find location id by name

        name : str
               location name

        return id (int) or None
        """
        sql = "SELECT id FROM {schema}.{table} WHERE station_name='{name}'".format(schema=self.tables['schema'], table=self.tables['loc_table'], name=name)
        res = self._query(sql)
        if len(res) > 0:
            return int(res[0][0])

        return None

    def execute(self, statement):
        """
        Execute single SQL statement in
        a proper manner
        """

        self._connect()
        with self.conn as conn:
            with conn.cursor() as curs:
                curs.execute(statement)

    def _query(self, sql):
        """
        Execute query and return results

        sql str sql to execute

        return list of sets
        """
        self._connect()
        with self.conn as conn:
            with conn.cursor() as curs:
                curs.execute(sql)
                results = curs.fetchall()
                return results
