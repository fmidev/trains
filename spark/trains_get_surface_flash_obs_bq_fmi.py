#!/usr/bin/python3
""" 
This program fetches the surface observation data and flash data from 
the Smartmet server for a period of 2009-11-30 to 2014-06-01. Combine
the above data with train types and delays and saves to BigQuery.
"""
from __future__ import absolute_import

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
import time
import json
import itertools
import numpy as np
from io import StringIO
from socket import timeout
import requests
import codecs
import multiprocessing
from io import StringIO
import pprint
import subprocess
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Retrieve Surface and Flash data") \
    .getOrCreate()


#from pyspark.sql.functions import concat, col, lit, expr
from pyspark.sql.functions import to_utc_timestamp
from pyspark.sql.types import *
from pyspark.sql import functions as f
from pyspark.sql.window import Window as w
from pyspark.sql.functions import when  


start_time = time.time()


# Use the Cloud Storage bucket for temporary BigQuery export data used
# by the InputFormat. This assumes the Cloud Storage connector for
# Hadoop is configured.
bucket = spark._jsc.hadoopConfiguration().get('fs.gs.system.bucket')
project = spark._jsc.hadoopConfiguration().get('fs.gs.project.id')

# Output Parameters.
# REMEMBER to give bq mk station_weather_flash_delay_dataset

output_dataset = 'trains_all'
output_table = 'features'

#output_dataset = 'station_weather_flash_delay_dataset'
#output_table = 'station_weather_flash_delay_output'

# Logging from the main and from the functions
try:
    spark.addPyFile('spark_logging.py')
except:
    # Probably running this locally. Make sure to have spark_logging in the PYTHONPATH
    print("except case")
    pass
finally:
    import spark_logging as logging




def read_parameters(filename):
    """
    Read parameter from file (one param per line)
    filename : str
    Each line contains params and the name
    """
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    params, names = [], []
    for line in lines:
        param, name = line.split(';')
        params.append(param)
        names.append(name)
    return params, names
 

def split_timerange_v0(starttime, endtime, days=1, hours=0, timestep=60):
    """
    Split timerange to n days
    
    starttime : Datetime
                starttime
    endtime : Datetime
              endtime
    days : int
           time chunk length, days
    hours : int
           time chunk length, hours
    timestep : int
               timestep in minutes

    return list of tuples [(starttime:Datetime, endtime:Datetime)]
    """
    chunks = []
    start = starttime
    end = start + timedelta(days=days, hours=hours)
    while end <= endtime:        
        chunks.append((start + timedelta(minutes=timestep), end))
        start = end
        end = start + timedelta(days=days, hours=hours)
    return chunks


def split_timerange(starttime, endtime, days=1, hours=0, timestep=60):
    """
    Split timerange to n days
    
    starttime : Datetime
                starttime
    endtime : Datetime
              endtime
    days : int
           time chunk length, days
    hours : int
           time chunk length, hours
    timestep : int
               timestep in minutes

    return list of tuples [(starttime:Datetime, endtime:Datetime)]
    """
    chunks = []
    start = starttime
    end = start + timedelta(days=days, hours=hours)
    while end <= endtime: 
        startt=start + timedelta(minutes=timestep)
        startstr = startt.strftime('%Y-%m-%dT%H:%M:%S')
        endstr = end.strftime('%Y-%m-%dT%H:%M:%S')
        chunks.append(("&starttime="+startstr + "&endtime="+endstr))
        start = end
        end = start + timedelta(days=days, hours=hours)
    return chunks




def requests_retry_session(
    retries=5,
    backoff_factor=0.5,
    status_forcelist=(400, 408,500, 502, 503, 504),
    session=None,
):
     """
     Setting the parameters for the url request
     """
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry,pool_connections=5,pool_maxsize=5)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def readurl(URL):
     """
     This function retrieves the data from the URL using the requests function
     URL: station name + the actual url
     text: string of the fetched data
     """
    station,url = URL.split('#')
    log3=logging.getLogger("executor")
    try:
        response = requests_retry_session().get(
            url,
            timeout = 6000
        )
    except Exception as x:
        print('It failed :(', x.__class__.__name__)
        log3.info("Retrieval Error Station {}".format(station))
        return -1
    else:
        data = response.content.decode('utf-8')
        if len(data) == 0:
            return -1
        return data

def read_from_URL(URL):
    """
    This function forms the actual URL from the base URL adding the dates
    for which the data is to be retrieved. It fetches data for the n number 
    of stations in  a certain radius. Take the values of the best station
    based on the least number of null/nan values.

    URL: station name + the base url
    text: strings of data
    """
    station,urlb = URL.split('#')
    log2=logging.getLogger("executor")
    starttime='2009-11-30'
    endtime='2014-06-01'
    start = datetime.strptime(starttime, '%Y-%m-%d')
    end = datetime.strptime(endtime, '%Y-%m-%d')
    dates = split_timerange(start, end, days=120)
    t=map(lambda d: readurl('%s%s' % (URL, d)), dates)
    tmp=list(filter(lambda x: x != -1, t))
    if len(tmp) != len(dates):
        log2.info("Nodata or Incomplete data Station {}".format(station))       
        return -1
    data="".join(tmp)
    text=data.replace('\n', ',{0}\n'.format(str(station)))
    log2.info("Station {}".format(station))
    tmp = text.rstrip().split('\n')
    min_index = np.argmin(list(map(lambda x: x.count('nan'),tmp)))
    wstation=tmp[min_index].split(',')[1]
    t = list(filter(lambda x: wstation in x, tmp))
    text = '\n'.join(map(str, t)) + '\n'
    return text


def getfashdates(starttime, endtime):
    """
    The flash data is taken only from 1st June to 31 August of each year.
    This function returns the list of periods from June to August for
    every year between starttime and endtime 
    """
    log5=logging.getLogger("executor")
    dates = []
    start = datetime.strptime(starttime, '%Y-%m-%d')
    end = datetime.strptime(endtime, '%Y-%m-%d')
    startyear = int(start.strftime('%Y'))
    endyear = int(end.strftime('%Y'))
    startmonth = int(start.strftime('%m'))
    endmonth = int(end.strftime('%m'))
    startday=int(start.strftime('%d'))
    endday=int(end.strftime('%d'))

    completeyears = endyear - startyear - 1
    if startmonth >= 6 and startmonth <=8:
        if startmonth == 6:
            t=datetime(startyear,6,startday,0,0)
        elif startmonth > 6:
            t=datetime(startyear,startmonth,startday,0,0)
        stime=t.strftime('%Y-%m-%dT%H:%M:%S')
        t=datetime(startyear,8,31,23, 59, 59)
        etime=t.strftime('%Y-%m-%dT%H:%M:%S')
        se='&starttime='+stime+'&endtime='+etime
        dates.append(se)
    if completeyears > 0:
        for year in range(startyear+1, endyear):
            t=datetime(year,6,1,0,0)
            stime=t.strftime('%Y-%m-%dT%H:%M:%S')
            t=datetime(year,8,31,23, 59, 59)
            etime=t.strftime('%Y-%m-%dT%H:%M:%S')
            se='&starttime='+stime+'&endtime='+etime
            dates.append(se)
    if endmonth >= 6:
        if endmonth >8:
            st=datetime(endyear,6,1,0,0)
            et=datetime(endyear,8,31,23, 59, 59)
        elif endmonth == 8:
            st=datetime(endyear,6,1,0,0)
            et=datetime(endyear,8,endday,23, 59, 59)
        elif endmonth >= 6:
            st=datetime(endyear,6,1,0,0)
            et=datetime(endyear,endmonth,endday,23, 59, 59)
        stime=st.strftime('%Y-%m-%dT%H:%M:%S')
        etime=et.strftime('%Y-%m-%dT%H:%M:%S')
        se='&starttime='+stime+'&endtime='+etime
        dates.append(se)
    return(dates)


def getFlash(URL):
    """
    Makes the URL requests from the base URL adding the dates and 
    returns the flash data 
    """
    station,urlb = URL.split('#')
    log4=logging.getLogger("executor")
    starttime='2009-11-30'
    endtime='2014-06-01'
    dates = getfashdates(starttime, endtime)
    t=map(lambda d: readurl('%s%s' % (URL, d)), dates)
    tmp=list(filter(lambda x: x != -1, t))
    if len(tmp) == 0:
        log4.info("Nodata {}".format(station))     
        return -1
    data="".join(tmp)
    text=data.replace('\n', ',{0}\n'.format(str(station)))
    log4.info("Station {}".format(station))
    return text



def main():
    """
    Get observations near locations from SmartMet Server
    
    Data start and end time and timestep is fetched from the
    data. Dataset is assumed coherent in means of time and
    locations. I.e. timestep is assumed to be constant between start
    and end time. 
    """
    log1=logging.getLogger("driver")


    output_directory = 'gs://{}/hadoop/tmp/bigquery/pyspark_output'.format(bucket)
    output_files = output_directory + '/part-*'

    # The trains stations data in stations.json
    # The type of trains and their delay in gratu_a_b_2010-14.csv

    JSON_PATH="gs://trains-data/data/stations.json"
    CSV_PATH="gs://trains-data/data/full/gratu_a_b_2010-14.csv"

    train_stations_df = spark.read \
        .json(JSON_PATH)

    # parameters for weather data to be fetched from Smartmet server
    params, names = read_parameters('parameters_shorten.txt')

    # base URL for the surface data
    baseurl = 'http://data.fmi.fi/fmi-apikey/9fdf9977-5d8f-4a1f-9800-d80a007579c9/timeseries?format=ascii&separator=,&producer=fmi&tz=local&timeformat=xml&timestep=60&numberofstations=5&maxdistance=100000&param={params}'.format(params=','.join(params))


    urlist= train_stations_df.rdd.flatMap(lambda x : ['%s#%s&latlons=%s,%s' % (x.stationShortCode,baseurl,x.latitude,x.longitude)]).repartition(16)


    data = urlist.map(read_from_URL)\
                 .filter(lambda x: x != -1)\
                 .flatMap(lambda x:x.splitlines())\
                 .map(lambda x: x.split(','))

    newColumns=names+["trainstation"]
    schemaString = ' '.join(str(x) for x in newColumns)

    fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
    schema = StructType(fields)

    # Apply the schema to the RDD.
    station_weather_df = spark.createDataFrame(data, schema)
    station_weather_df = station_weather_df.withColumn("time", to_utc_timestamp(station_weather_df.time, "%Y-%m-%dT%H"))

    # calculate max_precipitation 3h and max_precipitation6h
    col="max_precipitation1h"

    # to change the "no precipiation values" -1.0 to 0.0
    station_weather_df = station_weather_df.withColumn(col, f.when(station_weather_df[col] == -1.0, 0.0).otherwise(station_weather_df[col]))


    # using window functions to calculate the precipitation for the
    # previous 3 hours and 6 hours
    w3 = w.partitionBy("trainstation")\
          .orderBy(station_weather_df["time"])\
          .rowsBetween(-2,0)
        
    station_weather_df =station_weather_df.withColumn("max_precipitation3h",f.sum("max_precipitation1h").over(w3))


    w6 = w.partitionBy("trainstation")\
          .orderBy(station_weather_df["time"])\
          .rowsBetween(-5,0)
    
    station_weather_df =station_weather_df.withColumn("max_precipitation6h",f.sum("max_precipitation1h").over(w6))

    # making the surface observation dataframe

    cols = station_weather_df.columns  # list of all columns
    for col in cols:
        station_weather_df = station_weather_df.fillna({col:"-99"})
        station_weather_df = station_weather_df.withColumn(col, f.when(station_weather_df[col].isin("null", "nan", "NaN", "NULL"),"-99").otherwise(station_weather_df[col]))

    log1.info("Retrieved surface data")

   
   
    ## Get flash data

    baseurl = 'http://data.fmi.fi/fmi-apikey/9fdf9977-5d8f-4a1f-9800-d80a007579c9/timeseries?param=time,peak_current&producer=flash&tz=local&timeformat=xml&format=ascii&separator=,'
    urlist= train_stations_df.rdd.flatMap(lambda x : ['%s#%s&latlon=%s,%s:30' % (x.stationShortCode,baseurl,x.latitude,x.longitude)])
   
    data = urlist.map(getFlash)\
            .filter(lambda x: x != -1)\
            .flatMap(lambda x:x.splitlines())\
            .map(lambda x: x.split(','))


    schemaString = 'time peakcurrent trainstation'

    fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split()]
    schema = StructType(fields)

    flash_df =  spark.createDataFrame(data, schema)
    flash_df = flash_df.withColumn("time", to_utc_timestamp(flash_df.time, "%Y%m%dT%HMS"))

    # find the count of flashes in each hour
    extended = (flash_df
                .withColumn("date", f.col("time").cast("date"))
                .withColumn("hour", f.hour(f.col("time"))))

    flash_aggs = extended.groupBy("trainstation", "date", "hour").count()

    flash_aggs = flash_aggs.withColumn('time', f.concat(f.col("date"), f.lit("T"), f.col("hour")))

    flash =flash_aggs.withColumn('time',to_utc_timestamp(flash_aggs.time,"%Y-%m-%dT%H")).select("time", f.col("count").alias("flashcount"),"trainstation")

    log1.info("Retrieved flash data")
    
    # Combining surface and flash data

    cond = [flash.time == station_weather_df.time, flash.trainstation == station_weather_df.trainstation ]
    
    station_weather_flash_df = station_weather_df.alias('a').join(flash.alias('b'),cond, 'outer').select('a.*', 'b.flashcount').fillna({'flashcount':'0'}) 

    # Reading the train type and delay data
    df = spark.read \
            .csv(CSV_PATH)

    # combining the date and time columns and selecting the relevant columns
    df = df.withColumn('t', f.concat(f.col("_c0"), f.lit("T"), f.col("_c1"))).select("t","_c3", "_c4", "_c9", "_c7", "_c5")


    # converting the time to utc timestamp and adding 1 hour
    df = df.withColumn('t',to_utc_timestamp(df.t,"%Y-%m-%dT%H") + f.expr('INTERVAL 1 HOUR'))

    trains_df = df.select(f.col("t").alias("time"),f.col("_c3").alias("trainstation"), f.col("_c4").alias("train_type"), f.col("_c9").alias("train_count"), f.col("_c7").alias("total_delay"), f.col("_c5").alias("delay"))

    # Combining the weather data both surface and flash with
    #he train delay and type data
    cond = [trains_df.time == station_weather_flash_df.time, trains_df.trainstation == station_weather_flash_df.trainstation ]

    trains_station_weather_flash_delay_df = trains_df.join(station_weather_flash_df, cond).drop(station_weather_flash_df.time).drop(station_weather_flash_df.trainstation)

    log1.info("Created the dataframe with train delay and weather observations Finished!\n")

    # Saving the data to BigQuery

    (trains_station_weather_flash_delay_df
     .write.format('json').save(output_directory))

    # Shell out to bq CLI to perform BigQuery import.
    subprocess.check_call(
        'bq load --source_format NEWLINE_DELIMITED_JSON '
        '--replace '
        '--autodetect '
        '{dataset}.{table} {files}'.format(
            dataset=output_dataset, table=output_table, files=output_files
        ).split())

    # Manually clean up the staging_directories, otherwise BigQuery
    # files will remain indefinitely.
    output_path = spark._jvm.org.apache.hadoop.fs.Path(output_directory)
    output_path.getFileSystem(spark._jsc.hadoopConfiguration()).delete(
        output_path, True)

    
    elapsed_time = time.time() - start_time
    log1.info("Elapsed time to retreive train delay and observation data and save to bq {:10.3f}".format(elapsed_time))

if __name__ == "__main__":
    log = logging.getLogger("driver")
    log.info("Start! Create a dataframe with train delay and weather info \n ")
    main()
