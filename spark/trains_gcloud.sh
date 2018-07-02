#!/bin/bash
# script to create a BigQuery dataset, create cluster,
# run the data retrieval program and finally delate the cluster

echo "Creating bq dataset trains_all"

bq mk trains_all


echo "Creating cluster"
gcloud dataproc clusters create spark3 --master-boot-disk-size 64 --image-version 1.2 --bucket trains-data --master-machine-type n1-standard-4  --num-workers 2 --worker-boot-disk-size 32 --worker-machine-type n1-standard-2 --region europe-north1 --zone europe-north1-c  --properties spark:spark.executorEnv.PYTHONHASHSEED=0 --initialization-actions gs://trains-data/bin/install-py3-dataproc.sh


echo "running trains_get_surface_flash_obs_bq.py"

gcloud dataproc jobs submit pyspark trains_get_surface_flash_obs_bq_fmi.py --cluster spark3 --bucket trains-data --py-files gs://trains-data/bin/spark_logging.py --files gs://trains-data/cnf/parameters_shorten.txt  --region europe-north1

echo "deleting cluster"

gcloud dataproc clusters delete spark3 --region europe-north1


