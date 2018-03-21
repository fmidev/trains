#!/bin/bash

# curl -fsSL get.docker.com -o get-docker.sh
# sudo sh get-docker.sh
# sudo docker pull tervo/ml:cpu
# sudo usermod -aG docker $USER

sudo apt-get update
sudo apt-get install -y wget libtiff5-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev python3 python-gdal python3-pip postgresql libpq-dev postgresql-client postgresql-client-common
sudo pip3 install unicodecsv pyproj requests psycopg2-binary numpy google-cloud

echo "export PYSPARK_PYTHON=python3" | sudo tee -a  /etc/profile.d/spark_config.sh  /etc/*bashrc /usr/lib/spark/conf/spark-env.sh
echo "export PYTHONHASHSEED=0" | sudo tee -a /etc/profile.d/spark_config.sh /etc/*bashrc /usr/lib/spark/conf/spark-env.sh
echo "spark.executorEnv.PYTHONHASHSEED=0" >> /etc/spark/conf/spark-defaults.conf

mkdir /tmp/a
git clone https://github.com/fmidev/ml_feature_db.git /tmp/a
sudo pip3 install /tmp/a/api





