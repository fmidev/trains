#!/bin/bash

# curl -fsSL get.docker.com -o get-docker.sh
# sudo sh get-docker.sh
# sudo docker pull tervo/ml:cpu
# sudo usermod -aG docker $USER

apt-get update
apt-get install -y --reinstall wget libtiff5-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev python3 python-gdal python3-pip postgresql libpq-dev postgresql-client postgresql-client-common python-urllib3
pip3 install unicodecsv pyproj requests psycopg2-binary numpy google-cloud scipy scikit-learn PySocks
pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.6.0-cp34-cp34m-linux_x86_64.whl
pip3 install keras boto3

echo "export PYSPARK_PYTHON=python3" | tee -a  /etc/profile.d/spark_config.sh  /etc/*bashrc /usr/lib/spark/conf/spark-env.sh
echo "export PYTHONHASHSEED=0" | tee -a /etc/profile.d/spark_config.sh /etc/*bashrc /usr/lib/spark/conf/spark-env.sh
echo "spark.executorEnv.PYTHONHASHSEED=0" >> /etc/spark/conf/spark-defaults.conf

mkdir /tmp/a
git clone https://github.com/fmidev/ml_feature_db.git /tmp/a
pip3 install /tmp/a/api

cat << EOF | sudo tee -a /etc/profile.d/custom_env.sh /etc/*bashrc >/dev/null
export SPARK_HOME=/usr/lib/spark/
EOF

pip3 install --upgrade requests




