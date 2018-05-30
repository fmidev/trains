FROM python:3.6

RUN chgrp -R 0 /a && \
    chmod -R g=u /a

RUN mkdir /a
WORKDIR /a

#RUN groupadd --system --gid 92 www && \
#    useradd --system --uid 320 --gid www www #--no-log-init www
# RUN chown www:www /a

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_CTYPE=C.UTF-8

# RUN apt-get update
RUN apt-get install -y curl

# RUN apt-get install -y wget graphviz libtiff5-dev libjpeg8-dev zlib1g-dev libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev

RUN pip install --upgrade requests pandas tensorflow google-api-python-client google-cloud sklearn scipy keras boto3  google-api-python-client google-auth-httplib2

# mlfdb
RUN mkdir /tmp/a && git clone https://github.com/fmidev/ml_feature_db.git /tmp/a && pip install /tmp/a/api && rm -rf /tmp/a

## Google sdk python libraries
RUN echo "deb http://packages.cloud.google.com/apt cloud-sdk-jessie main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install -y google-cloud-sdk

# google-cloud-sdk-app-engine-python google-cloud-sdk-app-engine-python-extras

# ENV PYTHONPATH "/usr/lib/google-cloud-sdk:/usr/lib/google-cloud-sdk/lib:/usr/lib/google-cloud-sdk/lib/yaml"

ADD bin /a/bin
ADD cnf /a/cnf

RUN mkdir -p /home/www/.config/gcloud/logs && chown -R www:www /home/www

USER www:www

# ADD cnf/TRAINS-full.json /root/

RUN gcloud config set proxy/type http && gcloud config set proxy/address wwwproxy.fmi.fi && gcloud config set proxy/port 8080
RUN gcloud auth activate-service-account --key-file /a/cnf/TRAINS-full.json
RUN gcloud config set project trains-197305


CMD python -u bin/get_prediction.py --logging_level INFO
