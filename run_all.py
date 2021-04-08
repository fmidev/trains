#!/usr/local/bin/python3
from multiprocessing import Pool
import subprocess
from datetime import datetime

d=datetime.today().strftime('%Y-%m-%d-%H-%M')
runner='run_in_rahti.sh'
classifier_script='train_classifier.py'
endtime='2015-01-01'
#endtime=None
run_count=6

def wait(pod_name):
    """
    Wait until pod is done
    """
    command = 'while [[ $(oc get pod |grep trains-'+pod_name+' |awk \'{split($0,a," ");print a[3]}\') != "Completed" ]]; do sleep 10; done'
    subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).wait()

def save_logs(pod_name):
    command = f'oc logs trains-{pod_name} > logs/{pod_name}.log'
    subprocess.Popen(command, shell=True)

def run_command(config):
    """
    Run command
    """
    type, file, name = config
    if type == 'classification':
        run_classification_command(file, name)
    else:
        run_regression_command(file, name)

def run_classification_command(file, name):
    """
    Run classification commands
    """
    # RUN
    pod_name = f'train-{d}-{file}-{name}'

    if endtime is None:        
        command = f'bash {runner} -f=dual_dockerfile -n=trains:{pod_name} --command=\'["python","-u","bin/{classifier_script}","--config_filename","cnf/{file}.ini","--config_name","{name}"]\''
    else:
        command = f'bash {runner} -f=dual_dockerfile -n=trains:{pod_name} --command=\'["python","-u","bin/{classifier_script}","--config_filename","cnf/{file}.ini","--config_name","{name}","--endtime","{endtime}"]\''
        
    subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).wait()

    # WAIT and SAVE LOGS
    wait(pod_name)
    save_logs(pod_name)

    # COPY RESULTS TO LOCAL COMPUTER
    command = f'gsutil cp -r gs://trains-data/results/dual/trains_data/{name} results/dual/trains_data/'
    subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).wait()

def run_regression_command(file, name):
    """
    Run regression commands
    """
    pod_name = f'train-{d}-{file}-{name}'

    # RUN
    if endtime is None:
        command = f'bash {runner} -f=dual_dockerfile -n=trains:{pod_name} --command=\'["python","-u","bin/train_scikit.py","--config_filename","cnf/{file}.ini","--config_name","{name}"]\''
    else:
        command = f'bash {runner} -f=dual_dockerfile -n=trains:{pod_name} --command=\'["python","-u","bin/train_scikit.py","--config_filename","cnf/{file}.ini","--config_name","{name}","--endtime","{endtime}"]\''
    subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).wait()

    # WAIT and SAVE LOGS
    wait(pod_name)
    save_logs(pod_name)

    # VISUALISE
    pod_name = f'visualise-{d}-{file}-{name}'
    command = f'bash {runner} -f=dual_dockerfile -n=trains:{pod_name} --command=\'["python","-u","bin/viz_performance.py","--config_filename","cnf/{file}.ini","--config_name","{name}"]\''
    subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)

    # WAIT AND SAVE LOGS
    wait(pod_name)
    save_logs(pod_name)

if __name__ == '__main__':

    configs = [('classification', 'classifiers', 'bayes-all-codes'),
               ('classification', 'classifiers', 'bayes-weather-codes'),
               ('classification', 'classifiers', 'gbdt-weather-codes'),
               ('classification', 'classifiers', 'gbdt-all-codes'),
               ('classification', 'classifiers', 'rfc-all-codes'),
               ('classification', 'classifiers', 'rfc-weather-codes'),
               ('regression', 'lr', 'all-codes'),
               ('regression', 'lr', 'weather-codes'),
               ('regression', 'gbdt', 'all-codes'),
               ('regression', 'gbdt', 'weather-codes'),
               ('regression', 'rf', 'all-codes'),
               ('regression', 'rf', 'weather-codes'),
               ]

    pool = Pool(run_count)
    pool.map(run_command, configs)
    pool.close()
