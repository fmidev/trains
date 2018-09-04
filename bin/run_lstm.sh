tensorboard --logdir=/tmp/lstm --host 0.0.0.0 --port 80 &
#python -u bin/train.py --logging_level INFO --config_filename cnf/lstm.ini --config_name all_params_sample_2
python -u bin/train.py --logging_level INFO --config_filename cnf/lstm.ini --config_name all_params_sample_3
