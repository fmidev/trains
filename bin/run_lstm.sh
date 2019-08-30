tensorboard --logdir=/board --host 0.0.0.0 --port 8888 &

# python -u bin/train.py --logging_level DEBUG --config_filename cnf/lstm.ini --config_name test
# python -u bin/train.py --logging_level INFO --config_filename cnf/lstm.ini --config_name all_params_sample_2
# python -u bin/train.py --logging_level INFO --config_filename cnf/lstm.ini --config_name all_params_sample_dense_1
# python -u bin/train.py --logging_level INFO --config_filename cnf/lstm.ini --config_name all_params_sample_dense_2
# python -u bin/train.py --logging_level INFO --config_filename cnf/lstm.ini --config_name all_params_sample_dense_imputed_1
# python -u bin/train.py --logging_level INFO --config_filename cnf/lstm.ini --config_name all_params_sample_dense_imputed_avg_1
# python -u bin/train.py --logging_level INFO --config_filename cnf/lstm.ini --config_name all_params_sample_dense_imputed_avg_2
# python -u bin/train.py --logging_level INFO --config_filename cnf/lstm.ini --config_name all_params_sample_dense_imputed_avg_3
python -u bin/train.py --logging_level INFO --config_filename cnf/lstm.ini --config_name all_24_avg_1
