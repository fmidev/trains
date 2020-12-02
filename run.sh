python -u bin/train_scikit.py --logging_level INFO --config_filename cnf/gp.ini --config_name all-stations-all-codes-v1
python -u  bin/viz_performance.py --config_filename cnf/gp.ini --config_name all-stations-all-codes-v1

# python -u bin/train_dual.py --logging_level INFO --config_filename cnf/dual.ini --config_name bayes-rfr-hubs-all-codes-1
# python -u bin/train_dual.py --logging_level INFO --config_filename cnf/dual.ini --config_name bayes-rfr-hubs-weather-codes-1
# python -u bin/train_dual.py --logging_level INFO --config_filename cnf/dual.ini --config_name bayes-rfr-all-stations-all-codes-1
# python -u bin/train_dual.py --logging_level INFO --config_filename cnf/dual.ini --config_name bayes-rfr-all-stations-weather-codes-1
