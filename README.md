Research code for predicting train delays caused by weather.

You are most probably NOT able to run this code without author's help. You may anyway check some particular details about the implementation.

Some general notions about running the code:
- The data is assumed to be in Google Cloud BigQuery. Correct credentials has to be provided. For example (in dockerfile):
```
ADD cnf/TRAINS-xxx.json /a/cnf/TRAINS-xxx.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/a/cnf/TRAINS-xxx.json
```
- Most of the code is made to be ran in Docker containers. Dockerfiles are named *xx_dockerfile*. In most cases, there are dedicated dockerfile for each method/task.
- docker-compose.yaml provides also receipt for running (most) scripts.

## Directory Structure
```
.                        | root folder, contain dockerfiles and docker-compose
|- bin                   | executable scripts
|  |- lib                | libraries
|  |  |- model_functions | required functions to load data into the model
|- cnf                   | config files
|- data                  | subset of data as csv
|- labels_vis_1_0_arch   | visualisation of label data
|- labels_vis_1_1        | visualisation of label data
|- spark                 | data fetching and pre-processing done with spark
|- models                | trained models are saved here
|  |- model_type         | automatically generated, for example lstm
|  |  |- data_set_name   | automatically generated, for example trains_2009_18_wo_testset
|  |  |  |- config_name  | automatically generated, for example 16_params_1
|- results               | validation results are saved here
|  |- model_type         | automatically generated, for example lstm
|  |  |- data_set_name   | automatically generated, for example trains_2009_18_wo_testset
|  |  |  |- config_name  | automatically generated, for example 16_params_1
|  |- material           | related material like presentations etc.
|  |- model              | "operationally" ran model is stored here
```

## Data
Because of large amount of data, full data can't be added here. Subset of data is added for small tests and development.

### Delay data
Data fields are following:
* date
* Start hour (for example: 7 = 7:00:00 – 7:59:59.999 transactions)
* Start station
* End station
* Train type (K=intercity, L=commute, T=cargo, M=else)
* Sum of total delayed minutes
* Sum of total ahead of time minutes    
* Sum of additional delayed minutes
* Sum of additional ahead of time minutes    
* Train count during the hour between the stations

### Reason codes

Fields for a_b file:
* date
* hour
* train type
* start station
* end station
* reason code
* count of transactions with given reason code

Fields for b_c file:
* date
* hour
* train type
* start station
* reason code
* end station
* count of transactions with given reason code

Consult https://www.digitraffic.fi/rautatieliikenne/ for more information. Especially links https://rata.digitraffic.fi/api/v1/metadata/detailed-cause-category-codes?show_inactive=true and https://rata.digitraffic.fi/api/v1/metadata/third-cause-category-codes?show_inactive=true are most probably useful. Detailed reason codes have chanbged in the beginning ofg 2017 but first two characters should be coherent.

## Config
Configs are stored in `cnf/` directory config files named by used method. Config files are read by `bin/lib/config.py`. Example:
```
parser = argparse.ArgumentParser()
options = parser.parse_args()
_config.read(options)
```
All values are stored in options object.

Possible config values are (most of them are not used in every method):

| Value | Affect | Default | Notice |
|-------|--------|---------|--------|
| y_avg | Calculate average delay based on train count | false | - |
| y_avg_hours | Average delay with n hours (before) | 0 | - |
| cv | Run k-fold cross-validation | false | - |
| pca | Run PCA before training | false | - |
| whiten | Whiten data during PCA | false | - |
| normalize | Normalise data before training | false | - |
| impute | Impute missing values before training | false | - |
| n_loops | ? | ? | - |
| batch_size | - | ? | - |
| tf | ? | false | - |
| day_step | ? | 5000 | - |
| hour_step | ? | 0 | - |
| epochs | - | 3 | - |
| alpha | - | 0. | Used in linear regression |
| eta0 | - | 0. | Used in linear regression |
| power_t | - | 0. | Used in linear regression |
| alpha_1 | - | 0. | Used in ARD |
| alpha_2 | - | 0. | Used in ARD |
| lambda_1 | - | 0. | Used in ARD |
| lambda_2 | - | 0. | Used in ARD |
| fit_intercept | - | false | Used in ARD |
| copy_x | - | false | Used in ARD |
| threshold_lambda | - | 0. | Used in ARD |
| n_samples | - | 0 | Used in ARD |
| bootstrap | - | - | Used in RF |
| n_estimators | - | - | Used in RF |
| min_sample_split | - | - | Used in RF |
| min_samples_leaf | - | - | Used in RF |
| max_depth | - | - | Used in RF |
| time_steps | int | - |  Used in LSTM |
| cell_size | int | - | Used in LSTM |
| lr | learning lrate, float | - | Used in LSTM |
| n_hidden | int | - | Used in LSTM |
| quantile | - | - | Used in LSTM |
| p_drop | drop layer probability | - | Used in LSTM |
| slow | if true, proceed hour by hour while training | false | Used in LSTM |
| n_components | number of components | 4 | Used in BayesianGaussianMixture (BGM) |
| kmeans | Calculate kmeans and use feature's distance to cluster centers  in training | false | - |
| pca_components | If larger than 0, pre-process the data with PCA and keep corresponding components | 0 | - |
| dbscan | - | 0 | - |
| feature_selection | boolean | false | If set, feature selection based on coef_ or feature_importances_ is done |
| month | boolean | false | If set, month is extracted for each timestamp and used in training as a feature param |
| gmm_classifier | String | - | Used to determine filename of pre-trained GMM model (used in classify_scikit_gmm.py)|
| gmm_params | String | - | Params which are used in GMM model (used in classify_scikit_gmm.py) |
| only_winters | boolean | false | Read data only from winter months (Nov - Apr) |
| class_limit | int | 4 | Pick only smaller classes (Used in autoencoders) |


## Individual Scripts
Here are some cherry-picked notations about selected scripts. Mainly to help myself to run the scripts again.

### Classification

Classify datasets using LSTM.

Running example:
```
python -u bin/classify.py --config_filename cnf/lstm.ini --config_name test
```

### Copy Dataset

Copy dataset to another name (in BigQuery).

For example to copy dataset from trains-1.0 to trains-1.2 (as BigQuery dataset:
```
python -u bin/copy_dataset.py --src_dataset trains-1.0 --dst_dataset trains-1.2 --starttime 2010-01-01 --endtime 2013-01-10 --src_parameters cnf/parameters.txt --dst_parameters cnf/parameters_shorten.txt
```

### Train

To train regression, several scripts may be used. *train.py* is used to train LSTM model. *trains_scikit.py* is used to train all scikit models. *train_representation.py* is have to be used if one wants to use kmeans or dbscan preprocessing (scikit models supported). *train_gpflow.py* is used to train gpflow models.

Model is saved into directory **models/*model_type*/*data_set_name*/*config_name*** (for example: `models/lstm/trains_2009_18_wo_testset/test`). If google cloud bucket is configured in config file, model is uploaded into the bucket as well.

Examples:

Train random forest (scikit model):
```
python -u bin/train_scikit.py --config_filename cnf/rf.ini --config_name test
```

Train linear regression (scikit model):
```
python -u bin/train_scikit.py --config_filename cnf/lr.ini --config_name test
```

Train LSTM (tensorflow):
```
python -u bin/train.py --config_filename cnf/lstm.ini --config_name test
```

### Visualising Results
After training, model performance can be visualised using three selected months (dataset and table are hard coded in the script).

Model is assumed to be in directory **models/*model_type*/*data_set_name*/*config_name*** (for example: `models/lstm/trains_2009_18_wo_testset/test`). If model is NOT in correct folder, it is loaded from google cloud bucket (set in config file by param *gs_bucket*).

Results are saved to directory: **results/*model_type*/*data_set_name*/*config_name*/validation_testset** (for example: `models/lstm/trains_2009_18_wo_testset/test/validation_testset`). If google cloud bucket is configured in config file, results are uploaded into the bucket as well.

Example:
```
python -u bin/viz_performance.py --config_filename cnf/rf.ini --config_name 16_params_1
```
One may configure following parameters as well:
- *starttime*  | start time
- *endtime*    | end time
- *model_path* | path where model is saved
- *model_file* | specific file name of the saved model
- *stations*   | if set, only listed stations are handled, separated with comma
- *stations_file* | filename of stations.json file
- *only_avg* | only average of all stations are visualised
- *output_path* | output path

More complicated examples:
```
python -u bin/viz_performance.py --config_filename cnf/rf.ini --config_name 16_params_1 --starttime 2010-01-01 --endtime 2010-12-31 --only_avg 1
```
```
python -u bin/viz_performance.py --config_filename cnf/rf.ini --config_name 16_params_1 --starttime 2010-01-01 --endtime 2010-12-31 -- stations PSL,OL
```
### Balance Dataset

Run script `balance_dataset.py`. For example:

```
python -u bin/balance_dataset.py --logging_level INFO --src_dataset trains_data  --src_table features_wo_testset --dst_dataset trains_data --dst_table features_balanced_wo_testset
```

Note, at least 64Gi memory is required to run this for the whole dataset.

You may add --no_balance flag to just classify dataset without balancing.
