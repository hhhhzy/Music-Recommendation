# Recommender Systems

The final report is [here](./report/report.pdf).

## Usage

### Subsampling

For the subsampling script, see [subsample.sh](./scripts/subsample.sh) for usage. The subsampling logic is implemented
in [subsample.py](./src/subsample.py).

### ALS Model Training

The training using ALS is implemented in [als-train.py](./src/als-train.py). See [als-train.sh](./scripts/als-train.sh) for sample usage to train and evaluate a single model.

### Popularity-based baseline model

For the Popularity-based baseline model script, see [bias.sh](./scripts/bias.sh) for usage. The bias model logic is implemented
in [bias.py](./src/bias.py).

## Utility Commands

### HDFS Access

To give read and execute access for your HDFS folder to the user you want.

```shell
hfs -setfacl -R -m user:<user_id>:rwx </path/to/hdfs/folder>
```

Check that the permissions have been sucessfully granted by:

```shell
hfs -getfacl -R </path/to/hdfs/folder>
```

### Access full pre-processed data
View the files with `hfs -ls ${PATH}`. Full pre-processed files are located here:

```shell
hdfs:///user/yej208/quarantini/data/cf_test_processed.parquet
hdfs:///user/yej208/quarantini/data/cf_train_processed.parquet
hdfs:///user/yej208/quarantini/data/cf_val_processed.parquet
```
