# Recommender Systems

## Example Commands

### HDFS Access

To provide read and execute access to HDFS folder.

```shell
hfs -setfacl -R -m user:<user_id>:r-x </path/to/hdfs/folder>
```

### Subsampling from training data

To subsample top-k (`-k`) users from the training data set and an 80/20 split,
with a fixed seed (`-s`).

```shell
spark-submit --conf spark.executor.memory=4g src/subsample.py \
  -o hdfs:/user/${USER}/quarantini -s 1004 -k 10
```


### Popularity-based baseline model

To use the Popularity-based baseline model tran(`-tr`) and test(`-te`),
based on user damping(`-du`) and item damping(`-di`)

```shell
  spark-submit --conf spark.executor.memory=4g bias.py \
   -tr hdfs:/user/${USER}/quarantini/${TrainFile} \
    -te hdfs:/user/${USER}/quarantini/${TestFile} \
     -du ${USER_DAMPING} -di ${ITEM_DAMPING}
```