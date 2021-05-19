#!/usr/bin/env bash

SRCDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/../src"

SPARK_ARGS='--executor-memory=4g'
# SPARK_ARGS=${SPARK_ARGS}' --conf spark.yarn.submit.waitAppCompletion=false'

DATADIR="hdfs:/user/${USER}/quarantini"
TRAIN_FILE=cf_train_subtrain_top10_1004.parquet
VAL_FILE=cf_train_subval_top10_1004.parquet
# TODO: test file, add later for full dataset
TEST_FILE=

pushd "${SRCDIR}"

spark-submit ${SPARK_ARGS} \
  bias.py -tr="${DATADIR}/${TRAIN_FILE}" \
               -v="${DATADIR}/${VAL_FILE}" \
               -du 100 -di 300 --k=5

popd
