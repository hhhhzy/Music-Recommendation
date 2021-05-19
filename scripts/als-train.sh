#!/usr/bin/env bash

SRCDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/../src"

#SPARK_ARGS='--executor-memory=4g'
SPARK_ARGS='--executor-memory=16g --executor-cores=12 --conf spark.yarn.submit.waitAppCompletion=false'
#SPARK_ARGS='--executor-memory=4g --conf spark.yarn.submit.waitAppCompletion=false'

## Subsampled dataset
#DATADIR="hdfs:/user/${USER}/quarantini"
#TRAIN_FILE=cf_train_subtrain_top10_1004.parquet
#VAL_FILE=cf_train_subval_top10_1004.parquet

## Full dataset
DATADIR="hdfs:/user/${USER}/quarantini/data"
TRAIN_FILE=cf_train_processed.parquet
VAL_FILE=cf_val_processed.parquet
TEST_FILE=cf_test_processed.parquet

CKPT_ID=$(date +%F-%s)

pushd "${SRCDIR}"

## Train single model.
spark-submit ${SPARK_ARGS} \
  als-train.py --train-file="${DATADIR}/${TRAIN_FILE}" \
               --val-file="${DATADIR}/${VAL_FILE}" \
               --test-file="${DATADIR}/${TEST_FILE}" \
               --epochs=1 --rank=10 --lmbda=1.0 --alpha=10.0 --k=500
               --save-dir="${DATADIR}/ckpt-${CKPT_ID}"

## Grid search (define grid in code).
#spark-submit ${SPARK_ARGS} \
#  als-train.py --train-file="${DATADIR}/${TRAIN_FILE}" \
#               --val-file="${DATADIR}/${VAL_FILE}" \
#               --test-file="${DATADIR}/${TEST_FILE}" \
#               --save-dir="${DATADIR}/ckpt-${CKPT_ID}" \
#               --epochs=10 --k=15 --grid-search

popd
