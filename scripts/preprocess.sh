#!/usr/bin/env bash

SRCDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/../src"

SPARK_ARGS='--executor-memory=4g'
# SPARK_ARGS=${SPARK_ARGS}' --conf spark.yarn.submit.waitAppCompletion=false'

pushd "${SRCDIR}"

## NOTE: Change the values as needed.
spark-submit ${SPARK_ARGS} \
  preprocess.py -o hdfs:/user/${USER}/quarantini/data

popd
