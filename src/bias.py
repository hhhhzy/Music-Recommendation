#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import time
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
from pyspark.sql.functions import mean as _mean, col, sort_array, collect_list, struct, udf
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, FloatType, IntegerType
import argparse


class Bias:
    '''
        popularity baseline model.

        The predictor is based on the following algorithm:
        𝑏𝑢𝑖=𝜇+𝑏𝑢+𝑏𝑖

        where 𝜇 is the global mean rating. 𝑏𝑖 is the item bias, and 𝑏𝑢 is the user bias.
        hey are computed as follows:

        𝑏𝑖=∑𝑢∈R(𝑖)(𝑟𝑢𝑖−𝜇) / 𝜆2+|R(𝑖)|.

        𝑏𝑢=∑𝑖∈R(𝑢)(𝑟𝑢𝑖−𝜇−𝑏𝑖) / 𝜆3+|R(𝑢)|.

        Args:
            item: whether to compute item biases
            users: whether to compute user biases
            damping(number or tuple):
                Bayesian damping to apply to computed biases.  Either a number, to
                damp both user and item biases the same amount, or a (user,item) tuple
                providing separate damping values.
        Attributes:
            mean: The global mean rating.
            item_offsets_(DataFrame): The item offsets (𝑏𝑖 values)
            user_offsets_(DataFrame): The user offsets (𝑏𝑢 values)
    '''

    def __init__(self, item=True, users=True, damping=0.0):
        self.item = item
        self.users = users
        if isinstance(damping, tuple):
            self.damping = damping
            self.user_damping, self.item_damping = damping
        else:
            self.damping = damping
            self.user_damping = damping
            self.item_damping = damping

        if self.user_damping < 0:
            raise ValueError("user damping must be non-negative")
        if self.item_damping < 0:
            raise ValueError("item damping must be non-negative")

    def torating(self, data):
        describe = data.describe(['count']).collect()
        stddev = float(describe[2]['count'])
        mean = float(describe[1]['count'])
        max = mean + 2 * stddev
        min = mean - 2 * stddev
        mul = float((max - min) / 100)

        @udf(FloatType())
        def counttrans(count):
            if count < min:
                count = min
            elif count > max:
                count = max

            percentiles = (count - min) / mul
            if percentiles > 100:
                percentiles = 100
            elif percentiles < 0:
                percentiles = 0
            result = percentiles / 100 * 5

            return result if result > 0 else 0.05

        result = data.withColumn('count', counttrans(col('count')))

        return result



    def fit(self, ratings):
        """
        Train the bias model on some rating data.
        Args:
            ratings (DataFrame): the dataframe of song dataset.
        Returns:
            Bias: the fit bias object.
        """
        ratings_stat = ratings.select(_mean(col('count'))).collect()
        self.mean = ratings_stat[0]['avg(count)']
        print(f'global mean: {self.mean}')
        nrate = ratings.withColumn('minus', col('count') - self.mean).select('minus', 'user_id', 'track_id')

        if self.item:
            self.item_offsets_ = self._group(nrate.groupby('track_id'), self.item_damping, 'track_id')
        else:
            self.item_offsets_ = None

        if self.users:
            if self.item_offsets_ is not None:
                nrate \
                    .join(self.item_offsets_, on='track_id', how='inner') \
                    .withColumn('minus', col('minus') - col('track_id_off')) \
                    .select('minus', 'user_id', 'track_id')

                self.user_offsets_ = self._group(nrate.groupby('user_id'), self.user_damping, 'user_id')
        else:
            self.user_offsets_ = None

        return self

    def predict(self, ratings):
        """
        Do predict for the giving item and user.
        The method use the result of fit() and do not recalculate the 𝑏𝑖 and 𝑏𝑢
        Args:
            ratings (DataFrame): the dataframe of song dataset you want to predict,
                            must have track_id column and user_id column.
        Returns:
            rvsp(DataFrame): the DataFrame with predict result, which in prediction column
        """

        if self.item_offsets_ is not None:
            rvsp = self.item_offsets_.join(ratings, on='track_id', how='left') \
                .fillna({'track_id_off': 0}) \
                .withColumn('prediction', self.mean + col('track_id_off'))
        if self.user_offsets_ is not None:
            rvsp = self.user_offsets_.join(rvsp, on='user_id', how='left') \
                .fillna({'user_id_off': 0}) \
                .withColumn('prediction', col('prediction') + col('user_id_off')) \
                .select('prediction', 'user_id', 'track_id', 'count')
        return rvsp

    def recommendForUserSubset(self, data: SparkDataFrame, k: int = 50, colName: str = 'prediction'):

        stringIndexer = StringIndexer(inputCol="track_id", outputCol="track_id_indexed", handleInvalid="skip", stringOrderType="frequencyDesc")
        model = stringIndexer.fit(data)
        data = model.transform(data).withColumn('track_id_indexed', col('track_id_indexed').cast(IntegerType()))

        limit = udf(lambda z: z[0:k], ArrayType(IntegerType(), True))
        if colName == 'count':
            recommend = data.groupBy('user_id').agg(
                sort_array(collect_list(struct(colName, 'track_id_indexed')), asc=False).alias('list')) \
                .withColumn('labels', col('list.track_id_indexed')).drop('list')
        else:
            recommend = data.groupBy('user_id').agg(
                sort_array(collect_list(struct(colName, 'track_id_indexed')), asc=False).alias('list')) \
                .withColumn('labels', col('list.track_id_indexed')).drop('list')

        return recommend

    def score(self, data: SparkDataFrame, k: int = 50) -> dict:
        predictions = self.recommendForUserSubset(self.predict(data), k, 'prediction') \
            .withColumnRenamed('labels', 'prediction')
        label = self.recommendForUserSubset(data, k, 'count')
        results = label.join(predictions, on='user_id', how='left')
        # results.select('prediction', 'labels').show(n=30, truncate=False, vertical=True)
        _array = udf(lambda z: list() if z is None else z, ArrayType(IntegerType(), True))
        result_rdd = results.select('prediction', 'labels').withColumn('labels', _array(col('labels'))).withColumn('prediction', _array(col('prediction'))).rdd
        # print(results.select('prediction', 'labels').rdd.take(5))
        metrics = RankingMetrics(result_rdd)

        ###
        ## NOTE: RankingEvaluator does not work with PySpark 2.4.0.
        ## See https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RankingEvaluator.html

        # evaluator = RankingEvaluator(predictionCol='predictions', labelCol='labels')
        # value = evaluator.evaluate(results,
        #                            {evaluator.metricName: 'precisionAtK', evaluator.k: k})

        return {
            'map': metrics.meanAveragePrecision,
            f'ndcgAt{k}': metrics.ndcgAt(k),
            f'precisionAt{k}': metrics.precisionAt(k)
        }

    def save(self, spark, path):
        data = ([(self.mean,)])

        schema = StructType([StructField("mean", FloatType(), True)])

        df = spark.createDataFrame(data=data, schema=schema)
        df.write.mode('overwrite').parquet(path + 'mean.parquet')
        self.user_offsets_.write.mode('overwrite').parquet(path + 'user_offsets_.parquet')
        self.item_offsets_.write.mode('overwrite').parquet(path + 'item_offsets_.parquet')
        print(f'save {self.mean}, {self.user_offsets_}, {self.item_offsets_} to {path}')

    def load(self, spark, path):
        model = spark.read.parquet(path + 'mean.parquet')
        model_stat = model.select('*').collect()
        self.mean = model_stat[0]['mean']
        self.user_offsets_ = spark.read.parquet(path + 'user_offsets_.parquet')
        self.item_offsets_ = spark.read.parquet(path + 'item_offsets_.parquet')
        print(f'load {self.mean}, {self.user_offsets_}, {self.item_offsets_} from {path}')

        return self

    def _group(self, group, damping, id):
        if damping is not None and damping > 0:
            count = group \
                .count() \
                .withColumn('add', col('count') + self.item_damping)

            sum = group \
                .sum('minus') \
                .withColumnRenamed('sum(minus)', 'sum')

            off = sum \
                .join(count, on=id, how='inner') \
                .withColumn(id + '_off', col('sum') / col('add')) \
                .select(id, id + '_off')

            return off
        else:
            return group.mean('minus')


def main(spark, train_dir, test_dir, val_dir, damping, k, save_dir, load_dir, torating: bool=False):
    """
    test of Class Bias
    """
    test = Bias(item=True, users=True, damping=damping)
    raw_train_data = spark.read.parquet(train_dir)
    train_data = test.torating(raw_train_data) if torating else raw_train_data
    if (load_dir is not None):
        predictions = test.load(spark, load_dir)
    else:
        start = time.time()
        predictions = test.fit(train_data)
        end = time.time()
        print(f'fit function spends {end - start}')
        if (save_dir is not None):
            predictions.save(spark, save_dir)

    results = {}

    train_metrics = predictions.score(train_data, k)
    for m, v in train_metrics.items():
        results[f'train/{m}'] = v

    if val_dir:
        val_metrics = predictions.score(spark.read.parquet(val_dir), k)
        for m, v in val_metrics.items():
            results[f'val/{m}'] = v

    if test_dir:
        test_metrics = predictions.score(spark.read.parquet(test_dir), k)
        for m, v in test_metrics.items():
            results[f'test/{m}'] = v

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    '''

    test of Class Bias
  use this command to run this script
  ```shell
  spark-submit --conf spark.executor.memory=4g bias.py \
   -tr hdfs:/user/${USER}/quarantini/${TrainFile} \
    -te hdfs:/user/${USER}/quarantini/${TestFile} \
     -du ${USER_DAMPING} -di ${ITEM_DAMPING}
  ```
  '''
    parser = argparse.ArgumentParser(description='popularity baseline model bias.py test')
    parser.add_argument('-tr', '--train', type=str, dest='train_dir', metavar='',
                        default=None, help='Train data directory')
    parser.add_argument('-te', '--test', type=str, dest='test_dir', metavar='',
                        default=None, help='Test data directory')
    parser.add_argument('-v', '--val', type=str, dest='val_dir', metavar='',
                        default=None, help='Val data directory')
    parser.add_argument('-du', '--user_dampling', type=int, dest='user_damping', metavar='',
                        default=None,
                        help='damping of bias model')
    parser.add_argument('-di', '--item_dampling', type=int, dest='item_damping', metavar='',
                        default=None,
                        help='item damping of bias model')
    parser.add_argument('-k', '--k', type=int, dest='k', metavar='',
                        default=None,
                        help='top k recommendation')
    parser.add_argument('-l', '--load', type=str, dest='load_dir', metavar='',
                        default=None,
                        help='load file directory')
    parser.add_argument('-s', '--save', type=str, dest='save_dir', metavar='',
                        default=None,
                        help='save  file directory')
    parser.add_argument('-r', '--rating', type=int, dest='rating', metavar='',
                        default=None,
                        help='transfer to rating or not 1 for ture')
    args = parser.parse_args()

    spark = SparkSession.builder.appName('bias_test').config('spark.blacklist.enabled', False).getOrCreate()

    damping = (args.user_damping, args.item_damping)

    transferrate = True if args.rating == 1 else False

    main(spark=spark, train_dir=args.train_dir, test_dir=args.test_dir, val_dir=args.val_dir, damping=damping, k=args.k,
         save_dir=args.save_dir, load_dir=args.load_dir, torating=transferrate)

