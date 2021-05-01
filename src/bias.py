#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql.functions import mean as _mean, col
from pyspark.ml.evaluation import RegressionEvaluator
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
            rvsp = ratings.join(self.item_offsets_, on='track_id', how='left') \
                .fillna({'track_id_off': 0}) \
                .withColumn('prediction', self.mean + col('track_id_off'))
        if self.user_offsets_ is not None:
            rvsp = rvsp.join(self.user_offsets_, on='user_id', how='left') \
                .fillna({'user_id_off': 0}) \
                .withColumn('prediction', col('prediction') + col('user_id_off')) \
                .select('prediction', 'user_id', 'track_id', 'count')
        return rvsp

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


def main(spark, train_dir, test_dir, damping):
    """
    test of Class Bias
    """

    rating = spark.read.parquet(train_dir)

    rating_test = spark.read.parquet(test_dir)

    test = Bias(item=True, users=True, damping=damping)
    predictions = test.fit(rating).predict(rating_test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))


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
    parser.add_argument('-du', '--user_dampling', type=int, dest='user_damping', metavar='',
                        default=None,
                        help='damping of bias model')
    parser.add_argument('-di', '--item_dampling', type=int, dest='item_damping', metavar='',
                        default=None,
                        help='item damping of bias model')
    args = parser.parse_args()

    spark = SparkSession.builder.appName('bias_test').getOrCreate()

    damping = (args.user_damping, args.item_damping)

    main(spark=spark, train_dir=args.train_dir, test_dir=args.test_dir, damping=damping)

