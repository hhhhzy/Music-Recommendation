import argparse
from typing import Optional
try:
  from pyspark.sql import SparkSession
  import pyspark.sql.functions as F
except ImportError:
  # May not work without spark-submit.
  pass


class MSDSubSampler:
  DATADIR = 'hdfs:/user/bm106/pub/MSD'

  def __init__(self, out_dir):
    self.spark = SparkSession.builder.appName(self.__class__.__name__).getOrCreate()
    self.out_dir = out_dir

  def topk(self, k: int = 10000, seed: Optional[int] = None, mode: str = 'train'):
    '''Extracts top-k interacting users and creates a random 80/20 split among them.

    Arguments:
      k: Number of top users to fetch.
      seed: Seed for random split.
      mode: Relevant file (train/validation/test) to pick.
    '''
    df = self.spark.read.parquet(f'{self.DATADIR}/cf_{mode}.parquet')
    df.printSchema()
    # df.createOrReplaceTempView('ui')

    print(f'{df.select("user_id").distinct().count()} training users with {df.count()} interactions.')
    print(f'Selecting top {k}...')

    top_users = df.groupBy('user_id').count().orderBy(F.col('count').desc()).limit(k).select('user_id')
    # top_users.show()

    top_interactions = df.join(top_users, on='user_id', how='inner')
    print(f'Filtered {top_interactions.count()} interactions.')

    sub_train, sub_val = top_interactions.randomSplit([0.8, 0.2], seed=seed)
    print(f'(Sub)Train: {sub_train.count()}; (Sub)Validation: {sub_val.count()}.')

    out_suffix = f'top{k}_{seed if seed is not None else "unseeded"}'
    sub_train.write.mode('overwrite').parquet(f'{self.out_dir}/cf_{mode}_subtrain_{out_suffix}.parquet')
    print(f'Written {sub_train.count()} rows to "{self.out_dir}/cf_{mode}_subtrain_{out_suffix}.parquet".')
    sub_val.write.mode('overwrite').parquet(f'{self.out_dir}/cf_{mode}_subval_{out_suffix}.parquet')
    print(f'Written {sub_val.count()} rows to "{self.out_dir}/cf_{mode}_subval_{out_suffix}.parquet".')


if __name__ == "__main__":
  '''
  See `python subsample.py -h` for the help prompt related.
  See https://spark.apache.org/docs/latest/configuration.html for spark-submit configurations.

  Example command to split among top-10 interactive users.
  ```shell
  spark-submit --conf spark.executor.memory=4g subsample.py \
    -o hdfs:/user/${USER}/quarantini -s 1004 -k 10
  ```

  See defaults below.
  '''
  parser = argparse.ArgumentParser(description='Subsample MSD dataset.')
  parser.add_argument('-k', type=int, dest='k', metavar='',
                      default=100, help='Number of top interacting users')
  parser.add_argument('-s', '--seed', type=int, dest='seed', metavar='',
                      default=None, help='Subsampling seed')
  parser.add_argument('-o', '--output', type=str, dest='out_dir', metavar='',
                      default=None,
                      help='Target HDFS directory to save generated files.')
  args = parser.parse_args()

  MSDSubSampler(args.out_dir).topk(k=args.k, seed=args.seed)