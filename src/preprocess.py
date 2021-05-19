import argparse
from typing import Optional
from functools import reduce
try:
  from pyspark.sql import SparkSession

  # needed for preprocessing/indexing
  from pyspark.ml import Pipeline
  from pyspark.ml.feature import StringIndexer
  from pyspark.sql.functions import col
  from pyspark.sql.types import IntegerType
  from pyspark.sql import DataFrame
  from pyspark.sql.functions import lit
  from pyspark.sql.DataFrame import filter
  from pyspark.sql.DataFrame import collect
except ImportError:
  # Pyspark imports will not work on Peel without spark-submit
  pass

class Preprocessor:
  DATADIR = 'hdfs:/user/bm106/pub/MSD'

  TRAINFILE = 'cf_train.parquet'
  VALFILE = 'cf_validation.parquet'
  TESTFILE = 'cf_test.parquet'

  FILES = [TRAINFILE, VALFILE, TESTFILE]

  TRAIN_OUTFILE = 'cf_train_processed_droplowcounts.parquet'
  VAL_OUTFILE = 'cf_val_processed_droplowcounts.parquet'
  TEST_OUTFILE = 'cf_test_processed_droplowcounts.parquet'

  def __init__(self, outdir: str):
    self.spark = SparkSession.builder.appName('Preprocessor').getOrCreate()
    self.outdir = outdir

  def union(self, dataframes):
    return reduce(DataFrame.union, dataframes)

  def index(self):
    # read in original files
    train_original = self.spark.read.parquet(f'{self.DATADIR}/{self.TRAINFILE}')
    val_original = self.spark.read.parquet(f'{self.DATADIR}/{self.VALFILE}')
    test_original = self.spark.read.parquet(f'{self.DATADIR}/{self.TESTFILE}')

    # append original source as a column to each dataframe
    train = train_original.withColumn('source', lit(0))
    val = val_original.withColumn('source', lit(1))
    test = test_original.withColumn('source', lit(2))
    
    # verify that column has been appended
    print(f'>>> train set')
    train.show()
    print(f'>>> val set')
    val.show()
    print(f'>>> test set')
    test.show()
 
    # merge the files into one for running StringIndexer on it
    dataframes = [train, val, test]
    merged = self.union(dataframes)
    print(f'>>> merged dataframes')
    merged.show() 

    # in the merged dataset, drop all records with counts lower than dropThreshold=2
    dropThreshold = 2
    merged = merged.filter(merged.count < dropThreshold).collect()

    cols_to_index = ['user_id', 'track_id']
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid='skip') for col in cols_to_index]
    pipeline = Pipeline(stages=indexers)

    indexer_model = pipeline.fit(merged)
    indexed = indexer_model.transform(merged)

    # split the dataframes again based on original source
    train_indexed = indexed[indexed['source'] == 0]
    val_indexed = indexed[indexed['source'] == 1]
    test_indexed = indexed[indexed['source'] == 2]

    # format each one to only select necessary columns
    train_indexed = self.format_indexed_df(train_indexed)
    val_indexed = self.format_indexed_df(val_indexed)
    test_indexed = self.format_indexed_df(test_indexed)

    # verify format
    print(f'>>> indexed train set')
    train_indexed.show()
    print(f'>>> indexed val set')
    val_indexed.show()
    print(f'>>> indexed test set')
    test_indexed.show()

    # check counts of each split
    train_count = train_indexed.count()
    val_count = val_indexed.count()
    test_count = test_indexed.count()

    print(f'dataset entry counts with dropThreshold={dropThreshold}:')
    print(f'>>> train set count: {train_count}')
    print(f'>>> val set count: {val_count}')
    print(f'>>> test set count: {test_count}')

    test_indexed.write.parquet(f'{self.outdir}/{self.TRAIN_OUTFILE}')
    val_indexed.write.parquet(f'{self.outdir}/{self.VAL_OUTFILE}')
    test_indexed.write.parquet(f'{self.outdir}/{self.TEST_OUTFILE}')

  def format_indexed_df(self, df):
    df = df[['user_id_index', 'count', 'track_id_index']]
    df = df.withColumn('user_id_index', col('user_id_index').cast(IntegerType())) \
           .withColumn('track_id_index', col('track_id_index').cast(IntegerType()))
    return df

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Preprocess the data.')
  parser.add_argument('-o', '--outdir', type=str, dest='outdir', metavar='',
                      default='hdfs:/user/yej208/quarantini/data',
                      help='Target HDFS directory to save generated files.')
  args = parser.parse_args()
  print(args)

  Preprocessor(args.outdir).index()


