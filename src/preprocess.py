try:
  from pyspark.sql import SparkSession

  # needed for preprocessing/indexing
  from pyspark.ml import Pipeline
  from pyspark.ml.feature import StringIndexer
  from pyspark.sql.functions import col
  from pyspark.sql.types import IntegerType
except ImportError:
  # Pyspark imports will not work on Peel without spark-submit
  pass

class Preprocessor:
  DATADIR = 'hdfs:/user/yej208/quarantini'
  #DATADIR = 'hdfs:/user/bm106/pub/MSD'

  OUTPUTDIR = 'hdfs:/user/yej208/quarantini'
  TRAIN_OUTFILE = 'cf_train_subtrain_top10_1004_processed.parquet'
  VAL_OUTFILE = 'cf_train_subval_top10_1004_processed.parquet'

  # Run on smaller train/val files on personal HDFS directory
  TRAINFILE = 'cf_train_subtrain_top10_1004.parquet'
  VALFILE = 'cf_train_subval_top10_1004.parquet'

  # Run on master train/val/test files on public HDFS directory
  #TRAINFILE = 'cf_train.parquet'
  #VALFILE = 'cf_validation.parquet'
  #TESTFILE = 'cf_test.parquet'

  FILES = [TRAINFILE, VALFILE] # include TESTFILE if needed

  def __init__(self):
    self.spark = SparkSession.builder.appName('Preprocessor').getOrCreate()


  '''
  See following documentation re: "handleInvalid" attribute in StringIndexer
  https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html#pyspark.ml.feature.StringIndexer.handleInvalid
  '''
  def index(self):
    train = self.spark.read.parquet(f'{self.DATADIR}/{self.TRAINFILE}')
    val = self.spark.read.parquet(f'{self.DATADIR}/{self.VALFILE}')

    cols_to_index = ['user_id', 'track_id']
    indexers = [StringIndexer(inputCol=col, outputCol=col+"_index", handleInvalid='skip') for col in cols_to_index]
    pipeline = Pipeline(stages=indexers)

    indexer_model = pipeline.fit(train)
    train_indexed = indexer_model.transform(train)
    val_indexed = indexer_model.transform(val)

    print(f'>>> train indexed:')
    train_indexed.show()
    print(f'>>> val indexed:')
    val_indexed.show()

    train_formatted = self.format_indexed_df(val_indexed)
    val_formatted = self.format_indexed_df(train_indexed)

    print(f'>>> train formatted:')
    train_formatted.show()
    print(f'>>> train formatted:')
    val_formatted.show()

    train_formatted.write.parquet(f'{self.OUTPUTDIR}/{self.TRAIN_OUTFILE}')
    val_formatted.write.parquet(f'{self.OUTPUTDIR}/{self.VAL_OUTFILE}')


  def format_indexed_df(self, df):
    df = df[['user_id_index', 'count', 'track_id_index']]
    df = df.withColumn('user_id_index', col('user_id_index').cast(IntegerType())) \
           .withColumn('track_id_index', col('track_id_index').cast(IntegerType()))
    return df


  '''
  Simple demo implementation of StringIndexer
  '''
  def index_simple(self):
    train = self.spark.read.parquet(f'{self.DATADIR}/{self.TRAINFILE}')
    val = self.spark.read.parquet(f'{self.DATADIR}/{self.VALFILE}')

    cols_to_index = ['user_id', 'track_id']
    indexer = StringIndexer(inputCol='user_id', outputCol='user_id_index')

    indexer_model = indexer.fit(train)
    train_indexed = indexer_model.transform(train)
    val_indexed = indexer_model.transform(val)

    train_indexed.show()
    val_indexed.show()



if __name__ == "__main__":
  Preprocessor().index()


