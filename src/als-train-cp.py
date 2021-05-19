import argparse
from timeit import default_timer as timer
try:
  import pyspark.sql.functions as F
  from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
  from pyspark.ml.recommendation import ALS, ALSModel
  from pyspark.ml.evaluation import Evaluator
  from pyspark.mllib.evaluation import RankingMetrics
  from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
  from pyspark.sql import DataFrame
except ImportError as e:
  print(e)
  pass


class RankingEvaluator(Evaluator):
  def __init__(self, predictionCol: str = 'prediction', labelCol: str = 'count',
               k: int = 1):
    self.predictionCol = predictionCol
    self.labelCol = labelCol
    self.k = k

  def _get_user_groups(self, data: SparkDataFrame) -> SparkDataFrame:
    return data.select('user_id_index', 'track_id_index',
                       self.labelCol, self.predictionCol).groupBy('user_id_index')

  def _collect(self, user_groups: SparkDataFrame, srcCol: str, targetCol: str,
               do_slice: bool = False) -> SparkDataFrame:
    if do_slice:
      return user_groups \
              .agg(F.slice(F.sort_array(F.collect_list(
                F.struct(srcCol, 'track_id_index')),asc=False), 1, self.k).alias('list')) \
              .withColumn(targetCol, F.col('list.track_id_index')).drop('list')

    return user_groups \
            .agg(F.sort_array(F.collect_list(
              F.struct(srcCol, 'track_id_index')),asc=False).alias('list')) \
            .withColumn(targetCol, F.col('list.track_id_index')).drop('list')

  def _evaluate(self, data: SparkDataFrame) -> float:
    user_groups = self._get_user_groups(data)
    labels = self._collect(user_groups, 'count', 'labels')
    predictions = self._collect(user_groups, 'prediction', 'predictions', do_slice=True)

    results = labels.join(predictions, how='left', on='user_id_index').select('predictions', 'labels')

    ## Make RDDs a little faster, c.f. https://stackoverflow.com/a/42914002/2425365
    results = results.cache()
    #print(f'results show:')
    #results.show()

    ## @FIXME: RankingMetrics may crash weirdly or is slow with an RDD.
    ## PySpark 2.4.0 does not support RankingMetrics with DataFrame, and needs
    ## to be implemented by hand if speedup needed.
    metrics = RankingMetrics(results.rdd)
    metric_val = metrics.precisionAt(self.k)
    #metric_val = metrics.meanAveragePrecision
    #metric_val = metrics.ndcgAt(self.k)
    return metric_val

  def isLargerBetter(self) -> bool:
    return True


def _score(_model: ALSModel, _evaluator: RankingEvaluator, _data: SparkDataFrame) -> float:
  predictions = _model.transform(_data)
  return _evaluator.evaluate(predictions)


def train_als(spark: SparkSession, args: argparse.Namespace):
  print(f'rank={args.rank}')
  als = ALS(maxIter=args.epochs, implicitPrefs=True, nonnegative=True,
            alpha=args.alpha, rank=args.rank, regParam=args.lmbda,
            userCol='user_id_index', itemCol='track_id_index', ratingCol='count',
            coldStartStrategy='drop')

  evaluator = RankingEvaluator(k=args.k)

  #dropThreshold = 10
  train_data = spark.read.parquet(args.train_file)
  #train_data = train_data.filter(train_data['count'] > dropThreshold)
  val_data = spark.read.parquet(args.val_file)
  #val_data = val_data.filter(val_data['count'] > dropThreshold)

  #print('showing train data:')
  #train_data.show()
  #print('showing val data:')
  #val_data.show()

  if args.grid_search:
    rankList = [10]
    alphaList = [4., 5., 6., 7., 8.]
    lambdaList = [0.5, 1.0, 1.5, 2.]

    grid = ParamGridBuilder() \
          .addGrid(als.rank, rankList) \
          .addGrid(als.alpha, alphaList) \
          .addGrid(als.regParam, lambdaList) \
          .build()

    n_train = train_data.count()
    n_val = val_data.count()
    r = n_train / (n_train + n_val)

    tvs = TrainValidationSplit(estimator=als, estimatorParamMaps=grid,
                               evaluator=evaluator,
                               trainRatio=r, seed=args.seed)

    all_data = train_data.union(val_data)

    print(f'Fitting via grid search with train ratio {r:.2f}...')
    s = timer()
    model = tvs.fit(all_data).bestModel
  else:
    print('Fitting a single model...')
    s = timer()
    model = als.fit(train_data)

  print(f'Fit completed in {timer() - s:.4f} s.')
  print(f'Final hyperparameters: rank={model._java_obj.parent().getRank()}; alpha={model._java_obj.parent().getAlpha()}; lmbda={model._java_obj.parent().getRegParam()}')

  if args.save_dir:
    model.write().overwrite().save(f'{args.save_dir}/als_model')
    print(f'Saved final ALSModel to "{args.save_dir}/als_model"')

  # model.recommendForUserSubset(data.select('user_id_index'), args.k)

  ## Evaluation.
  print(f'Starting top-{args.k} evaluation...')

  #s = timer()
  #train_score = _score(model, evaluator, train_data)
  #print(f'Final train score: {train_score:.4f} (took {timer() - s:.4f} s)')
  
  #s = timer()
  #val_score = _score(model, evaluator, val_data)
  #print(f'Final val score: {val_score:.4f} (took {timer() - s:.4f} s)')

  if args.test_file:
    test_data = spark.read.parquet(args.test_file)
    #test_data = test_data.filter(test_data['count'] > dropThreshold)
    #print('showing test data:')
    #test_data.show()

    s = timer()
    test_score = _score(model, evaluator, test_data)
    print(f'Final test score: {test_score:.4f} (took {timer() - s:.4f} s)')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Train ALS model.')
  parser.add_argument('--train-file', type=str, dest='train_file', metavar='',
                      help='Path to train file.')
  parser.add_argument('--val-file', type=str, dest='val_file', metavar='',
                      help='Path to validation file.')
  parser.add_argument('--test-file', type=str, dest='test_file', metavar='',
                      help='Path to test file.')
  parser.add_argument('--save-dir', type=str, dest='save_dir', metavar='',
                      default=None, help='Path to output directory for model storage.')
  parser.add_argument('--seed', type=int, dest='seed', metavar='',
                      default=None, help='Seed value.')
  parser.add_argument('--epochs', type=int, dest='epochs', metavar='',
                      default=10, help='Maximum number of training iterations.')
  parser.add_argument('--rank', type=int, dest='rank', metavar='',
                      default=10, help='Feature size.')
  parser.add_argument('--lmbda', type=float, dest='lmbda', metavar='',
                      default=1.0, help='Regularization parameter.')
  parser.add_argument('--alpha', type=float, dest='alpha', metavar='',
                      default=1.0, help='Confidence weighting parameter.')
  parser.add_argument('--k', type=int, dest='k', metavar='',
                      default=500, help='For ranking evaluation metrics.')
  parser.add_argument('--grid-search', action='store_true', help='Flag for grid search.')
  args = parser.parse_args()

  sc = SparkSession.builder.appName('ALSTrainer').getOrCreate()
  train_als(sc, args)
