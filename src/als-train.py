try:
    from pyspark.sql import SparkSession

    # needed for ALS training
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.recommendation import ALS
    from pyspark.sql import Row
except ImportError:
  # Pyspark imports will not work on Peel without spark-submit
    pass

class ALSTrainer:
    DATADIR = 'hdfs:/user/yej208/quarantini'
    TRAINFILE = 'cf_train_subtrain_top10_1004_processed.parquet'
    TESTFILE = 'cf_train_subval_top10_1004_processed.parquet' # replace with testfile past prototyping stage
  
    def __init__(self):
        self.spark = SparkSession.builder.appName('ALSTrainer').getOrCreate()

        '''
        Follow example in docs:
        https://spark.apache.org/docs/2.4.7/ml-collaborative-filtering.html
        '''
    def train(self):
        train = self.spark.read.parquet(f'{self.DATADIR}/{self.TRAINFILE}')
        test = self.spark.read.parquet(f'{self.DATADIR}/{self.TESTFILE}')
    
        # Build the recommendation model using ALS on the training data
        # Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
        als = ALS(maxIter=5, \
                  regParam=0.01, \
                  userCol="user_id_index", \
                  itemCol="track_id_index", \
                  ratingCol="count", \
                  coldStartStrategy="drop")

        model = als.fit(train)

        # Evaluate the model by computing the RMSE on the test data
        predictions = model.transform(test)
        evaluator = RegressionEvaluator(metricName="rmse", labelCol="count", predictionCol="prediction")
        rmse = evaluator.evaluate(predictions)
        print("Root-mean-square error = " + str(rmse))

        # Generate top 10 track recommendations for each user
        userRecs = model.recommendForAllUsers(10)
        # Generate top 10 user recommendations for each track
        songRecs = model.recommendForAllItems(10)


if __name__ == "__main__":
    ALSTrainer().train()


