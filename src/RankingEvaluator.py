import sys
from abc import abstractmethod, ABCMeta

from pyspark import since, keyword_only
from pyspark.ml.wrapper import JavaParams
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasLabelCol, HasPredictionCol, HasProbabilityCol, \
    HasRawPredictionCol, HasFeaturesCol, HasWeightCol
from pyspark.ml.common import inherit_doc
from pyspark.ml.util import JavaMLReadable, JavaMLWritable


class RankingEvaluator(JavaEvaluator, HasLabelCol, HasPredictionCol,
                       JavaMLReadable, JavaMLWritable):

    metricName = Param(Params._dummy(), "metricName",
                       "metric name in evaluation "
                       "(meanAveragePrecision|meanAveragePrecisionAtK|"
                       "precisionAtK|ndcgAtK|recallAtK)",
                       typeConverter=TypeConverters.toString)
    k = Param(Params._dummy(), "k",
              "The ranking position value used in meanAveragePrecisionAtK|precisionAtK|"
              "ndcgAtK|recallAtK. Must be > 0. The default value is 10.",
              typeConverter=TypeConverters.toInt)

    def __init__(self, *, predictionCol="prediction", labelCol="label",
                 metricName="meanAveragePrecision", k=10):
        """
        __init__(self, \\*, predictionCol="prediction", labelCol="label", \
                 metricName="meanAveragePrecision", k=10)
        """
        super(RankingEvaluator, self).__init__()
        self._java_obj = self._new_java_obj(
            "org.apache.spark.ml.evaluation.RankingEvaluator", self.uid)
        self._setDefault(metricName="meanAveragePrecision", k=10)
        kwargs = self._input_kwargs
        self._set(**kwargs)

    def setMetricName(self, value):
        """
        Sets the value of :py:attr:`metricName`.
        """
        return self._set(metricName=value)

    def getMetricName(self):
        """
        Gets the value of metricName or its default value.
        """
        return self.getOrDefault(self.metricName)

    def setK(self, value):
        """
        Sets the value of :py:attr:`k`.
        """
        return self._set(k=value)

    def getK(self):
        """
        Gets the value of k or its default value.
        """
        return self.getOrDefault(self.k)

    def setLabelCol(self, value):
        """
        Sets the value of :py:attr:`labelCol`.
        """
        return self._set(labelCol=value)

    def setPredictionCol(self, value):
        """
        Sets the value of :py:attr:`predictionCol`.
        """
        return self._set(predictionCol=value)

    def setParams(self, *, predictionCol="prediction", labelCol="label",
                  metricName="meanAveragePrecision", k=10):
        """
        setParams(self, \\*, predictionCol="prediction", labelCol="label", \
                  metricName="meanAveragePrecision", k=10)
        Sets params for ranking evaluator.
        """
        kwargs = self._input_kwargs
        return self._set(**kwargs)

if __name__ == "__main__":
    import doctest
    import tempfile
    import pyspark.ml.evaluation
    from pyspark.sql import SparkSession
    globs = pyspark.ml.evaluation.__dict__.copy()
    # The small batch size here ensures that we see multiple batches,
    # even in these small test examples:
    spark = SparkSession.builder\
        .master("local[2]")\
        .appName("ml.evaluation tests")\
        .getOrCreate()
    globs['spark'] = spark
    temp_path = tempfile.mkdtemp()
    globs['temp_path'] = temp_path
    try:
        (failure_count, test_count) = doctest.testmod(globs=globs, optionflags=doctest.ELLIPSIS)
        spark.stop()
    finally:
        from shutil import rmtree
        try:
            rmtree(temp_path)
        except OSError:
            pass
    if failure_count:
        sys.exit(-1)
