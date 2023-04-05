print("###############################  Spark-APP ###########################")

from pyspark.sql import SparkSession

spark = SparkSession.builder \
        .master("local") \
        .appName("SparkSession Name") \
        .config("spark.executor.memory", '8g') \
        .config("spark.executor.core", 2) \
        .getOrCreate()

print("######### Spark: ", spark.version)
print("######### Hadoop: ", spark.sparkContext._gateway.jvm.org.apache.hadoop.util.VersionInfo.getVersion())


data = [1,2,3,4,5,6]
distData = spark.sparkContext.parallelize(data,2)
print(type(data))
print(type(distData))

print("####### ReadFile")
lines = spark.sparkContext.textFile('/user/hmoradi/Lec24_Book.txt')
ll = lines.map(lambda s: len(s))
tl = ll.reduce(lambda a, b:a+b)

print("# totalLenght:", tl)

# ll.saveAsTextFile("/user/hmoradi/results_book/")

def myFunc(l):
    words = l.split(" ")
    return(len(words))
wordLine = lines.map(myFunc)
print("# ",wordLine.take(5))
print("# words: ", wordLine.reduce(lambda a,b: a+b))

count = lines.flatMap(lambda line: line.split(" "))\
        .map(lambda word: (word,1))\
        .reduceByKey(lambda a,b: a+b).sortBy(lambda a: a[1],0)\
        .collect()
print("# count type: ", type(count))
print("# count: ", count[0:5])

rangeVar = spark.range(1000).toDF("number")
print("# print type: ", type(rangeVar))
print("# print types: ", rangeVar.dtypes)
rangeVar.show(5)

print("# ",rangeVar.schema)
rangeVar.printSchema()

even = rangeVar.where(" number % 2 = 0")
print("# count of even: ", even.count())


fd = spark.read.format("csv")\
        .option("inferSchema","true")\
        .option("header","true")\
        .load("/user/hmoradi/Lec24_Flight.csv")
fd.explain()
fd.printSchema()


fd.createOrReplaceTempView("fd")
qUsingSql = spark.sql(""" 
         select DEST_COUNTRY_NAME, count(*) as cnt
         from fd
         group by DEST_COUNTRY_NAME
         sort by cnt desc
        """)
print("# sql Query Answ:")
qUsingSql.show(5)

qUsingSql = fd.filter(fd.DEST_COUNTRY_NAME == "United States")  
print("# Function Call: ")
qUsingSql.show(5)

from pyspark.sql import functions as F
null = fd.filter(F.col("DEST_COUNTRY_NAME").isNull())
print("# isNull: ")
null.show(5)

qUsingDot = fd.groupby("DEST_COUNTRY_NAME")\
        .sum("count")\
        .withColumnRenamed("sum(count)","dest_count")\
        .sort(F.desc("dest_count"))\
        .limit(5)
qUsingDot.show()



mixQuery = spark.sql("""
        Select DEST_COUNTRY_NAME, sum(count)
        from fd 
        group by DEST_COUNTRY_NAME
        """)\
                .where("DEST_COUNTRY_NAME like 'So%' or DEST_COUNTRY_NAME like 'Un%' ")\
                .sort(F.asc("DEST_COUNTRY_NAME"))
mixQuery.show(5)


print("### ML in Spark")
# climatological variables (including visibility, temperature, wind speed and direction, humidity, dew point, and pressure)
# https://developer.ibm.com/exchanges/data/all/jfk-weather-data/
df = spark.read.option("header", "true").option("inferSchema","true").csv('file:///home/hmoradi/spark/noaa-weather-data-jfk-airport/jfk_weather_cleaned.csv')
df.createOrReplaceTempView('df')
df.printSchema() # or df.schema
df.show(5)



#exit()

from pyspark.ml.feature import VectorAssembler
vectorAssembler = VectorAssembler(inputCols=["HOURLYWindDirectionSin","HOURLYWindDirectionCos","HOURLYStationPressure"],outputCol="features")
df_transformed = vectorAssembler.transform(df)
df_transformed.show(5)



#exit()

# https://spark.apache.org/docs/latest/ml-features#normalizer
from pyspark.ml.feature import Normalizer
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)



#exit()

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(labelCol="HOURLYWindSpeed", featuresCol='features_norm', maxIter=100, regParam=0.0, elasticNetParam=0.0)



#exit()

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, normalizer,lr])



#exit()

splits = df.randomSplit([0.8, 0.2])
df_train = splits[0]
df_test = splits[1]



#exit()

model = pipeline.fit(df_train)
prediction = model.transform(df_test)



#exit() 

from pyspark.ml.evaluation import RegressionEvaluator
evaluator = RegressionEvaluator(labelCol="HOURLYWindSpeed", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print("RMSE on test data = %g" % rmse)

