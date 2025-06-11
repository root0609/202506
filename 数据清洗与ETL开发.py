# ===================== 数据准备与建模 =====================
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# 1. Spark 初始化（关联 Hive 数仓）
spark = SparkSession.builder \
    .appName("Adidas Analysis") \
    .config("spark.sql.warehouse.dir", "/user/hive/warehouse") \
    .config("spark.sql.catalogImplementation", "hive") \
    .config("hive.metastore.uris", "thrift://localhost:9083") \
    .enableHiveSupport() \
    .getOrCreate()

# 2. 数据读取与清洗
file_path = "/data/阿迪达斯订单数据_前 10000 条.csv"  
df = spark.read.csv(file_path, header=True, inferSchema=True)
clean_df = df.dropDuplicates().fillna({"年龄": 30}) \
    .filter((col("单价") > 0) & (col("购买数量").between(1, 10)))

# 3. 数据分层（Hive 表/Parquet 备份）
try:
    clean_df.write.mode("overwrite").saveAsTable("dwd_orders")
except Exception as e:
    clean_df.createOrReplaceTempView("dwd_orders")
    clean_df.write.mode("overwrite").parquet("/data/dwd_orders.parquet")

# 4. 特征工程与模型训练
assembler = VectorAssembler(inputCols=["单价", "购买数量", "年龄"], outputCol="features")
data = assembler.transform(clean_df)
train, test = data.randomSplit([0.8, 0.2], seed=42)
lr = LinearRegression(featuresCol="features", labelCol="订单总额")
model = lr.fit(train)
predictions = model.transform(test)

# 5. 模型评估
evaluator = RegressionEvaluator(labelCol="订单总额", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

# 6. 关键指标计算
total_sales = clean_df.agg({"订单总额": "sum"}).first()[0]
avg_order_value = clean_df.agg({"订单总额": "avg"}).first()[0]
member_ratio = clean_df.filter(col("是否会员") == True).count() / clean_df.count() * 100
