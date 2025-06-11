from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, PCA
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import pandas as pd
import numpy as np
from pyspark.sql.functions import col as spark_col, concat, lit

# 创建 SparkSession
spark = SparkSession.builder \
   .appName("InventoryDemandPrediction") \
   .getOrCreate()

# 加载数据集
df = pd.read_csv('已清洗.csv')

# 手动转换 pandas DataFrame 为 Spark DataFrame
data_rows = []
for index in range(len(df)):
    row = []
    for col in df.columns:
        value = df.iloc[index][col]
        if isinstance(value, (np.int64, np.float64)):
            row.append(value.item())
        else:
            row.append(value)
    data_rows.append(row)
spark_df = spark.createDataFrame(data_rows, df.columns.tolist())

# 对商品子类进行编码
indexer = StringIndexer(inputCol="商品子类", outputCol="商品子类_index")
encoder = OneHotEncoder(inputCol="商品子类_index", outputCol="商品子类_encoded")

# 选择特征列和目标列
feature_cols = ["单价", "购买数量", "年龄", "商品子类_encoded"]
target_col = "订单总额"

# 准备特征向量
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 使用 PCA 进行降维
pca = PCA(k=2, inputCol="features", outputCol="pca_features")

# 创建随机森林回归模型，使用降维后的特征
rf = RandomForestRegressor(featuresCol="pca_features", labelCol=target_col)

# 创建 Pipeline
pipeline = Pipeline(stages=[indexer, encoder, assembler, pca, rf])

# 划分训练集和测试集
train_data, test_data = spark_df.randomSplit([0.8, 0.2], seed=42)

# 训练模型
model = pipeline.fit(train_data)

# 在测试集上进行预测
predictions = model.transform(test_data)

# 显示真实值与预测值的对比
print("真实值与预测值对比:")
predictions.select("商品子类", "单价", "购买数量", "年龄", target_col, "prediction") \
           .withColumn("误差", spark_col("prediction") - spark_col(target_col)) \
           .withColumn("误差百分比", (spark_col("prediction") - spark_col(target_col)) / spark_col(target_col) * 100) \
           .show(20)

# 评估模型
evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)

print(f"均方根误差 (RMSE): {rmse:.2f}")

# 计算 R² 评估指标
r2_evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2")
r2 = r2_evaluator.evaluate(predictions)
print(f"决定系数 (R²): {r2:.4f}")

# 假设库存需求是购买数量的 1.2 倍（可根据实际调整）
inventory_factor = 1.2

# 计算每个预测结果对应的库存需求
inventory_predictions = predictions.withColumn(
    "库存需求预测", spark_col("购买数量") * inventory_factor
)

# 按商品子类分组并汇总库存需求预测
grouped_inventory = inventory_predictions.groupBy("商品子类") \
                                        .agg({"库存需求预测": "sum", target_col: "sum"}) \
                                        .withColumnRenamed("sum(库存需求预测)", "总库存需求预测") \
                                        .withColumnRenamed(f"sum({target_col})", "总订单金额")

# 显示按商品子类分组的库存需求预测结果
print("\n按商品子类分组的库存需求预测:")
grouped_inventory.show(truncate=False)

# 停止 SparkSession
spark.stop()
