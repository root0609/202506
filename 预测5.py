from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, PCA, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col as spark_col, expr, percentile_approx
import pandas as pd
import numpy as np

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

# 异常值处理 - 基于IQR方法过滤订单总额异常值
quantiles = spark_df.approxQuantile(["订单总额"], [0.25, 0.75], 0.05)
Q1 = quantiles[0][0]
Q3 = quantiles[0][1]
IQR = Q3 - Q1
spark_df = spark_df.filter(
    (spark_col("订单总额") >= Q1 - 1.5 * IQR) &
    (spark_col("订单总额") <= Q3 + 1.5 * IQR)
)

# 特征工程 - 添加交互特征
spark_df = spark_df.withColumn("单价_年龄交互", spark_col("单价") * spark_col("年龄"))
spark_df = spark_df.withColumn("单价_数量交互", spark_col("单价") * spark_col("购买数量"))

# 对商品子类进行编码
indexer = StringIndexer(inputCol="商品子类", outputCol="商品子类_index")
encoder = OneHotEncoder(inputCol="商品子类_index", outputCol="商品子类_encoded")

# 选择特征列和目标列
feature_cols = ["单价", "购买数量", "年龄", "商品子类_encoded",
                "单价_年龄交互", "单价_数量交互"]
target_col = "订单总额"

# 准备特征向量
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# 特征缩放
scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                       withStd=True, withMean=True)

# 使用PCA进行降维（增加主成分数量）
pca = PCA(k=3, inputCol="scaled_features", outputCol="pca_features")

# 创建随机森林回归模型
rf = RandomForestRegressor(
    featuresCol="pca_features",
    labelCol=target_col,
    numTrees=150,          # 增加树的数量
    maxDepth=10,           # 增加树的深度
    minInstancesPerNode=2,
    featureSubsetStrategy="auto",
    seed=42
)

# 创建Pipeline
pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler, pca, rf])

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
evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction")

# 计算多种评估指标
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print(f"均方根误差 (RMSE): {rmse:.2f}")
print(f"平均绝对误差 (MAE): {mae:.2f}")
print(f"决定系数 (R²): {r2:.4f}")

# 分析特征重要性
rf_model = model.stages[-1]  # 获取随机森林模型
feature_importances = rf_model.featureImportances
print("\n特征重要性:")
print(feature_importances)

# 停止 SparkSession
spark.stop()