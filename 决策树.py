from pyspark.sql import SparkSession
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    QuantileDiscretizer, ChiSqSelector
)
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from pyspark.sql.functions import (
    count, col, lit, when, avg,
)
import seaborn as sns
import warnings

# ======================== 解决字体警告与中文显示 ========================
# 过滤 tkinter 字体缺失警告
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="tkinter",
    message="Glyph .* missing from current font."
)

# 强制设置中文字体（优先用系统已安装字体，如无则 fallback 到支持中文的字体）
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示异常

# 尝试加载系统中文字体（可选，若已知字体路径可直接指定）
try:
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # 示例路径，根据实际调整
    font = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font.get_name()
    print(f"已设置中文字体: {font.get_name()}")
except:
    print("警告: 自定义字体加载失败，使用默认回退字体（仍支持中文显示）")

# ======================== SparkSession 初始化 ========================
spark = SparkSession.builder \
    .appName("CustomerPreferenceAnalysis") \
    .config("spark.broadcast.blockSize", "16m") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# ======================== 数据读取与清洗 ========================
# 读取数据（路径、编码按需调整）
df = spark.read.csv(
    "/home/hadoop/666.csv",
    header=True,
    inferSchema=True,
    encoding="gb18030"
)

# 缺失值处理
print(f"原始数据行数: {df.count()}")
df = df.na.drop(subset=["性别", "年龄", "是否会员", "商品大类", "收货省份", "单价"])
print(f"处理后数据行数: {df.count()}")

# 优化数据类型
df = df.withColumn("单价", col("单价").cast("float"))
df = df.withColumn("年龄", col("年龄").cast("integer"))

# ======================== 基础统计分析 ========================
# 平均年龄、平均单价
age_stats = df.select("年龄").describe().toPandas()
price_stats = df.select("单价").describe().toPandas()
mean_age = float(age_stats[age_stats['summary'] == 'mean']['年龄'].values[0])
mean_price = float(price_stats[price_stats['summary'] == 'mean']['单价'].values[0])

# 性别分布
gender_distribution = df.groupBy("性别").count().toPandas()

# 商品大类分布
category_distribution = df.groupBy("商品大类").count().toPandas()

print("\n===== 基础统计结果 =====")
print(f"- 平均年龄: {mean_age:.2f} 岁")
print(f"- 平均单价: {mean_price:.2f} 元")
print(f"- 性别分布: {gender_distribution.to_string(index=False)}")
print(f"- 商品大类分布: {category_distribution.to_string(index=False)}")

# ======================== 特征工程与模型训练 ========================
# 标签编码（商品大类 -> label）
label_indexer = StringIndexer(inputCol="商品大类", outputCol="label")

# 分类特征处理（性别、是否会员、收货省份）
categorical_cols = ["性别", "是否会员", "收货省份"]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_index") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vec") for c in categorical_cols]

# 年龄分箱（4段）
age_discretizer = QuantileDiscretizer(
    numBuckets=4,
    inputCol="年龄",
    outputCol="年龄分段_num"
)

# 单价分箱（5段）
price_discretizer = QuantileDiscretizer(
    numBuckets=5,
    inputCol="单价",
    outputCol="单价分段_num"
)

# 特征组装
assembler = VectorAssembler(
    inputCols=["性别_vec", "是否会员_vec", "收货省份_vec", "年龄分段_num", "单价分段_num"],
    outputCol="features"
)

# 特征选择（保留 top10 特征）
selector = ChiSqSelector(
    featuresCol="features",
    outputCol="selectedFeatures",
    labelCol="label",
    numTopFeatures=10
)

# 模型定义（随机森林、逻辑回归）
classifiers = {
    "随机森林": RandomForestClassifier(
        labelCol="label",
        featuresCol="selectedFeatures",
        numTrees=100,
        maxDepth=8,
        seed=42
    ),
    "逻辑回归": LogisticRegression(
        labelCol="label",
        featuresCol="selectedFeatures",
        maxIter=100,
        regParam=0.1
    )
}

# 划分数据集
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# 模型训练与评估
results = {}
for name, classifier in classifiers.items():
    print(f"\n===== 训练 {name} 模型 =====")

    # 构建 Pipeline
    pipeline = Pipeline(stages=[
        label_indexer,
        *indexers,
        age_discretizer,
        price_discretizer,
        *encoders,
        assembler,
        selector,
        classifier
    ])

    # 交叉验证（仅随机森林）
    if name == "随机森林":
        paramGrid = ParamGridBuilder() \
            .addGrid(classifier.numTrees, [100]) \
            .addGrid(classifier.maxDepth, [8]) \
            .build()

        crossval = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=MulticlassClassificationEvaluator(metricName="accuracy"),
            numFolds=3
        )
        model = crossval.fit(train_df).bestModel
        print(f"最佳参数: numTrees={model.stages[-1].getNumTrees}, maxDepth={model.stages[-1].getMaxDepth()}")
    else:
        model = pipeline.fit(train_df)

    # 模型评估
    predictions = model.transform(test_df)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    accuracy = evaluator.evaluate(predictions)
    results[name] = {
        "model": model,
        "accuracy": accuracy
    }
    print(f"{name} 准确率: {accuracy:.4f}")

# ======================== 模型对比与最佳模型选择 ========================
print("\n===== 模型性能对比 =====")
for name, res in results.items():
    print(f"{name}: {res['accuracy']:.4f}")

best_model_name = max(results, key=lambda k: results[k]["accuracy"])
best_model = results[best_model_name]["model"]
print(f"\n最佳模型: {best_model_name}")

# ======================== 特征重要性可视化（随机森林） ========================
if best_model_name == "随机森林":
    feature_importance = best_model.stages[-1].featureImportances.toArray()
    selector_model = best_model.stages[-2]
    selected_features = selector_model.selectedFeatures

    # 简化特征名称映射（按业务逻辑调整）
    feature_names = []
    for idx in selected_features:
        if idx < 2:
            feature_names.append(categorical_cols[idx])
        elif 2 <= idx < (2 + df.select("收货省份").distinct().count()):
            feature_names.append(f"收货省份_{idx - 2}")
        else:
            feature_names.append("年龄分段" if idx == (2 + df.select("收货省份").distinct().count()) else "单价分段")

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.barh(feature_names, feature_importance, color="#66b3ff")
    plt.xlabel("重要性", fontsize=12)
    plt.title(f"{best_model_name} 特征重要性", fontsize=14)
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("特征重要性图已保存: feature_importance.png")

# ======================== 业务可视化分析（保留核心图表） ========================
# 1. 商品大类分布
plt.figure(figsize=(10, 6))
sns.barplot(
    data=category_distribution,
    x="count",
    y="商品大类",
    hue="商品大类",
    palette="viridis",
    legend=False
)
plt.title("商品大类分布", fontsize=14)
plt.xlabel("数量", fontsize=12)
plt.ylabel("商品大类", fontsize=12)
plt.tight_layout()
plt.savefig("category_distribution.png")
print("商品大类分布图已保存: category_distribution.png")

# 2. 年龄分布直方图
plt.figure(figsize=(10, 6))
ages = df.select("年龄").rdd.flatMap(lambda x: x).collect()
sns.histplot(ages, bins=20, kde=True, color="#ff9999")
plt.title("用户年龄分布", fontsize=14)
plt.xlabel("年龄", fontsize=12)
plt.ylabel("频次", fontsize=12)
plt.tight_layout()
plt.savefig("age_distribution.png")
print("年龄分布图已保存: age_distribution.png")

# 3. 性别与商品类别关系（如需保留可调整，此处演示删除 gender_distribution 后逻辑）
gender_category = df.groupBy("性别").pivot("商品大类").count().toPandas()
plt.figure(figsize=(12, 6))
gender_category.set_index("性别").plot(kind="bar", ax=plt.gca(), cmap="tab20")
plt.title("不同性别对商品类别的偏好", fontsize=14)
plt.xlabel("性别", fontsize=12)
plt.ylabel("购买次数", fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.legend(title="商品大类", prop={"size": 10})
plt.tight_layout()
plt.savefig("gender_category.png")
print("性别与商品类别关系图已保存: gender_category.png")

# 4. 年龄分段与商品类别关系
age_buckets = [0, 25, 35, 45, 100]
age_labels = ["0-25岁", "26-35岁", "36-45岁", "46岁以上"]
df_with_age_bin = df.withColumn(
    "年龄分段",
    when(col("年龄") < 25, age_labels[0])
    .when(col("年龄") < 35, age_labels[1])
    .when(col("年龄") < 45, age_labels[2])
    .otherwise(age_labels[3])
)

age_category = df_with_age_bin.groupBy("年龄分段").pivot("商品大类").count().toPandas()
plt.figure(figsize=(12, 6))
age_category.set_index("年龄分段").plot(kind="bar", ax=plt.gca(), cmap="tab20")
plt.title("不同年龄段对商品类别的偏好", fontsize=14)
plt.xlabel("年龄分段", fontsize=12)
plt.ylabel("购买次数", fontsize=12)
plt.xticks(rotation=0, fontsize=10)
plt.legend(title="商品大类", prop={"size": 10})
plt.tight_layout()
plt.savefig("age_category.png")
print("年龄与商品类别关系图已保存: age_category.png")

# 5. 会员与非会员消费差异
member_stats = df.groupBy("是否会员").agg(
    avg("单价").alias("平均单价"),
    avg("年龄").alias("平均年龄")
).toPandas()

plt.figure(figsize=(14, 6))
# 平均消费金额
ax1 = plt.subplot(1, 2, 1)
sns.barplot(
    data=member_stats,
    x="是否会员",
    y="平均单价",
    hue="是否会员",
    palette="pastel",
    legend=False,
    ax=ax1
)
plt.title("会员与非会员的平均消费金额", fontsize=12)
plt.xlabel("会员状态", fontsize=10)
plt.ylabel("平均单价 (元)", fontsize=10)

# 平均年龄
ax2 = plt.subplot(1, 2, 2)
sns.barplot(
    data=member_stats,
    x="是否会员",
    y="平均年龄",
    hue="是否会员",
    palette="pastel",
    legend=False,
    ax=ax2
)
plt.title("会员与非会员的平均年龄", fontsize=12)
plt.xlabel("会员状态", fontsize=10)
plt.ylabel("平均年龄 (岁)", fontsize=10)

plt.tight_layout()
plt.savefig("member_stats.png")
print("会员与非会员消费差异图已保存: member_stats.png")

# 6. 地域消费差异（前10省份）
province_stats = df.groupBy("收货省份").count() \
    .orderBy("count", ascending=False).limit(10).toPandas()

plt.figure(figsize=(12, 6))
sns.barplot(
    data=province_stats,
    x="count",
    y="收货省份",
    hue="收货省份",
    palette="mako",
    legend=False
)
plt.title("各省份消费分布（前10）", fontsize=14)
plt.xlabel("消费次数", fontsize=12)
plt.ylabel("省份", fontsize=12)
plt.tight_layout()
plt.savefig("province_distribution.png")
print("地域消费差异图已保存: province_distribution.png")

# ======================== 生成分析报告（简化版） ========================
print("\n\n===== 消费者偏好分析报告（简化版） =====")
print(f"1. 基础数据")
print(f"- 数据总量: {df.count()} 条")
print(f"- 平均年龄: {mean_age:.2f} 岁 | 平均单价: {mean_price:.2f} 元")
print(
    f"- 女性占比: {gender_distribution[gender_distribution['性别'] == '女']['count'].values[0] / df.count() * 100:.2f}%")
print(f"- 会员占比: {df.filter(col('是否会员') == '是').count() / df.count() * 100:.2f}%")

print(f"\n2. 商品销售")
top_category = category_distribution.sort_values("count", ascending=False).iloc[0]
bottom_category = category_distribution.sort_values("count", ascending=True).iloc[0]
print(
    f"- 最畅销: {top_category['商品大类']} ({top_category['count']} 销量, {top_category['count'] / df.count() * 100:.2f}%)")
print(
    f"- 最冷门: {bottom_category['商品大类']} ({bottom_category['count']} 销量, {bottom_category['count'] / df.count() * 100:.2f}%)")

print(f"\n3. 特征重要性")
if best_model_name == "随机森林":
    print(
        f"- 年龄: {feature_importance[feature_names.index('年龄分段')]:.4f} | 性别: {feature_importance[feature_names.index('性别')]:.4f}")

print(f"\n4. 建议")
print(f"- 针对 {age_labels[age_category.set_index('年龄分段').sum(axis=1).argmax()]} 推广 {top_category['商品大类']}")
print(f"- 优化会员权益（当前会员平均消费与非会员差异需结合业务校验）")

# ======================== 关闭 SparkSession ========================
spark.stop()