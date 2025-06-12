from pyspark.sql import SparkSession
from pyspark.ml.feature import (StringIndexer, OneHotEncoder, VectorAssembler, QuantileDiscretizer, ChiSqSelector)
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from pyspark.sql.functions import (count, col, lit, when, avg,)
import seaborn as sns
import warnings

# ======================== 解决字体警告与中文显示 ========================
warnings.filterwarnings("ignore", category=UserWarning, module="tkinter", message="Glyph .* missing from current font.")  # 过滤tkinter字体缺失警告
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "SimSun", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
try:
    font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"  # 指定中文字体路径
    font = fm.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = font.get_name()
    print(f"已设置中文字体: {font.get_name()}")
except:  # 字体加载失败时
    print("警告: 自定义字体加载失败，使用默认回退字体（仍支持中文显示）")

# ======================== SparkSession 初始化 ========================
spark = SparkSession.builder.appName("CustomerPreferenceAnalysis").config("spark.broadcast.blockSize", "16m").config("spark.serializer", "org.apache.spark.serializer.KryoSerializer").config("spark.executor.memory", "4g").config("spark.driver.memory", "2g").getOrCreate()  # 创建SparkSession并配置内存参数

# ======================== 数据读取与清洗 ========================
df = spark.read.csv("/home/hadoop/666.csv", header=True, inferSchema=True, encoding="gb18030")  # 读取CSV文件，自动推断数据类型，设置编码
print(f"原始数据行数: {df.count()}")
df = df.na.drop(subset=["性别", "年龄", "是否会员", "商品大类", "收货省份", "单价"])  # 删除指定列包含缺失值的记录
print(f"处理后数据行数: {df.count()}")
df = df.withColumn("单价", col("单价").cast("float")).withColumn("年龄", col("年龄").cast("integer"))

# ======================== 基础统计分析 ========================
age_stats, price_stats = df.select("年龄").describe().toPandas(), df.select("单价").describe().toPandas()  # 计算年龄和价格的统计指标并转换为Pandas DataFrame
mean_age, mean_price = float(age_stats[age_stats['summary'] == 'mean']['年龄'].values[0]), float(price_stats[price_stats['summary'] == 'mean']['单价'].values[0])  # 提取平均值
gender_distribution, category_distribution = df.groupBy("性别").count().toPandas(), df.groupBy("商品大类").count().toPandas()  # 计算性别和商品大类分布
print("\n===== 基础统计结果 =====")  # 打印统计结果标题
print(f"- 平均年龄: {mean_age:.2f} 岁")
print(f"- 平均单价: {mean_price:.2f} 元")
print(f"- 性别分布: {gender_distribution.to_string(index=False)}")
print(f"- 商品大类分布: {category_distribution.to_string(index=False)}")

# ======================== 特征工程与模型训练 ========================
label_indexer = StringIndexer(inputCol="商品大类", outputCol="label")  # 因变量
categorical_cols = ["性别", "是否会员", "收货省份"]  # 自变量
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_index") for c in categorical_cols]  # 创建分类特征索引器列表
encoders = [OneHotEncoder(inputCol=f"{c}_index", outputCol=f"{c}_vec") for c in categorical_cols]  # 创建独热编码器列表
###########################自变量分箱############################################################
age_discretizer = QuantileDiscretizer(numBuckets=4, inputCol="年龄", outputCol="年龄分段_num")  # 创建年龄分箱器，分为4个桶
price_discretizer = QuantileDiscretizer(numBuckets=5, inputCol="单价", outputCol="单价分段_num")  # 创建价格分箱器，分为5个桶
####合并筛选
assembler = VectorAssembler(inputCols=["性别_vec", "是否会员_vec", "收货省份_vec", "年龄分段_num", "单价分段_num"], outputCol="features")  # 创建特征组装器，将所有特征合并为向量
selector = ChiSqSelector(featuresCol="features", outputCol="selectedFeatures", labelCol="label", numTopFeatures=10)  # 创建卡方特征选择器，选择前10个特征
classifiers = {"随机森林": RandomForestClassifier(labelCol="label", featuresCol="selectedFeatures", numTrees=100, maxDepth=8, seed=42), "逻辑回归": LogisticRegression(labelCol="label", featuresCol="selectedFeatures", maxIter=100, regParam=0.1)}  # 定义随机森林和逻辑回归分类器
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)  # 划分训练集和测试集
results = {}  # 初始化结果字典
for name, classifier in classifiers.items():  # 遍历分类器
    print(f"\n===== 训练 {name} 模型 =====")  # 打印训练模型信息
    pipeline = Pipeline(stages=[label_indexer, *indexers, age_discretizer, price_discretizer, *encoders, assembler, selector, classifier])  # 创建Pipeline
    if name == "随机森林":  # 如果是随机森林模型
        paramGrid = ParamGridBuilder().addGrid(classifier.numTrees, [100]).addGrid(classifier.maxDepth, [8]).build()  # 创建参数网格
        crossval = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=MulticlassClassificationEvaluator(metricName="accuracy"), numFolds=3)  # 创建交叉验证器
        model = crossval.fit(train_df).bestModel  # 训练模型并获取最佳模型
        print(f"最佳参数: numTrees={model.stages[-1].getNumTrees}, maxDepth={model.stages[-1].getMaxDepth()}")  # 打印最佳参数
    else:  # 如果是逻辑回归模型
        model = pipeline.fit(train_df)  # 直接训练模型
    predictions = model.transform(test_df)  # 在测试集上进行预测
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")  # 创建评估器
    accuracy = evaluator.evaluate(predictions)  # 计算准确率
    results[name] = {"model": model, "accuracy": accuracy}  # 保存模型和准确率
    print(f"{name} 准确率: {accuracy:.4f}")  # 打印模型准确率

# ======================== 模型对比与最佳模型选择 ========================
print("\n===== 模型性能对比 =====")  # 打印模型性能对比标题
for name, res in results.items():  # 遍历结果字典
    print(f"{name}: {res['accuracy']:.4f}")  # 打印各模型准确率
best_model_name = max(results, key=lambda k: results[k]["accuracy"])  # 找出准确率最高的模型
best_model = results[best_model_name]["model"]  # 获取最佳模型
print(f"\n最佳模型: {best_model_name}")  # 打印最佳模型名称

# ======================== 特征重要性可视化（随机森林） ========================
if best_model_name == "随机森林":  # 如果最佳模型是随机森林
    feature_importance = best_model.stages[-1].featureImportances.toArray()  # 获取特征重要性
    selector_model = best_model.stages[-2]  # 获取特征选择器模型
    selected_features = selector_model.selectedFeatures  # 获取选中的特征索引
    feature_names = []  # 初始化特征名称列表
    for idx in selected_features:  # 遍历选中的特征索引
        if idx < 2:  # 如果是前两个特征
            feature_names.append(categorical_cols[idx])  # 添加分类特征名称
        elif 2 <= idx < (2 + df.select("收货省份").distinct().count()):  # 如果是收货省份特征
            feature_names.append(f"收货省份_{idx - 2}")  # 添加收货省份特征名称
        else:  # 如果是年龄或价格特征
            feature_names.append("年龄分段" if idx == (2 + df.select("收货省份").distinct().count()) else "单价分段")  # 添加年龄或价格特征名称
    plt.figure(figsize=(12, 6))  # 创建图形
    plt.barh(feature_names, feature_importance, color="#66b3ff")  # 绘制水平条形图
    plt.xlabel("重要性", fontsize=12)  # 设置x轴标签
    plt.title(f"{best_model_name} 特征重要性", fontsize=14)  # 设置标题
    plt.tight_layout()  # 调整布局
    plt.savefig("feature_importance.png")  # 保存图形
    print("特征重要性图已保存: feature_importance.png")  # 打印保存信息

# ======================== 业务可视化分析（保留核心图表） ========================
plt.figure(figsize=(10, 6))  # 创建图形
sns.barplot(data=category_distribution, x="count", y="商品大类", hue="商品大类", palette="viridis", legend=False)  # 绘制商品大类分布条形图
plt.title("商品大类分布", fontsize=14)  # 设置标题
plt.xlabel("数量", fontsize=12)  # 设置x轴标签
plt.ylabel("商品大类", fontsize=12)  # 设置y轴标签
plt.tight_layout()  # 调整布局
plt.savefig("category_distribution.png")  # 保存图形
print("商品大类分布图已保存: category_distribution.png")  # 打印保存信息

plt.figure(figsize=(10, 6))  # 创建图形
ages = df.select("年龄").rdd.flatMap(lambda x: x).collect()  # 收集年龄数据
sns.histplot(ages, bins=20, kde=True, color="#ff9999")  # 绘制年龄分布直方图
plt.title("用户年龄分布", fontsize=14)  # 设置标题
plt.xlabel("年龄", fontsize=12)  # 设置x轴标签
plt.ylabel("频次", fontsize=12)  # 设置y轴标签
plt.tight_layout()  # 调整布局
plt.savefig("age_distribution.png")  # 保存图形
print("年龄分布图已保存: age_distribution.png")  # 打印保存信息

gender_category = df.groupBy("性别").pivot("商品大类").count().toPandas()  # 计算性别与商品大类交叉表
plt.figure(figsize=(12, 6))  # 创建图形
gender_category.set_index("性别").plot(kind="bar", ax=plt.gca(), cmap="tab20")  # 绘制柱状图
plt.title("不同性别对商品类别的偏好", fontsize=14)  # 设置标题
plt.xlabel("性别", fontsize=12)  # 设置x轴标签
plt.ylabel("购买次数", fontsize=12)  # 设置y轴标签
plt.xticks(rotation=0, fontsize=10)  # 设置x轴刻度
plt.legend(title="商品大类", prop={"size": 10})  # 设置图例
plt.tight_layout()  # 调整布局
plt.savefig("gender_category.png")  # 保存图形
print("性别与商品类别关系图已保存: gender_category.png")  # 打印保存信息

age_buckets, age_labels = [0, 25, 35, 45, 100], ["0-25岁", "26-35岁", "36-45岁", "46岁以上"]  # 定义年龄分箱边界和标签
df_with_age_bin = df.withColumn("年龄分段", when(col("年龄") < 25, age_labels[0]).when(col("年龄") < 35, age_labels[1]).when(col("年龄") < 45, age_labels[2]).otherwise(age_labels[3]))  # 创建年龄分段列
age_category = df_with_age_bin.groupBy("年龄分段").pivot("商品大类").count().toPandas()  # 计算年龄分段与商品大类交叉表
plt.figure(figsize=(12, 6))  # 创建图形
age_category.set_index("年龄分段").plot(kind="bar", ax=plt.gca(), cmap="tab20")  # 绘制柱状图
plt.title("不同年龄段对商品类别的偏好", fontsize=14)  # 设置标题
plt.xlabel("年龄分段", fontsize=12)  # 设置x轴标签
plt.ylabel("购买次数", fontsize=12)  # 设置y轴标签
plt.xticks(rotation=0, fontsize=10)  # 设置x轴刻度
plt.legend(title="商品大类", prop={"size": 10})  # 设置图例
plt.tight_layout()  # 调整布局
plt.savefig("age_category.png")  # 保存图形
print("年龄与商品类别关系图已保存: age_category.png")  # 打印保存信息

member_stats = df.groupBy("是否会员").agg(avg("单价").alias("平均单价"), avg("年龄").alias("平均年龄")).toPandas()  # 计算会员与非会员的统计指标
plt.figure(figsize=(14, 6))  # 创建图形
ax1 = plt.subplot(1, 2, 1)  # 创建第一个子图
sns.barplot(data=member_stats, x="是否会员", y="平均单价", hue="是否会员", palette="pastel", legend=False, ax=ax1)  # 绘制平均单价柱状图
plt.title("会员与非会员的平均消费金额", fontsize=12)  # 设置标题
plt.xlabel("会员状态", fontsize=10)  # 设置x轴标签
plt.ylabel("平均单价 (元)", fontsize=10)  # 设置y轴标签
ax2 = plt.subplot(1, 2, 2)  # 创建第二个子图
sns.barplot(data=member_stats, x="是否会员", y="平均年龄", hue="是否会员", palette="pastel", legend=False, ax=ax2)  # 绘制平均年龄柱状图
plt.title("会员与非会员的平均年龄", fontsize=12)  # 设置标题
plt.xlabel("会员状态", fontsize=10)  # 设置x轴标签
plt.ylabel("平均年龄 (岁)", fontsize=10)  # 设置y轴标签
plt.tight_layout()  # 调整布局
plt.savefig("member_stats.png")  # 保存图形
print("会员与非会员消费差异图已保存: member_stats.png")  # 打印保存信息

province_stats = df.groupBy("收货省份").count().orderBy("count", ascending=False).limit(10).toPandas()  # 计算前10个收货省份的统计指标
plt.figure(figsize=(12, 6))  # 创建图形
sns.barplot(data=province_stats, x="count", y="收货省份", hue="收货省份", palette="mako", legend=False)  # 绘制收货省份分布条形图
plt.title("各省份消费分布（前10）", fontsize=14)  # 设置标题
plt.xlabel("消费次数", fontsize=12)  # 设置x轴标签
plt.ylabel("省份", fontsize=12)  # 设置y轴标签
plt.tight_layout()  # 调整布局
plt.savefig("province_distribution.png")  # 保存图形
print("地域消费差异图已保存: province_distribution.png")  # 打印保存信息

# ======================== 生成分析报告（简化版） ========================
print("\n\n===== 消费者偏好分析报告（简化版） =====")  # 打印报告标题
print(f"1. 基础数据")  # 打印基础数据标题
print(f"- 数据总量: {df.count()} 条")  # 打印数据总量
print(f"- 平均年龄: {mean_age:.2f} 岁 | 平均单价: {mean_price:.2f} 元")  # 打印平均年龄和单价
print(f"- 女性占比: {gender_distribution[gender_distribution['性别'] == '女']['count'].values[0] / df.count() * 100:.2f}%")  # 打印女性占比
print(f"- 会员占比: {df.filter(col('是否会员') == '是').count() / df.count() * 100:.2f}%")  # 打印会员占比
print(f"\n2. 商品销售")  # 打印商品销售标题
top_category, bottom_category = category_distribution.sort_values("count", ascending=False).iloc[0], category_distribution.sort_values("count", ascending=True).iloc[0]  # 获取最畅销和最冷门商品
print(f"- 最畅销: {top_category['商品大类']} ({top_category['count']} 销量, {top_category['count'] / df.count() * 100:.2f}%)")  # 打印最畅销商品
print(f"- 最冷门: {bottom_category['商品大类']} ({bottom_category['count']} 销量, {bottom_category['count'] / df.count() * 100:.2f}%)")  # 打印最冷门商品
print(f"\n3. 特征重要性")  # 打印特征重要性标题
if best_model_name == "随机森林":  # 如果最佳模型是随机森林
    print(f"- 年龄: {feature_importance[feature_names.index('年龄分段')]:.4f} | 性别: {feature_importance[feature_names.index('性别')]:.4f}")  # 打印年龄和性别特征重要性
print(f"\n4. 建议")  # 打印建议标题
print(f"- 针对 {age_labels[age_category.set_index('年龄分段').sum(axis=1).argmax()]} 推广 {top_category['商品大类']}")  # 打印针对特定年龄段推广畅销商品的建议
print(f"- 优化会员权益（当前会员平均消费与非会员差异需结合业务校验）")  # 打印优化会员权益的建议

# ======================== 关闭 SparkSession ========================
spark.stop()  # 关闭SparkSession，释放资源