#1.构建 ODS 层表（原始数据层，存储未加工原始数据 ）
#使用 ods_db 存储 ODS 层表
USE ods_db;
#创建 ODS 层表
CREATE TABLE IF NOT EXISTS ods_product_orders (
    order_id BIGINT,
    user_id BIGINT,
    order_time STRING,  -- 暂时保留字符串类型，后续 DWD 层处理时间格式
    product_category STRING,
    product_subcategory STRING,
    product_name STRING,
    unit_price DOUBLE,
    quantity BIGINT,
    total_amount DOUBLE,
    province STRING,
    city STRING,
    gender STRING,
    age BIGINT,
    is_member STRING
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ','
STORED AS TEXTFILE;
#加载数据到 ODS 层表
LOAD DATA INPATH '/data/data_utf8.csv' INTO TABLE ods_product_orders;
#验证 ODS 层表数据
#执行简单查询，查看数据是否加载成功（取前 5 条示例 ）：
SELECT order_id, product_category, province FROM ods_product_orders LIMIT 5;

#2.构建 DWD 层表（清洗层，处理时间格式、过滤无效数据 ）
#使用 dwd_db 存储 DWD 层表
USE dwd_db;
#插入清洗后数据到 DWD 层表
#从 ODS 层表读取数据，处理时间格式（将原始 order_time 转换为标准时间格式 ）、过滤空值
INSERT INTO TABLE dwd_product_orders
SELECT
    order_id,
    user_id,
    FROM_UNIXTIME(
        UNIX_TIMESTAMP(
            CONCAT(
                '2024-01-01 ',  -- 假设日期为 2024-01-01，实际需按业务调整
                SPLIT(order_time, '\\.')[0],  -- 提取 HH:mm 部分
                ':',
                LPAD(CAST(ROUND(SPLIT(order_time, '\\.')[1] * 60) AS INT), 2, '0')  -- 将 .ss 转换为秒（如 .2 → 12 秒）
            ),
            'yyyy-MM-dd HH:mm:ss'
        )
    ) AS order_time,
    product_category,
    product_subcategory,
    product_name,
    unit_price,
    quantity,
    total_amount,
    province,
    city,
    gender,
    age,
    is_member
FROM ods_db.ods_product_orders
WHERE order_id IS NOT NULL AND user_id IS NOT NULL;  -- 过滤无效空值数据
#验证 DWD 层表数据
#查询处理后的数据（取前 5 条示例 ），重点检查时间格式是否转换正确
SELECT order_id, order_time, product_name FROM dwd_product_orders LIMIT 5;

#3.构建 DWS 层表（统计层，按商品类别、省份等维度统计 ）
#使用 dws_db 存储 DWS 层表
USE dws_db;
#创建 DWS 层统计表（以商品类别、省份维度为例 ）
#按商品类别统计的表：
CREATE TABLE IF NOT EXISTS dws_sales_by_category (
    product_category STRING COMMENT '商品类别',
    total_orders BIGINT COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总销售额',
    avg_price DOUBLE COMMENT '平均单价',
    create_time STRING COMMENT '统计时间'
)
PARTITIONED BY (dt STRING)  -- 按日期分区，便于增量统计
STORED AS PARQUET;  -- 使用 Parquet 格式存储，优化查询性能
#按省份统计的表：
CREATE TABLE IF NOT EXISTS dws_sales_by_province (
    province STRING COMMENT '省份',
    total_orders BIGINT COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总销售额',
    user_count BIGINT COMMENT '购买用户数',
    create_time STRING COMMENT '统计时间'
)
PARTITIONED BY (dt STRING)
STORED AS PARQUET;
#插入统计数据到 DWS 层表
#按商品类别统计并插入数据
#从 DWD 层表读取数据，按商品类别聚合，统计订单数、销售额等指标，插入 dws_sales_by_category 表（分区 dt='2024-01-01' 示例 ）
INSERT OVERWRITE TABLE dws_sales_by_category PARTITION (dt='2024-01-01')
SELECT
    product_category,
    COUNT(*) AS total_orders,
    SUM(total_amount) AS total_amount,
    AVG(unit_price) AS avg_price,
    '2024-01-01' AS create_time
FROM dwd_db.dwd_product_orders
WHERE DATE(order_time) = '2024-01-01'  -- 按日期过滤数据
GROUP BY product_category;
#按省份统计并插入数据
#从 DWD 层表读取数据，按省份聚合，统计订单数、销售额、去重用户数等指标，插入 dws_sales_by_province 表（分区 dt='2024-01-01' 示例 ）
INSERT OVERWRITE TABLE dws_sales_by_province PARTITION (dt='2024-01-01')
SELECT
    province,
    COUNT(*) AS total_orders,
    SUM(total_amount) AS total_amount,
    COUNT(DISTINCT user_id) AS user_count,  -- 去重统计购买用户数
    '2024-01-01' AS create_time
FROM dwd_db.dwd_product_orders
WHERE DATE(order_time) = '2024-01-01'
GROUP BY province;

#4.验证 DWS 层统计结果
#查看表分区
SHOW PARTITIONS dws_sales_by_category;
SHOW PARTITIONS dws_sales_by_province;
#查询统计数据
#按商品类别统计结果（取前 10 条示例 ）：
SELECT * FROM dws_sales_by_category LIMIT 10;
#按省份统计结果（取前 10 条示例 ）：
SELECT * FROM dws_sales_by_province LIMIT 10;




