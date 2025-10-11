#!/bin/bash

# 干净输出的Spark运行脚本
echo "=== 运行Spark脚本（干净输出模式）==="

# 设置环境变量
export SPARK_HOME="/opt/anaconda3/lib/python3.13/site-packages/pyspark"
export JAVA_HOME=$(/usr/libexec/java_home)
export PATH="$SPARK_HOME/bin:$PATH"
export PYTHONPATH="$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.9-src.zip:$PYTHONPATH"

# 使用自定义日志配置运行
spark-submit \
  --master "local[*]" \
  --driver-memory 4g \
  --executor-memory 2g \
  --conf "spark.driver.extraJavaOptions=-Dlog4j.configuration=file:$(pwd)/log4j2.properties" \
  --conf "spark.executor.extraJavaOptions=-Dlog4j.configuration=file:$(pwd)/log4j2.properties" \
  fraud_detection_spark.py

echo "=== 运行完成 ==="