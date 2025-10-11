#!/bin/bash

# 修复Spark环境配置脚本
echo "=== 修复Spark环境配置 ==="

# 设置SPARK_HOME
export SPARK_HOME="/opt/anaconda3/lib/python3.13/site-packages/pyspark"
echo "SPARK_HOME设置为: $SPARK_HOME"

# 设置JAVA_HOME
export JAVA_HOME=$(/usr/libexec/java_home)
echo "JAVA_HOME设置为: $JAVA_HOME"

# 设置PATH
export PATH="$SPARK_HOME/bin:$PATH"
echo "PATH已更新"

# 设置PYTHONPATH
export PYTHONPATH="$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.9-src.zip:$PYTHONPATH"
echo "PYTHONPATH已更新"

# 验证设置
echo ""
echo "=== 验证设置 ==="
echo "SPARK_HOME: $SPARK_HOME"
echo "JAVA_HOME: $JAVA_HOME"
echo "Python版本: $(python3 --version)"
echo "Java版本:"
java -version

echo ""
echo "=== 检查关键文件 ==="
echo "spark-submit位置: $(which spark-submit)"
echo "spark-class位置: $SPARK_HOME/bin/spark-class"

if [ -f "$SPARK_HOME/bin/spark-class" ]; then
    echo "✅ spark-class 文件存在"
else
    echo "❌ spark-class 文件不存在"
fi

if [ -f "$SPARK_HOME/bin/spark-submit" ]; then
    echo "✅ spark-submit 文件存在"
else
    echo "❌ spark-submit 文件不存在"
fi

echo ""
echo "=== 环境修复完成 ==="
echo "现在可以尝试运行: spark-submit --master \"local[*]\" --driver-memory 4g --executor-memory 2g fraud_detection_spark.py"