=============================================================================
INFOSYS 722 - Iteration 4: Fraud Detection (BDAS Solution)
Google Colab 运行指南
=============================================================================

学生姓名: Xiaoqing Miao
学号: xmia665@aucklanduni.ac.nz
提交日期: 2025-10-15

=============================================================================
快速开始 (3 步)
=============================================================================

步骤 1: 解压文件
  - 解压 zip 文件到本地文件夹
  - 应包含以下文件:
    ✓ fraud_detection_colab.py
    ✓ Financial Fraud-Rawdata.csv
    ✓ README_COLAB.txt (本文件)

步骤 2: 上传到 Google Colab
  1. 访问 https://colab.research.google.com/
  2. 创建新笔记本或打开现有笔记本
  3. 点击左侧文件夹图标 📁
  4. 点击上传按钮
  5. 上传 fraud_detection_colab.py
  6. 上传 Financial Fraud-Rawdata.csv (约需 5-10 分钟)

步骤 3: 运行脚本
  在 Colab 的代码单元格中输入并运行:
  
    !python fraud_detection_colab.py
  
  或者:
  
    %run fraud_detection_colab.py

=============================================================================
自动化功能说明
=============================================================================

脚本会自动执行以下操作:

1. 环境配置 (自动)
   ✓ 检测 Colab 环境
   ✓ 安装 Java 11 (Spark 依赖)
   ✓ 安装 PySpark
   ✓ 验证安装

2. 数据处理 (自动)
   ✓ 如果数据文件未找到，会弹出上传对话框
   ✓ 加载 6.36M 条交易记录
   ✓ 数据质量检查
   ✓ 特征工程 (过滤、编码、log1p 转换)

3. 模型训练 (自动)
   ✓ 70/15/15 数据分割
   ✓ 类别加权处理不平衡
   ✓ Pipeline 训练 (Assembler → Scaler → LogisticRegression)
   ✓ 保存模型

4. 阈值优化 (自动)
   ✓ 在验证集上扫描 99 个阈值点
   ✓ 使用业务成本函数 (FP:FN = 1:25)
   ✓ 找到最优阈值 τ*

5. 测试评估 (自动)
   ✓ 在测试集上应用 τ*
   ✓ 计算 ROC-AUC, PR-AUC, Precision, Recall, F1
   ✓ 生成 ROC 和 PR 曲线数据

=============================================================================
预期执行时间
=============================================================================

阶段                          时间
--------------------------------------------
环境设置 (自动)              1-2 分钟
数据上传 (如需要)            5-10 分钟  
数据加载与质量检查           1-2 分钟
特征工程                     1 分钟
模型训练                     2-3 分钟
阈值优化                     1-2 分钟
测试集评估                   1 分钟
--------------------------------------------
总计                         15-25 分钟

=============================================================================
预期输出文件
=============================================================================

执行完成后会在 Colab 文件系统中生成两个文件夹:

1. outputs/ 目录 (结果文件)
   ├── metrics_test_at_tau.json      测试集性能指标
   ├── tau_star.json                 最优阈值信息
   ├── lr_coefficients.csv           模型系数
   ├── cost_curve.csv                成本曲线数据
   ├── roc_points.csv                ROC 曲线坐标点
   ├── pr_points.csv                 PR 曲线坐标点
   ├── class_counts_train.csv        训练集类别分布
   ├── class_counts_validation.csv   验证集类别分布
   ├── class_counts_test.csv         测试集类别分布
   ├── dist_evidence.json            分布式执行证据
   ├── quality_null_counts.json      空值统计
   ├── quality_duplicates.json       重复行统计
   └── schema.txt                    数据集 schema

2. models/ 目录 (训练模型)
   └── bdas_lr_pipeline_YYYYMMDD_HHMMSS/
       ├── metadata/                 Pipeline 元数据
       └── stages/                   各阶段模型
           ├── 0_VectorAssembler_*/
           ├── 1_StandardScaler_*/
           └── 2_LogisticRegression_*/

可以右键点击文件夹选择 Download 下载所有结果。

=============================================================================
关键性能指标 (预期结果)
=============================================================================

测试集性能 (阈值 τ* = 0.72):
  ROC-AUC:   0.9613
  PR-AUC:    0.5903
  Precision: 0.3494
  Recall:    0.6196
  F1-Score:  0.4468

混淆矩阵:
              Predicted
              0        1
  Actual  0   412,658  1,462   (TN, FP)
          1   482      785     (FN, TP)

分布式配置:
  Spark Master:        local[*]
  Default Parallelism: 8
  Driver Memory:       8GB
  Train Partitions:    8

与 Iteration 3 (OSAS) 对比:
  指标          I3 (OSAS)  I4 (BDAS)  Δ (I4-I3)
  -----------------------------------------------
  ROC-AUC       0.9756     0.9613     -0.0143
  PR-AUC        0.5821     0.5903     +0.0082
  Recall        0.6595     0.6196     -0.0399
  Precision     0.2742     0.3494     +0.0752
  F1-Score      0.3870     0.4468     +0.0598

结论: BDAS 方案成功保持了与 OSAS 方案的性能一致性，
      验证了技术栈迁移的可行性。

=============================================================================
验证成功运行
=============================================================================

执行完成后，终端应显示:

  ============================================================
  TEST SET PERFORMANCE SUMMARY
  ============================================================
  ROC-AUC:    0.9613
  PR-AUC:     0.5903
  Precision:  0.3494
  Recall:     0.6196
  F1-Score:   0.4468
  Confusion Matrix: TP=785, FP=1462, TN=412658, FN=482
  ============================================================
  
  ============================================================
  PIPELINE COMPLETE!
  ============================================================
  
  Outputs saved to: outputs/
  Model saved to: models/bdas_lr_pipeline_YYYYMMDD_HHMMSS
  
  📥 To download results:
  1. Click the folder icon on the left sidebar
  2. Right-click 'outputs' folder → Download
  3. Right-click 'models' folder → Download
  
  ✓ SparkSession stopped

=============================================================================
故障排除
=============================================================================

问题 1: "Data file not found" 错误
  原因: 数据文件未上传或文件名不匹配
  解决: 
    - 确保文件名完全匹配: Financial Fraud-Rawdata.csv
    - 脚本会自动弹出上传对话框，选择正确的文件
    - 或手动上传到 Colab 根目录

问题 2: 内存不足 (OOM) 错误
  原因: Colab 免费版 RAM 限制 (约 12GB)
  解决: 
    - 脚本已配置 Spark Driver Memory = 8GB
    - 如仍有问题，可修改脚本第 39 行:
      SPARK_DRIVER_MEMORY = "6g"
    - 或使用 Colab Pro (更多 RAM)

问题 3: Java 未安装或版本不匹配
  原因: 环境配置失败
  解决: 
    - 脚本会自动安装 Java 11
    - 如失败，重启 Colab runtime 后重新运行
    - 或手动运行环境设置部分

问题 4: PySpark 导入错误
  原因: PySpark 未正确安装
  解决:
    - 脚本会自动安装 PySpark
    - 如失败，手动运行: !pip install pyspark
    - 验证: import pyspark; print(pyspark.__version__)

问题 5: 执行时间过长
  说明: 
    - 完整流程需要 15-25 分钟是正常的
    - 数据集有 6.36M 行，处理需要时间
    - Colab 免费版 CPU 较慢，请耐心等待
    - 可以看到每个步骤的进度提示

=============================================================================
技术细节
=============================================================================

数据集: PaySim Synthetic Financial Dataset
  - 来源: Kaggle
  - 规模: 6,362,620 条交易记录
  - 特征: 11 个字段
  - 标签: isFraud (0/1)
  - 类别不平衡: 欺诈率 0.13%

算法: Logistic Regression (与 Iteration 3 保持一致)
  - 不平衡处理: 类别加权 (weightCol)
  - 正则化: L2 (regParam=0.01)
  - Pipeline: VectorAssembler → StandardScaler → LR

阈值优化:
  - 成本比例: FP:FN = 1:25 (业务驱动)
  - 优化集: 验证集 (独立于训练和测试)
  - 扫描范围: τ ∈ [0.01, 0.99]，步长 0.01
  - 最优阈值: τ* = 0.72

=============================================================================
系统要求
=============================================================================

Google Colab 环境:
  ✓ Python 3.10+
  ✓ 12GB RAM (免费版)
  ✓ 网络连接 (用于安装依赖和上传数据)

自动安装的依赖:
  ✓ OpenJDK 11
  ✓ PySpark 3.4+

=============================================================================
项目结构
=============================================================================

完整项目包含以下文件 (本提交仅包含 Colab 运行所需文件):

核心文件 (已包含):
  ✓ fraud_detection_colab.py     Colab 适配脚本
  ✓ Financial Fraud-Rawdata.csv  PaySim 数据集
  ✓ README_COLAB.txt             本说明文件

其他文件 (在完整项目中):
  - fraud_detection_final.py     本地运行版本
  - plot_results.py              结果可视化脚本
  - 722-assignment04.tex         LaTeX 报告源文件
  - 722-assignment04.pdf         最终报告 PDF
  - Figure/                      报告图表目录

=============================================================================
引用与致谢
=============================================================================

数据集:
  PaySim: Synthetic Financial Dataset for Fraud Detection
  Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1

技术框架:
  - Apache Spark: https://spark.apache.org/
  - PySpark MLlib: https://spark.apache.org/docs/latest/ml-guide.html

课程:
  INFOSYS 722 - Big Data Analytics Solutions
  University of Auckland, 2025 Semester 2

=============================================================================
联系方式
=============================================================================

如有任何问题或需要澄清，请联系:

  学生: Xiaoqing Miao
  邮箱: xmia665@aucklanduni.ac.nz
  机构: University of Auckland
  课程: INFOSYS 722

=============================================================================
学术诚信声明
=============================================================================

本项目为原创学术作业，遵循奥克兰大学学术诚信政策。
所有代码、分析和结论均为本人独立完成。
数据使用已获得适当授权。

参考: https://www.auckland.ac.nz/en/students/forms-policies-and-guidelines/student-policies-and-guidelines/academic-integrity-copyright.html

=============================================================================
最后更新: 2025-10-15
版本: Iteration 4 (BDAS - Google Colab)
=============================================================================

