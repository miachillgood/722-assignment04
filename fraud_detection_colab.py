# -*- coding: utf-8 -*-
"""
INFOSYS 722 — Iteration 4 (BDAS) — Fraud Detection Project
Google Colab Adapted Version

Instructions:
1. Upload this file to Google Colab
2. Upload 'Financial Fraud-Rawdata.csv' to Colab's file system
3. Run all cells sequentially
4. Download output files from 'outputs/' and 'models/' directories

Note: This script installs PySpark and Java automatically in Colab environment.
"""

# =============================================================================
# SECTION 0: Google Colab Environment Setup
# =============================================================================
print("=" * 80)
print("STEP 0: Setting up Google Colab environment for PySpark")
print("=" * 80)

# Install Java (required for Spark)
import os
import sys

# Check if running in Colab
try:
    import google.colab
    IN_COLAB = True
    print("✓ Running in Google Colab")
except ImportError:
    IN_COLAB = False
    print("✗ Not running in Colab - skipping environment setup")

if IN_COLAB:
    # Install OpenJDK 11
    print("\n[1/3] Installing Java...")
    !apt-get install -y openjdk-11-jdk-headless -qq > /dev/null
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64"
    print(f"✓ JAVA_HOME set to: {os.environ['JAVA_HOME']}")
    
    # Install PySpark
    print("\n[2/3] Installing PySpark...")
    !pip install -q pyspark
    print("✓ PySpark installed")
    
    # Verify installation
    print("\n[3/3] Verifying installation...")
    import pyspark
    print(f"✓ PySpark version: {pyspark.__version__}")
    
    # Set working directory
    from google.colab import files
    print("\n" + "=" * 80)
    print("READY TO UPLOAD DATA FILE")
    print("=" * 80)
    print("\nPlease upload 'Financial Fraud-Rawdata.csv' using the file upload button")
    print("or run the following cell to trigger upload:")
    print("  uploaded = files.upload()")
    print("\n" + "=" * 80)

# =============================================================================
# 1. Imports
# =============================================================================
import json
import csv
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

# =============================================================================
# 2. Configuration
# =============================================================================
SPARK_APP_NAME = "FraudDetection_BDAS_Colab"
SPARK_MASTER = "local[*]"
# Colab typically has 12-13GB RAM, allocate 8GB to Spark driver
SPARK_DRIVER_MEMORY = "8g"

# File paths - adjust if needed
INPUT_CSV_PATH = "Financial Fraud-Rawdata.csv"
OUT_DIR = Path("outputs")
MODEL_DIR = Path("models")
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
SPLITS = [0.7, 0.15, 0.15]

LABEL_COL = "isFraud"
SKEWED_COLS = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
FEATURES_FINAL = ["step"] + [f"{c}_log1p" for c in SKEWED_COLS] + ["type_encoded"]

COST_FP, COST_FN = 1.0, 25.0
THRESH_SCAN = [i / 100 for i in range(1, 100)]
VALIDATION_COLLECT_MAX = 500_000  # safety limit when collecting to driver

# =============================================================================
# 3. Helpers
# =============================================================================
def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_csv(rows, header, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def find_optimal_threshold(model, validation_df, fp_cost, fn_cost):
    """Scan thresholds on the validation set to minimize a business cost."""
    print("\nOptimizing threshold on validation set...")
    preds_val = model.transform(validation_df).select(
        vector_to_array("probability")[1].alias("score"),
        F.col(LABEL_COL).cast("int").alias("label")
    )

    val_count = preds_val.count()
    use_df = preds_val
    sampled = False
    if val_count > VALIDATION_COLLECT_MAX:
        print(f"Validation set too large ({val_count}); sampling negatives for threshold search...")
        fractions = {0: 0.1, 1: 1.0}
        sampled = True
        use_df = model.transform(
            validation_df.sampleBy(LABEL_COL, fractions=fractions, seed=SEED)
        ).select(
            vector_to_array("probability")[1].alias("score"),
            F.col(LABEL_COL).cast("int").alias("label")
        )

    probs_labels = [(float(r["score"]), int(r["label"])) for r in use_df.collect()]
    best_tau, min_cost = 0.5, float("inf")
    
    # Record cost curve data
    cost_curve_data = []

    for tau in THRESH_SCAN:
        fp_count, fn_count = 0, 0
        cost = 0.0
        for p, y in probs_labels:
            if p >= tau and y == 0:
                fp_count += 1
                cost += fp_cost
            elif p < tau and y == 1:
                fn_count += 1
                cost += fn_cost
        
        cost_curve_data.append([tau, fp_count, fn_count, cost])
        
        if cost < min_cost:
            min_cost, best_tau = cost, tau

    # Save cost curve data
    save_csv(cost_curve_data, ["threshold", "fp_count", "fn_count", "total_cost"], 
             OUT_DIR / "cost_curve.csv")
    print(f"Cost curve saved to: {OUT_DIR}/cost_curve.csv")

    print(f"Optimal threshold (τ*): {best_tau:.4f} | Min business cost: {min_cost:.2f}")
    save_json(
        {
            "tau_star": best_tau,
            "min_cost": min_cost,
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
            "validation_rows": val_count,
            "validation_rows_used": len(probs_labels),
            "scan_points": len(THRESH_SCAN),
            "sampled": sampled,
            "timestamp": datetime.now().isoformat()
        },
        OUT_DIR / "tau_star.json"
    )
    return best_tau

def evaluate_at_threshold(model, test_df, tau_star):
    """Evaluate model on test set at a fixed threshold."""
    print(f"\nEvaluating on test set at τ* = {tau_star:.4f}...")
    preds = model.transform(test_df).select(
        vector_to_array("probability")[1].alias("score"),
        F.col(LABEL_COL).cast("int").alias("label")
    )
    
    # Compute ROC-AUC and PR-AUC
    evaluator_roc = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction", labelCol=LABEL_COL, metricName="areaUnderROC"
    )
    evaluator_pr = BinaryClassificationEvaluator(
        rawPredictionCol="rawPrediction", labelCol=LABEL_COL, metricName="areaUnderPR"
    )
    
    preds_full = model.transform(test_df)
    roc_auc = evaluator_roc.evaluate(preds_full)
    pr_auc = evaluator_pr.evaluate(preds_full)
    
    # Collect predictions for confusion matrix
    rows = [(float(r["score"]), int(r["label"])) for r in preds.collect()]
    
    tp = fp = tn = fn = 0
    for score, label in rows:
        pred = 1 if score >= tau_star else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        "tau_star": tau_star,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "timestamp": datetime.now().isoformat()
    }
    
    save_json(metrics, OUT_DIR / "metrics_test_at_tau.json")
    print(f"\n{'='*60}")
    print("TEST SET PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"ROC-AUC:    {roc_auc:.4f}")
    print(f"PR-AUC:     {pr_auc:.4f}")
    print(f"Precision:  {precision:.4f}")
    print(f"Recall:     {recall:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print(f"Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print(f"{'='*60}\n")
    
    # Save PR and ROC curve points
    save_pr_roc_curves(rows, tau_star)
    
    return metrics

def save_pr_roc_curves(score_label_pairs, tau_star):
    """Generate PR and ROC curve data points."""
    print("Generating PR and ROC curve data...")
    
    # Sort by score descending
    sorted_pairs = sorted(score_label_pairs, key=lambda x: x[0], reverse=True)
    
    # Calculate cumulative TP, FP for ROC
    total_pos = sum(1 for _, label in sorted_pairs if label == 1)
    total_neg = len(sorted_pairs) - total_pos
    
    roc_points = [[0.0, 0.0]]  # Start at origin
    pr_points = []
    
    tp = fp = 0
    for score, label in sorted_pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        
        tpr = tp / total_pos if total_pos > 0 else 0
        fpr = fp / total_neg if total_neg > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tpr
        
        roc_points.append([fpr, tpr])
        if tp > 0:  # Only add PR points where we have positive predictions
            pr_points.append([recall, precision])
    
    roc_points.append([1.0, 1.0])  # End at (1,1)
    
    save_csv(roc_points, ["fpr", "tpr"], OUT_DIR / "roc_points.csv")
    save_csv(pr_points, ["recall", "precision"], OUT_DIR / "pr_points.csv")
    print(f"✓ ROC curve points saved: {len(roc_points)} points")
    print(f"✓ PR curve points saved: {len(pr_points)} points")

# =============================================================================
# 4. Main Pipeline
# =============================================================================
def main():
    print("\n" + "="*80)
    print("FRAUD DETECTION — BDAS (PySpark) — Google Colab Version")
    print("="*80 + "\n")
    
    # Check if data file exists, if not trigger upload in Colab
    if not Path(INPUT_CSV_PATH).exists():
        print(f"⚠️  Data file '{INPUT_CSV_PATH}' not found!")
        
        if IN_COLAB:
            print("\n📤 Triggering file upload dialog...")
            print("Please select 'Financial Fraud-Rawdata.csv' from your computer\n")
            
            from google.colab import files
            uploaded = files.upload()
            
            if INPUT_CSV_PATH not in uploaded:
                print(f"\n❌ ERROR: Expected file '{INPUT_CSV_PATH}' was not uploaded!")
                print(f"Uploaded files: {list(uploaded.keys())}")
                print("\nPlease ensure the file is named exactly: 'Financial Fraud-Rawdata.csv'")
                return
            
            print(f"✓ File '{INPUT_CSV_PATH}' uploaded successfully!")
            print(f"✓ File size: {len(uploaded[INPUT_CSV_PATH]) / (1024**2):.2f} MB\n")
        else:
            print("\nPlease upload the CSV file using one of these methods:")
            print("1. Click the folder icon on the left sidebar")
            print("2. Click 'Upload' button and select 'Financial Fraud-Rawdata.csv'")
            print("3. Or run: from google.colab import files; files.upload()")
            return
    
    # Initialize Spark
    print("[1/11] Initializing SparkSession...")
    spark = (
        SparkSession.builder
        .appName(SPARK_APP_NAME)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .getOrCreate()
    )
    
    spark.sparkContext.setLogLevel("WARN")
    print(f"✓ Spark Master: {spark.sparkContext.master}")
    print(f"✓ Default Parallelism: {spark.sparkContext.defaultParallelism}")
    
    # Load data
    print(f"\n[2/11] Loading data from {INPUT_CSV_PATH}...")
    df_raw = spark.read.csv(INPUT_CSV_PATH, header=True, inferSchema=True)
    print(f"✓ Loaded {df_raw.count():,} rows, {len(df_raw.columns)} columns")
    
    # Save schema
    with open(OUT_DIR / "schema.txt", "w") as f:
        f.write(df_raw._jdf.schema().treeString())
    
    # Data quality checks
    print("\n[3/11] Data quality checks...")
    null_counts = {c: df_raw.filter(F.col(c).isNull()).count() for c in df_raw.columns}
    save_json(null_counts, OUT_DIR / "quality_null_counts.json")
    print(f"✓ Null counts saved")
    
    dup_count = df_raw.count() - df_raw.dropDuplicates().count()
    save_json({"duplicate_rows": dup_count}, OUT_DIR / "quality_duplicates.json")
    print(f"✓ Duplicate check: {dup_count} duplicates found")
    
    # Filter to TRANSFER and CASH_OUT only
    print("\n[4/11] Filtering to TRANSFER and CASH_OUT transactions...")
    df_filtered = df_raw.filter(F.col("type").isin("TRANSFER", "CASH_OUT"))
    filtered_count = df_filtered.count()
    print(f"✓ Filtered to {filtered_count:,} rows")
    
    # Manual encoding for 'type'
    print("\n[5/11] Encoding 'type' column...")
    df_encoded = df_filtered.withColumn(
        "type_encoded",
        F.when(F.col("type") == "CASH_OUT", 0)
         .when(F.col("type") == "TRANSFER", 1)
         .otherwise(-1)
    )
    
    # Apply log1p transformation
    print("\n[6/11] Applying log1p transformation to skewed features...")
    for col_name in SKEWED_COLS:
        df_encoded = df_encoded.withColumn(f"{col_name}_log1p", F.log1p(F.col(col_name)))
    
    # Select final features
    df_final = df_encoded.select(FEATURES_FINAL + [LABEL_COL])
    print(f"✓ Final feature set: {FEATURES_FINAL}")
    
    # Train/Val/Test split
    print(f"\n[7/11] Splitting data ({SPLITS[0]:.0%}/{SPLITS[1]:.0%}/{SPLITS[2]:.0%})...")
    train_df, val_df, test_df = df_final.randomSplit(SPLITS, seed=SEED)
    train_df.cache()
    val_df.cache()
    test_df.cache()
    
    train_count = train_df.count()
    val_count = val_df.count()
    test_count = test_df.count()
    
    print(f"✓ Train: {train_count:,} rows")
    print(f"✓ Validation: {val_count:,} rows")
    print(f"✓ Test: {test_count:,} rows")
    
    # Save split counts
    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        class_counts = df.groupBy(LABEL_COL).count().toPandas()
        class_counts.to_csv(OUT_DIR / f"class_counts_{name}.csv", index=False)
    
    # Calculate class weights
    print("\n[8/11] Calculating class weights...")
    label_counts = train_df.groupBy(LABEL_COL).count().collect()
    label_dict = {int(row[LABEL_COL]): row["count"] for row in label_counts}
    total = sum(label_dict.values())
    weight_0 = total / (2.0 * label_dict[0])
    weight_1 = total / (2.0 * label_dict[1])
    
    print(f"✓ Class 0 weight: {weight_0:.4f}")
    print(f"✓ Class 1 weight: {weight_1:.4f}")
    
    train_df = train_df.withColumn(
        "weight",
        F.when(F.col(LABEL_COL) == 0, weight_0).otherwise(weight_1)
    )
    
    # Build ML Pipeline
    print("\n[9/11] Building ML Pipeline...")
    assembler = VectorAssembler(inputCols=FEATURES_FINAL, outputCol="features_raw")
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=False
    )
    lr = LogisticRegression(
        featuresCol="features",
        labelCol=LABEL_COL,
        weightCol="weight",
        maxIter=100,
        regParam=0.01,
        elasticNetParam=0.0
    )
    
    pipeline = Pipeline(stages=[assembler, scaler, lr])
    
    print("✓ Training model...")
    model = pipeline.fit(train_df)
    print("✓ Model training complete")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"bdas_lr_pipeline_{timestamp}"
    model.write().overwrite().save(str(model_path))
    print(f"✓ Model saved to: {model_path}")
    
    # Extract and save coefficients
    lr_model = model.stages[-1]
    coeffs = lr_model.coefficients.toArray()
    intercept = lr_model.intercept
    
    coeff_data = [[feat, float(coef), abs(float(coef))] 
                  for feat, coef in zip(FEATURES_FINAL, coeffs)]
    coeff_data.append(["intercept", intercept, abs(intercept)])
    save_csv(coeff_data, ["feature", "coefficient", "abs_coefficient"], 
             OUT_DIR / "lr_coefficients.csv")
    print("✓ Coefficients saved")
    
    # Threshold optimization
    print("\n[10/11] Threshold optimization...")
    tau_star = find_optimal_threshold(model, val_df, COST_FP, COST_FN)
    
    # Final evaluation
    print("\n[11/11] Final evaluation on test set...")
    metrics = evaluate_at_threshold(model, test_df, tau_star)
    
    # Save distributed evidence
    evidence = {
        "spark_master": spark.sparkContext.master,
        "default_parallelism": spark.sparkContext.defaultParallelism,
        "driver_memory": SPARK_DRIVER_MEMORY,
        "train_partitions": train_df.rdd.getNumPartitions(),
        "val_partitions": val_df.rdd.getNumPartitions(),
        "test_partitions": test_df.rdd.getNumPartitions(),
        "timestamp": datetime.now().isoformat()
    }
    save_json(evidence, OUT_DIR / "dist_evidence.json")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nOutputs saved to: {OUT_DIR}/")
    print(f"Model saved to: {model_path}")
    
    if IN_COLAB:
        print("\n📥 To download results:")
        print("1. Click the folder icon on the left sidebar")
        print("2. Right-click 'outputs' folder → Download")
        print("3. Right-click 'models' folder → Download")
    
    spark.stop()
    print("\n✓ SparkSession stopped")

# =============================================================================
# Run Main
# =============================================================================
if __name__ == "__main__":
    main()

