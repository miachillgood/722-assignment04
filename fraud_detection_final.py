# -*- coding: utf-8 -*-
"""
INFOSYS 722 — Iteration 4 (BDAS) — Fraud Detection Project (Final Optimized Version)

End-to-end PySpark workflow:
  1) Init SparkSession
  2) Load PaySim CSV
  3) Data quality checks (nulls, duplicates, ranges, balance consistency)
  4) EDA aggregates (type × label)
  5) Controlled feature engineering (filtering, manual encoding, log1p)
  6) 70/15/15 split (train/val/test), cache
  7) Class imbalance handling via weightCol
  8) ML pipeline (Assembler → Scaler → LogisticRegression)
  9) Threshold tuning on validation (FP:FN = 1:25)
 10) Final evaluation on test (lock τ*), export metrics + PR/ROC points
 11) Save model, coefficients, and distributed evidence
"""

# =============================================================================
# 1. Imports
# =============================================================================
import os, json, csv
from datetime import datetime
from pathlib import Path

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array  # robust way to index probability

# =============================================================================
# 2. Configuration
# =============================================================================
SPARK_APP_NAME = "FraudDetection_BDAS_Final"
SPARK_MASTER = "local[*]"
SPARK_DRIVER_MEMORY = "8g"

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
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        OUT_DIR / "tau_star.json",
    )
    return best_tau

def evaluate_on_test_set(model, test_df, optimal_tau):
    """Final evaluation on the test set with a fixed optimal threshold."""
    print(f"\n--- Final Evaluation on Test Set (at τ* = {optimal_tau:.4f}) ---")
    preds = model.transform(test_df).select(
        "rawPrediction",
        vector_to_array("probability")[1].alias("proba1"),
        F.col(LABEL_COL).cast("int").alias("label")
    )

    # Threshold-independent metrics
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction")
    roc_auc = evaluator.setMetricName("areaUnderROC").evaluate(preds)
    pr_auc  = evaluator.setMetricName("areaUnderPR").evaluate(preds)
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR  AUC: {pr_auc:.4f}")

    # Threshold-dependent metrics
    preds_y = preds.withColumn("yhat", (F.col("proba1") >= F.lit(optimal_tau)).cast("int"))
    tp = preds_y.filter("yhat = 1 AND label = 1").count()
    fp = preds_y.filter("yhat = 1 AND label = 0").count()
    tn = preds_y.filter("yhat = 0 AND label = 0").count()
    fn = preds_y.filter("yhat = 0 AND label = 1").count()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(f"TP: {tp:<6} FP: {fp:<6} | FN: {fn:<6} TN: {tn:<6}")

    save_json(
        {
            "tau_star": optimal_tau,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
        OUT_DIR / "metrics_test_at_tau.json",
    )
    save_csv(
        [[optimal_tau, roc_auc, pr_auc, precision, recall, f1, tp, fp, tn, fn]],
        ["tau_star", "roc_auc", "pr_auc", "precision", "recall", "f1", "tp", "fp", "tn", "fn"],
        OUT_DIR / "metrics_test_at_tau.csv",
    )

    # Export PR/ROC curve points (numpy fallback, no sklearn dependency)
    try:
        import numpy as np
        pdf = preds.select("proba1", "label").toPandas()
        scores = pdf["proba1"].to_numpy(dtype=float)
        labels = pdf["label"].to_numpy(dtype=int)

        order = np.argsort(-scores)
        scores = scores[order]; labels = labels[order]
        P = labels.sum(); N = len(labels) - P
        tp_cum = np.cumsum(labels)
        fp_cum = np.cumsum(1 - labels)

        # PR
        recall_pts = tp_cum / (P if P > 0 else 1)
        prec_pts   = tp_cum / (tp_cum + fp_cum + 1e-12)
        recall_curve = np.concatenate(([0.0], recall_pts, [1.0]))
        prec_curve   = np.concatenate(([prec_pts[0]], prec_pts, [prec_pts[-1]]))

        # ROC
        tpr_pts = tp_cum / (P if P > 0 else 1)
        fpr_pts = fp_cum / (N if N > 0 else 1)
        tpr_curve = np.concatenate(([0.0], tpr_pts, [1.0]))
        fpr_curve = np.concatenate(([0.0], fpr_pts, [1.0]))

        save_csv([(float(fpr), float(tpr)) for fpr, tpr in zip(fpr_curve, tpr_curve)],
                 ["FPR", "TPR"], OUT_DIR / "roc_points.csv")
        save_csv([(float(r), float(p)) for r, p in zip(recall_curve, prec_curve)],
                 ["Recall", "Precision"], OUT_DIR / "pr_points.csv")
        print("ROC/PR curve points saved to outputs/ (numpy fallback).")
    except Exception as e:
        print(f"Warning: Could not save ROC/PR curves: {e}")
        save_csv([(0.0, 0.0), (1.0, 1.0)], ["FPR", "TPR"], OUT_DIR / "roc_points.csv")
        save_csv([(0.0, 1.0), (1.0, 0.0)], ["Recall", "Precision"], OUT_DIR / "pr_points.csv")

def lr_coefficients_to_csv(trained_pipeline_model, feature_names, out_path: Path):
    """Map LR coefficients to feature names and save (sorted by |coef|)."""
    lr_model = trained_pipeline_model.stages[-1]
    rows = sorted(
        [(fn, float(w), float(abs(w))) for fn, w in zip(feature_names, lr_model.coefficients)],
        key=lambda x: x[2], reverse=True
    )
    save_csv(rows, ["feature", "coefficient", "abs_coefficient"], out_path)

# =============================================================================
# 4. Main
# =============================================================================
if __name__ == "__main__":
    # Stop any existing Spark sessions first
    try:
        SparkSession.builder.getOrCreate().stop()
    except:
        pass
    
    # Init Spark
    spark = SparkSession.builder \
        .appName(SPARK_APP_NAME) \
        .master(SPARK_MASTER) \
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.driver.host", "localhost") \
        .config("spark.driver.bindAddress", "localhost") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    print("SparkSession initialized successfully.")

    # Load data
    try:
        raw = spark.read.csv(INPUT_CSV_PATH, header=True, inferSchema=True)
        print(f"Data loaded successfully: {raw.count()} rows, {len(raw.columns)} columns")
    except Exception as e:
        print(f"Failed to load data. Check path. Error: {e}")
        spark.stop()
        raise SystemExit(1)

    # === 2.4 Data Quality Checks ===
    print("\n=== Data Quality Checks ===")
    # 1) Missing values
    nulls = raw.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in raw.columns])
    nulls.show(truncate=False)
    save_csv([[c, int(nulls.collect()[0][c])] for c in raw.columns],
             ["column", "null_count"], OUT_DIR / "quality_null_counts.csv")

    # 2) Duplicates (memory-optimized approach)
    print("Checking for duplicates...")
    try:
        # Use a more memory-efficient approach for large datasets
        total_count = raw.count()
        unique_count = raw.dropDuplicates().count()
        dup_count = total_count - unique_count
        print(f"Duplicate rows: {dup_count}")
        save_csv([[dup_count]], ["duplicate_rows"], OUT_DIR / "quality_duplicates.csv")
    except Exception as e:
        print(f"Warning: Could not check duplicates due to memory constraints: {e}")
        # Save a placeholder value
        save_csv([["N/A - Memory constraint"]], ["duplicate_rows"], OUT_DIR / "quality_duplicates.csv")

    # 3) Range / validity checks
    num_cols = ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]
    neg_counts = raw.select([F.sum((F.col(c) < 0).cast("int")).alias(c) for c in num_cols])
    neg_counts.show(truncate=False)
    neg_counts.coalesce(1).write.mode("overwrite").option("header", True) \
              .csv(str(OUT_DIR / "quality_negative_value_counts"))
    zero_amt = raw.filter(F.col("amount") == 0).count()
    print(f"Zero-amount transactions: {zero_amt}")
    raw.select("amount").summary("count","min","50%","90%","99%","max").show()

    # 4) Business consistency (on TRANSFER/CASH_OUT)
    df_two = raw.filter(F.col("type").isin("TRANSFER","CASH_OUT")).select(
        "amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"
    )
    eps = 1e-6
    ok_org = F.abs(F.col("oldbalanceOrg") - (F.col("newbalanceOrig") + F.col("amount"))) < eps
    ok_dst = F.abs(F.col("newbalanceDest") - (F.col("oldbalanceDest") + F.col("amount"))) < eps
    total_two = df_two.count()
    ok_both  = df_two.filter(ok_org & ok_dst).count()
    print(f"Balance-consistency (both hold) on TRANSFER/CASH_OUT: {ok_both}/{total_two} "
          f"({ok_both/total_two:.2%})")

    # === 2.3 EDA Aggregates ===
    print("\n=== EDA: Transaction Types ===")
    type_counts = raw.groupBy("type", LABEL_COL).count()
    type_counts.orderBy("type", LABEL_COL).show()
    type_counts.coalesce(1).write.mode("overwrite").option("header", True) \
               .csv(str(OUT_DIR / "eda_type_by_label_counts"))

    # === 2.3.1 EDA Viz: Type × Label (bar) ===
    print("\n=== EDA Viz: Type × Label (bar) ===")
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
        import numpy as np

        # Convert to pandas (small table)
        pdf_counts = type_counts.toPandas()
        pivot = (pdf_counts
                 .pivot_table(index="type", columns=LABEL_COL, values="count", fill_value=0)
                 .rename(columns={0: "not_fraud", 1: "fraud"})
                 .sort_index())

        # Save pivot for appendix/reference
        pivot.to_csv(OUT_DIR / "eda_type_by_label_pivot.csv", index=True)

        # Grouped bar plot
        types = pivot.index.tolist()
        x = np.arange(len(types))
        w = 0.38
        fig = plt.figure(figsize=(8, 4.5), dpi=160)
        ax = fig.add_subplot(111)
        ax.bar(x - w/2, pivot["not_fraud"].values, width=w, label="isFraud=0")
        ax.bar(x + w/2, pivot["fraud"].values,     width=w, label="isFraud=1")
        ax.set_xticks(x)
        ax.set_xticklabels(types, rotation=0)
        ax.set_xlabel("Transaction Type")
        ax.set_ylabel("Count")
        ax.set_title("Type × Label Counts (Raw Data)")
        ax.legend()
        fig.tight_layout()
        figpath1 = OUT_DIR / "fig_2_3a_type_by_label_bar.png"
        fig.savefig(figpath1)
        plt.close(fig)
        print(f"Saved: {figpath1}")
    except Exception as e:
        print(f"Warning: failed to create type×label bar chart: {e}")

    # === Preprocessing ===
    print("\n=== Preprocessing ===")
    df_sel = raw.filter(F.col("type").isin("TRANSFER", "CASH_OUT"))
    df_enc = df_sel.withColumn("type_encoded", F.when(F.col("type") == "TRANSFER", 1).otherwise(0))
    df_log = df_enc
    for c in SKEWED_COLS:
        df_log = df_log.withColumn(f"{c}_log1p", F.log1p(F.col(c)))
    processed = df_log.select(FEATURES_FINAL + [LABEL_COL])
    print("Preprocessing complete.")

    # === 2.3.2 EDA Viz: amount (raw) vs log1p(amount) (hist) ===
    print("\n=== EDA Viz: amount vs log1p(amount) (hist) ===")
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sample_frac = 0.05
        max_rows = 300_000

        hist_df = (df_log
                   .select(F.col("amount"), F.col("amount_log1p"))
                   .sample(withReplacement=False, fraction=sample_frac, seed=SEED)
                   .na.drop())
        hist_df = hist_df.limit(max_rows)
        pdf_hist = hist_df.toPandas()

        # Clip raw amount at 99th percentile for readability
        p99 = float(np.quantile(pdf_hist["amount"].values, 0.99)) if len(pdf_hist) else 0.0
        amt_clip = np.clip(pdf_hist["amount"].values, 0, p99) if len(pdf_hist) else []
        amt_log  = pdf_hist["amount_log1p"].values if len(pdf_hist) else []

        fig = plt.figure(figsize=(9.5, 4.5), dpi=160)
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.hist(amt_clip, bins=60)
        ax1.set_title(f"amount (clipped at 99th={p99:.0f})")
        ax1.set_xlabel("amount (raw)")
        ax1.set_ylabel("frequency")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(amt_log, bins=60)
        ax2.set_title("log1p(amount)")
        ax2.set_xlabel("log1p(amount)")
        ax2.set_ylabel("frequency")

        fig.suptitle("Distribution: amount (raw) vs log1p(amount)")
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        figpath2 = OUT_DIR / "fig_2_3b_amount_vs_log1p_hist.png"
        fig.savefig(figpath2)
        plt.close(fig)
        print(f"Saved: {figpath2}")
    except Exception as e:
        print(f"Warning: failed to create amount vs log1p histograms: {e}")

    # === Split (70/15/15) ===
    print("\n=== Data Splits ===")
    train, val, test = processed.randomSplit(SPLITS, seed=SEED)
    for name, ds in [("train", train), ("validation", val), ("test", test)]:
        cnt = ds.count(); pos = ds.filter(F.col(LABEL_COL) == 1).count()
        print(f"{name:10s}: {cnt:>9} rows | positives: {pos} ({pos/cnt:.1%})")
        save_csv([[name, cnt, pos, (pos/cnt if cnt else 0.0)]],
                 ["split","rows","positives","pos_ratio"],
                 OUT_DIR / f"class_counts_{name}.csv")
    train.cache(); val.cache(); test.cache()

    # === Imbalance handling ===
    n_pos = train.filter(F.col(LABEL_COL) == 1).count()
    n_neg = train.filter(F.col(LABEL_COL) == 0).count()
    balancing_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0
    print(f"\nBalancing ratio (weight for positive class): {balancing_ratio:.2f}")
    train_w = train.withColumn("weight", F.when(F.col(LABEL_COL) == 1, balancing_ratio).otherwise(1.0))

    # === Pipeline ===
    assembler = VectorAssembler(inputCols=FEATURES_FINAL, outputCol="unscaled_features")
    scaler    = StandardScaler(inputCol="unscaled_features", outputCol="features", withMean=False, withStd=True)
    lr        = LogisticRegression(featuresCol="features", labelCol=LABEL_COL, weightCol="weight",
                                   regParam=0.1, maxIter=100)
    pipe = Pipeline(stages=[assembler, scaler, lr])

    # === Train ===
    print("\n=== Training Logistic Regression ===")
    t0 = datetime.now()
    model = pipe.fit(train_w)
    print(f"Training completed in {datetime.now() - t0}.")

    # === PAUSE FOR SPARK UI SCREENSHOT ===
    print("\n\n=== Training completed ===")
    print("Spark UI available at: http://localhost:4040")
    print("Taking screenshot of Spark UI for report...")
    input("Press Enter when ready to continue...")

    # Save model & LR coefficients
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"bdas_lr_pipeline_{ts}"
    model.save(str(model_path))
    print(f"Model saved to: {model_path}")
    lr_coefficients_to_csv(model, FEATURES_FINAL, OUT_DIR / "lr_coefficients.csv")
    print("LR coefficients saved.")

    # === Threshold tuning & final evaluation ===
    tau_star = find_optimal_threshold(model, val, COST_FP, COST_FN)
    evaluate_on_test_set(model, test, tau_star)

    # === Distributed evidence ===
    dist_info = {
        "spark_master": spark.sparkContext.master,
        "default_parallelism": spark.sparkContext.defaultParallelism,
        "train_partitions": train.rdd.getNumPartitions(),
        "timestamp": datetime.now().isoformat(timespec="seconds")
    }
    save_json(dist_info, OUT_DIR / "dist_evidence.json")
    with open(OUT_DIR / "dist_evidence.txt", "w", encoding="utf-8") as f:
        f.write("=== BDAS Distributed Evidence ===\n")
        f.write(f"Spark Master URL:    {dist_info['spark_master']}\n")
        f.write(f"Default Parallelism: {dist_info['default_parallelism']}\n")
        f.write(f"Train Partitions:    {dist_info['train_partitions']}\n")
        f.write("\nOpen http://localhost:4040 and screenshot Jobs/Stages while training.\n")
    print("\nDistributed evidence saved.")

    # === Stop Spark ===
    spark.stop()
    print("\nSparkSession stopped. All artifacts have been saved under ./outputs and ./models.")