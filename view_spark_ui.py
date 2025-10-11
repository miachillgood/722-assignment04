# -*- coding: utf-8 -*-
"""
Spark UI Viewer Script
This script starts a Spark session and keeps it running so you can view the Spark UI.
"""

from pyspark.sql import SparkSession
import time

# Initialize Spark with UI enabled
spark = SparkSession.builder \
    .appName("FraudDetection_UI_Viewer") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.driver.host", "localhost") \
    .config("spark.driver.bindAddress", "localhost") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("🚀 SparkSession initialized successfully!")
print("📊 Spark UI is now available at: http://localhost:4040")
print("\n📋 What you can view in the Spark UI:")
print("   • Jobs tab - See job execution timeline")
print("   • Stages tab - See detailed stage execution")
print("   • Storage tab - See cached RDDs/DataFrames")
print("   • Environment tab - See Spark configuration")
print("   • Executors tab - See executor details")
print("\n⏰ Spark session will remain active for 10 minutes...")
print("   Press Ctrl+C to stop early if needed.")

try:
    # Keep the session alive for 10 minutes
    time.sleep(600)  # 600 seconds = 10 minutes
except KeyboardInterrupt:
    print("\n🛑 Stopping Spark session...")

spark.stop()
print("✅ SparkSession stopped. Spark UI is no longer available.")
