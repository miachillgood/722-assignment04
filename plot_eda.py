# -*- coding: utf-8 -*-
"""
Plotting script for EDA visualization (Section 2.3).
This script reads the aggregated data produced by the main Spark job
and generates a high-quality bar chart for the report.

This demonstrates the "separation of concerns" principle:
Spark for heavy computation, Python/matplotlib for visualization.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# --- Configuration ---
# The main script saves the output in a folder with a part-*.csv file
# We need to find that CSV file to read it.
INPUT_DIR = Path("outputs/eda_type_by_label_counts")
OUTPUT_IMAGE_PATH = Path("outputs/plot_fraud_distribution_by_type.png")

# --- Main plotting logic ---
def main():
    print("Starting EDA plotting script...")
    
    # Find the part-*.csv file written by Spark
    try:
        # Use glob to find the file since the name is not fixed
        csv_files = glob.glob(str(INPUT_DIR / "part-*.csv"))
        if not csv_files:
            print(f"Error: No CSV file found in '{INPUT_DIR}'.")
            print("Please run the main Spark script first to generate the data.")
            return
        
        # Read the aggregated data into a pandas DataFrame
        source_csv = csv_files[0]
        df = pd.read_csv(source_csv)
        print(f"Successfully loaded data from '{source_csv}'")
        
    except Exception as e:
        print(f"Error reading the data file: {e}")
        return

    # --- Create the visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    
    # Create the bar plot, which is identical to the one from Iteration 3
    sns.barplot(
        data=df,
        x="type",
        y="count",
        hue="isFraud",
        palette={0: "skyblue", 1: "coral"}
    )
    
    # Use a logarithmic scale on the y-axis to make the small fraud counts visible
    plt.yscale('log')
    
    # Add titles and labels for clarity
    plt.title("Fraud Distribution by Transaction Type", fontsize=16, fontweight='bold')
    plt.ylabel("Transaction Count (Log Scale)", fontsize=12)
    plt.xlabel("Transaction Type", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Is Fraud?")
    plt.tight_layout()
    
    # Save the figure to the outputs directory
    OUTPUT_IMAGE_PATH.parent.mkdir(exist_ok=True)
    plt.savefig(OUTPUT_IMAGE_PATH, dpi=300)
    
    print(f"\nSuccess! Plot saved to: {OUTPUT_IMAGE_PATH}")

if __name__ == "__main__":
    main()
