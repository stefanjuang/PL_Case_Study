# -------------------------------
# Priceline Hotel Booking Prediction
# -------------------------------

# main.py
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
from scripts.data_preprocessing import preprocess_data
from scripts.train_rankers import train_and_evaluate_rankers
from scripts.visualize_results import visualize_ndcg_and_feature_importance
import warnings
warnings.filterwarnings("ignore")


# -------------------------------
# MAIN EXECUTION PIPELINE
# -------------------------------

print("Loading raw dataset...")

# Adjust this path accordingly
unzip_dir = "./data"
parquet_file_path = os.path.join(unzip_dir, "case_study_dataset.parquet")

# Read Parquet file robustly with PyArrow
table = pq.read_table(parquet_file_path)
data_dict = {}

for field in table.schema:
    col_name = field.name
    col_type = field.type
    col_data = table.column(col_name).to_pylist()

    # Decode binary columns
    if pa.types.is_binary(col_type):
        decoded = [x.decode('utf-8', errors='ignore') if isinstance(x, bytes) else x for x in col_data]
        data_dict[col_name] = decoded

    # Handle dbdate/date32 columns
    elif pa.types.is_date(col_type):
        first = next((x for x in col_data if x is not None), None)
        if isinstance(first, int):
            data_dict[col_name] = pd.to_datetime(col_data, unit='D', origin='unix')
        else:
            data_dict[col_name] = pd.to_datetime(col_data)

    else:
        data_dict[col_name] = col_data

df_raw = pd.DataFrame(data_dict)
print("Dataset loaded successfully.")

# Preprocessing
print("Preprocessing...")
df_processed = preprocess_data(df_raw)

# Train & Evaluate LTR Models
print("Training Ranking Models...")
results_df, best_model, latencies, rank_comparison_df = train_and_evaluate_rankers(df_processed)


# Visualize Results
print("Generating Visualizations...")
visualize_ndcg_and_feature_importance(
    results_df=results_df,
    best_model=best_model,
    model_latencies=latencies,
    rank_comparison_df=rank_comparison_df
)

print("All steps completed.")

