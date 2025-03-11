import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

def visualize_ndcg_and_feature_importance(
    results_df,
    best_model,
    model_latencies=None,
    rank_comparison_df=None,
    output_dir="visuals_and_results"
):
    os.makedirs(output_dir, exist_ok=True)

    # Save NDCG score comparison CSV
    results_df.to_csv(os.path.join(output_dir, "model_comparison_ndcg_scores.csv"), index=False)

    # Horizontal Bar Chart of Model Comparison
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    barplot = sns.barplot(
        data=results_df.sort_values(by='NDCG_Score', ascending=True),
        x="NDCG_Score",
        y="Model",
        palette="viridis",
        edgecolor="black"
    )
    plt.title('Model NDCG Score Comparison', fontsize=16)
    plt.xlabel('NDCG Score', fontsize=12)
    plt.ylabel('Model', fontsize=12)

    for p in barplot.patches:
        barplot.annotate(f"{p.get_width():.4f}", (p.get_width() + 0.001, p.get_y() + p.get_height() / 2),
                         va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ndcg_score_comparison.png"))
    plt.show()

    # Feature Importance from LTR Model
    feature_df = None

    if hasattr(best_model, 'feature_importances_') and hasattr(best_model, 'feature_name_'):
        # LightGBM case
        feature_df = pd.DataFrame({
            'Feature': best_model.feature_name_,
            'Importance': best_model.feature_importances_
        })
    elif hasattr(best_model, 'get_booster'):
        # XGBoost case
        booster = best_model.get_booster()
        importance_dict = booster.get_score(importance_type='weight')
        feature_df = pd.DataFrame(list(importance_dict.items()), columns=['Feature', 'Importance'])

    if feature_df is not None:
        feature_df.sort_values(by='Importance', ascending=False, inplace=True)
        feature_df.to_csv(os.path.join(output_dir, "feature_importance_best_model.csv"), index=False)

        plt.figure(figsize=(10, 7))
        sns.barplot(data=feature_df.head(20), y='Feature', x='Importance', palette='magma')
        plt.title("Top 20 Feature Importances (LTR Model)", fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_importance_best_model.png"))
        plt.show()

        print("\n🔍 Top 10 Most Important Features (LTR Model):")
        print(feature_df.head(10))
    else:
        print("⚠ Feature importance extraction not supported for this model type.")

    # Inference Latency Plot
    if model_latencies:
        latency_df = pd.DataFrame.from_dict(model_latencies, orient='index', columns=['Latency (ms)'])
        latency_df = latency_df.sort_values(by='Latency (ms)', ascending=True)

        latency_df.to_csv(os.path.join(output_dir, "inference_latency_comparison.csv"))

        plt.figure(figsize=(10, 6))
        sns.barplot(data=latency_df, x='Latency (ms)', y=latency_df.index, palette="flare")
        plt.title("Model Inference Latency (Per Sample)", fontsize=16)
        plt.xlabel("Latency (ms)", fontsize=12)
        plt.ylabel("Model")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "inference_latency_comparison.png"))
        plt.show()
