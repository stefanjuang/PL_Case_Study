import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRanker
from xgboost import XGBRanker
from sklearn.metrics import ndcg_score
import joblib
import os
import time

def train_and_evaluate_rankers(df, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    df_ltr = df.copy()

    if df_ltr['brandId'].dtype == 'object':
        df_ltr['brandId'] = LabelEncoder().fit_transform(df_ltr['brandId'])

    drop_cols = ['bookingLabel', 'ranking_label', 'searchId']
    feature_cols = [col for col in df_ltr.columns if col not in drop_cols]
    X = df_ltr[feature_cols]
    y = df_ltr['ranking_label']

    search_ids = df_ltr['searchId'].unique()
    train_search_ids, test_search_ids = train_test_split(search_ids, test_size=0.2, random_state=42)
    train_idx = df_ltr['searchId'].isin(train_search_ids)
    test_idx = df_ltr['searchId'].isin(test_search_ids)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_group = df_ltr[train_idx].groupby('searchId').size().values
    test_group = df_ltr[test_idx].groupby('searchId').size().values

    models = {
        'LightGBM Ranker': LGBMRanker(objective='lambdarank', metric='ndcg', boosting_type='gbdt', random_state=42),
        'LightGBM DART Ranker': LGBMRanker(objective='lambdarank', metric='ndcg', boosting_type='dart', random_state=42),
        'XGBoost Ranker': XGBRanker(objective='rank:pairwise', booster='gbtree', random_state=42),
        'XGBoost DART Ranker': XGBRanker(objective='rank:pairwise', booster='dart', random_state=42),
        'LambdaMART (LightGBM)': LGBMRanker(objective='lambdarank', metric='ndcg', boosting_type='gbdt', random_state=42)
    }

    results = []
    latencies = {}
    rank_comparison_df = None

    for name, model in models.items():
        model_filename = os.path.join(model_dir, f"trained_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl")

        if os.path.exists(model_filename):
            print(f"\n Loading pre-trained model for {name} from {model_filename}")
            model = joblib.load(model_filename)
        else:
            print(f"\n Training {name}...")
            model.fit(X_train, y_train, group=train_group)
            joblib.dump(model, model_filename)
            print(f" Saved trained model to {model_filename}")

        start = time.time()
        preds = model.predict(X_test)
        end = time.time()
        latency_ms = ((end - start) / len(X_test)) * 1000
        latencies[name] = latency_ms

        test_data = df_ltr[test_idx][['searchId', 'rank']].copy()
        test_data['true_label'] = y_test.values
        test_data['predicted_score'] = preds
        test_data['predicted_rank'] = test_data.groupby('searchId')['predicted_score'].rank(ascending=False, method='first')

        true_relevance, predicted_scores = [], []
        max_len = test_data.groupby('searchId').size().max()

        for sid in test_data['searchId'].unique():
            g = test_data[test_data['searchId'] == sid]
            true = np.pad(g['true_label'].values, (0, max_len - len(g)), 'constant')
            pred = np.pad(g['predicted_score'].values, (0, max_len - len(g)), 'constant')
            true_relevance.append(true)
            predicted_scores.append(pred)

        ndcg = ndcg_score(true_relevance, predicted_scores)
        print(f" {name} - NDCG Score: {ndcg:.4f}")
        results.append((name, ndcg))

        if name == 'LightGBM Ranker':
            rank_comparison_df = test_data.copy()
            rank_comparison_df['rank_diff'] = rank_comparison_df['rank'] - rank_comparison_df['predicted_rank']

    # Add rank feature baseline score
    print("\n Evaluating Rank Feature Baseline...")
    rank_feature_ndcg = compute_ndcg_from_rank_feature(df_ltr[test_idx], group_col="searchId", rank_col="rank", label_col="ranking_label")
    print(f" Rank Feature Baseline - NDCG Score: {rank_feature_ndcg:.4f}")
    
    results_df = pd.DataFrame(results, columns=['Model', 'NDCG_Score'])

    results_df = pd.concat([
        results_df,
        pd.DataFrame([{"Model": "Rank_Feature_Baseline", "NDCG_Score": rank_feature_ndcg}])
    ], ignore_index=True)

    best_model_name = results_df.sort_values(by="NDCG_Score", ascending=False).iloc[0]['Model']
    best_model_file = os.path.join(model_dir, f"trained_{best_model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.pkl")
    best_model = joblib.load(best_model_file) if "Rank_Feature_Baseline" not in best_model_name else None

    return results_df, best_model, latencies, rank_comparison_df

def compute_ndcg_from_rank_feature(df, group_col, rank_col, label_col):
    ndcg_scores = []
    for group_id, group_df in df.groupby(group_col):
        # Sort by rank ascending = better ranked
        group_df_sorted = group_df.sort_values(by=rank_col, ascending=True)
        
        # Reconstruct y_true and y_score arrays
        true_relevance = group_df_sorted[label_col].values
        predicted_rank_score = -group_df_sorted[rank_col].rank(method="first").values  # invert rank
        
        # Compute NDCG
        score = ndcg_score([true_relevance], [predicted_rank_score])
        ndcg_scores.append(score)
    
    return sum(ndcg_scores) / len(ndcg_scores)
