"""
GMM Training for HRI Trajectory Data

Train Gaussian Mixture Models for trajectory classification based on Final Target.
Uses data from Experiment_Train_Trajectory_HRI organized by Target folders.
"""

import os
import json
import pickle
import time
import re
import numpy as np
import pandas as pd
import warnings
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# Tắt cảnh báo
warnings.filterwarnings("ignore")

# ─── CẤU HÌNH (CONFIGURATION) ───────────────────────────────────────────
DATA_DIR        = "Experiment_Train_Trajectory_HRI"
TRAIN_LIST_FILE = "train_file_list.json"
VAL_LIST_FILE   = "val_file_list.json"
CHECKPOINT_DIR  = "gmm_model_hri"

# Tham số mô hình GMM
K_MIN = 10
K_MAX = 20

# ─── HELPERS ────────────────────────────────────────────────────────────
def read_and_smooth(filepath, sigma=1):
    """
    Read trajectory CSV and apply Gaussian smoothing.
    HRI data has columns: Timestamp, X, Y, Z, Joint, ScenarioId
    """
    try:
        df = pd.read_csv(filepath)
        # Check for HRI format (uppercase X, Y, Z)
        if {'X', 'Y', 'Z'}.issubset(df.columns):
            df = df.dropna()
            df["X"] = gaussian_filter1d(df["X"].values, sigma=sigma)
            df["Y"] = gaussian_filter1d(df["Y"].values, sigma=sigma)
            df["Z"] = gaussian_filter1d(df["Z"].values, sigma=sigma)
            return df[['X', 'Y', 'Z']].values  # 3D trajectory
        # Fallback for lowercase x, y format
        elif {'x', 'y'}.issubset(df.columns):
            df = df.dropna()
            df["x"] = gaussian_filter1d(df["x"].values, sigma=sigma)
            df["y"] = gaussian_filter1d(df["y"].values, sigma=sigma)
            return df[['x', 'y']].values  # 2D trajectory
        return None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def get_target_id(folder_name):
    """Extract target ID from folder name (e.g., 'Target_1' -> 1)"""
    nums = re.findall(r'\d+', folder_name)
    return int(nums[0]) if nums else None


# ─── 1. LOAD DATA ───────────────────────────────────────────────────────
def load_dataset(root_dir):
    print(f"\n--- 1. LOADING DATA FROM {root_dir} ---")
    train_dict, val_dict = {}, {}
    
    if not os.path.exists(root_dir):
        print(f"❌ Error: Directory {root_dir} not found.")
        return {}, {}

    folders = sorted(os.listdir(root_dir))
    for folder in folders:
        path = os.path.join(root_dir, folder)
        if not os.path.isdir(path): 
            continue
        tid = get_target_id(folder)
        if tid is None: 
            continue
        
        print(f"  Loading Target {tid} from: {folder}")
        
        # Load Train
        f_train = os.path.join(path, TRAIN_LIST_FILE)
        if os.path.exists(f_train):
            train_dict[tid] = []
            with open(f_train, 'r') as f:
                file_list = json.load(f)
                for name in file_list:
                    d = read_and_smooth(os.path.join(path, name), sigma=1.0)
                    if d is not None: 
                        train_dict[tid].append(d)
            print(f"    Train: {len(train_dict[tid])} trajectories loaded")

        # Load Val
        f_val = os.path.join(path, VAL_LIST_FILE)
        if os.path.exists(f_val):
            val_dict[tid] = []
            with open(f_val, 'r') as f:
                file_list = json.load(f)
                for name in file_list:
                    d = read_and_smooth(os.path.join(path, name), sigma=1.0)
                    if d is not None: 
                        val_dict[tid].append(d)
            print(f"    Val:   {len(val_dict[tid])} trajectories loaded")
    
    print(f"\n✓ Loaded {len(train_dict)} targets total.")
    return train_dict, val_dict


# ─── 2. NORMALIZE DATA ──────────────────────────────────────────────────
def normalize_data(train_dict, val_dict):
    print(f"\n--- 2. NORMALIZATION (StandardScaler) ---")
    all_points = []
    for trajs in train_dict.values():
        for t in trajs: 
            all_points.append(t)
        
    if not all_points: 
        return {}, {}, None
    
    X_global = np.vstack(all_points)
    scaler = StandardScaler()
    scaler.fit(X_global)
    
    print(f"✓ Scaler fitted on {X_global.shape[0]} points ({X_global.shape[1]}D)")

    norm_train = {t: [scaler.transform(x) for x in v] for t, v in train_dict.items()}
    norm_val   = {t: [scaler.transform(x) for x in v] for t, v in val_dict.items()}
    
    return norm_train, norm_val, scaler


# ─── 3. TRAINING (BIC SELECTION) ────────────────────────────────────────
def train_system(train_dict, val_dict):
    print(f"\n--- 3. TRAINING GMM (K from {K_MIN} to {K_MAX}, BIC minimization) ---")
    start_time = time.time()
    
    best_models = {}
    training_log = []
    
    sorted_targets = sorted(train_dict.keys())
    
    for tid in sorted_targets:
        X_train = np.vstack(train_dict[tid])
        has_val = (tid in val_dict and len(val_dict[tid]) > 0)
        X_val = np.vstack(val_dict[tid]) if has_val else None
        
        print(f"Target {tid:>2}: Training with {X_train.shape[0]} points...", end=" ")
        
        # Biến lưu kết quả tốt nhất (BIC càng thấp càng tốt)
        best_bic_score = np.inf 
        best_k = -1
        best_gmm = None
        
        for k in range(K_MIN, K_MAX + 1):
            try:
                gmm = GaussianMixture(
                    n_components=k, 
                    covariance_type='full', 
                    random_state=42, 
                    n_init=3, 
                    max_iter=100
                )
                gmm.fit(X_train)
                
                # Tính BIC trên tập Validation (nếu có) hoặc Train
                if has_val:
                    score = gmm.bic(X_val)
                else:
                    score = gmm.bic(X_train)
                
                # Chọn BIC nhỏ nhất
                if score < best_bic_score:
                    best_bic_score = score
                    best_k = k
                    best_gmm = gmm
                    
            except Exception as e:
                continue
        
        best_models[tid] = best_gmm
        training_log.append({
            "Target": tid, 
            "Best_K": best_k, 
            "Best_BIC": best_bic_score,
            "N_Train": X_train.shape[0]
        })
        print(f"➜ Selected K={best_k:<2} | BIC={best_bic_score:.2f}")

    duration = time.time() - start_time
    print(f"\n✓ Training completed in {duration:.2f} seconds.")
    return best_models, training_log


# ─── 4. EVALUATION ──────────────────────────────────────────────────────
def evaluate_accuracy(models, val_dict):
    print(f"\n--- 4. FINAL EVALUATION (BIC-based Target Classification) ---")
    
    correct = 0
    total = 0
    
    for true_tid, trajs in val_dict.items():
        for traj in trajs:
            total += 1
            
            # So sánh BIC của mỗi model
            scores = {}
            for model_tid, model in models.items():
                if model is None:
                    continue
                # Tính Raw Log-Likelihood
                raw_ll = np.sum(model.score_samples(traj))
                
                # Tính BIC: -2*LL + k*log(n)
                k = model.n_components
                n = len(traj)
                bic_val = -2 * raw_ll + k * np.log(n)
                scores[model_tid] = -bic_val  # Lấy số đối để dùng max
            
            # Chọn model có -BIC lớn nhất (BIC nhỏ nhất)
            if scores:
                pred_tid = max(scores, key=scores.get)
                if pred_tid == true_tid:
                    correct += 1
    
    acc = (correct / total) * 100 if total > 0 else 0
    print(f"  Total Validation Samples: {total}")
    print(f"  Correct Predictions:      {correct}")
    print(f"  ✓ System Accuracy:        {acc:.2f}%")
    return acc


# ─── 5. SAVING ──────────────────────────────────────────────────────────
def save_checkpoints(models, scaler, logs, accuracy):
    print(f"\n--- 5. SAVING CHECKPOINTS ---")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    with open(os.path.join(CHECKPOINT_DIR, "gmm_models.pkl"), 'wb') as f:
        pickle.dump(models, f)
    with open(os.path.join(CHECKPOINT_DIR, "scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
        
    pd.DataFrame(logs).to_csv(os.path.join(CHECKPOINT_DIR, "training_log.csv"), index=False)
    
    with open(os.path.join(CHECKPOINT_DIR, "info.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Data: Experiment_Train_Trajectory_HRI\n")
        f.write(f"K Range: {K_MIN}-{K_MAX}\n")
        f.write("Note: Models trained using BIC minimization on 3D trajectories (X, Y, Z).\n")
        
    print(f"✓ Saved all checkpoints to: {CHECKPOINT_DIR}/")
    print(f"  - gmm_models.pkl")
    print(f"  - scaler.pkl")
    print(f"  - training_log.csv")
    print(f"  - info.txt")


# ─── MAIN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("GMM Training for HRI Trajectory Classification")
    print("=" * 60)
    
    raw_train, raw_val = load_dataset(DATA_DIR)
    
    if raw_train:
        X_train_s, X_val_s, scaler = normalize_data(raw_train, raw_val)
        
        # Train (BIC Selection)
        models, logs = train_system(X_train_s, X_val_s)
        
        # Evaluate (BIC Comparison)
        accuracy = 0
        if X_val_s:
            accuracy = evaluate_accuracy(models, X_val_s)
            
        save_checkpoints(models, scaler, logs, accuracy)
        
        print("\n" + "=" * 60)
        print("DONE!")
        print("=" * 60)
    else:
        print("❌ No data loaded. Please check data directory structure.")
