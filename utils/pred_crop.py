# utils/pred_crop.py

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import shap

# ─── 1) Load encoder & determine num_classes ─────────────────────────────────
enc_path = os.path.join("model","pkl_files","encoder.pkl")
with open(enc_path, "rb") as f:
    label_encoder = pickle.load(f)
CROP_LIST   = list(label_encoder.classes_)
NUM_CLASSES = len(CROP_LIST)

# ─── 2) Load your trained model ──────────────────────────────────────────────
from model.net import CropRecommendationNet

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join("model","baseline","baseline.hdf5")

model = CropRecommendationNet(input_dim=7,
                              output_dim=NUM_CLASSES,
                              dropout_p=0.2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ─── 3) MC-Dropout prediction ─────────────────────────────────────────────────
def predict_with_uncertainty(features, n_samples=50):
    """
    Returns (mean_probs, std_probs), each shape=(NUM_CLASSES,)
    """
    x_np = np.asarray(features, dtype=np.float32).reshape(1, -1)
    x_t  = torch.from_numpy(x_np).to(device)

    model.train()
    out = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(x_t)             # (1, NUM_CLASSES)
            p      = F.softmax(logits, dim=1)
            out.append(p.cpu().numpy()[0])
    model.eval()

    arr = np.stack(out, axis=0)          # (n_samples, NUM_CLASSES)
    return arr.mean(axis=0), arr.std(axis=0)

# ─── 4) SHAP explanation for *predicted class only* ──────────────────────────
def shap_single_explanation(background_data, features, pred_idx):
    """
    background_data: np.array (M,7)
    features:        list or np.array length-7
    pred_idx:        int, which class index to explain

    Returns:
      contrib: np.array shape=(7,)
    """
    bg = np.asarray(background_data, dtype=np.float32)
    sf = np.asarray(features,          dtype=np.float32).reshape(1, -1)

    # function that returns only the predicted-class probability
    def f_single(x_numpy):
        x_t = torch.from_numpy(x_numpy.astype(np.float32)).to(device)
        with torch.no_grad():
            logits = model(x_t)
            p = F.softmax(logits, dim=1)
        return p[:, pred_idx].cpu().numpy()  # shape (batch,)

    explainer  = shap.KernelExplainer(f_single, bg)
    # nsamples can be tuned lower/higher
    shap_vals  = explainer.shap_values(sf, nsamples=100)  # returns array (1,7)
    return np.asarray(shap_vals[0], dtype=float)

# ─── 5) Main entry point ──────────────────────────────────────────────────────
def recommend_crop(features, background_data=None):
    """
    features:        list/array of length 7
    background_data: optional M×7 array for SHAP; if None, load first 100 rows

    Returns:
      crop_name (str),
      mean_probs (np.array, NUM_CLASSES),
      std_probs  (np.array, NUM_CLASSES),
      contrib    (np.array, length-7)
    """
    # load default background if not provided
    if background_data is None:
        df = pd.read_csv(os.path.join("data","Crop_recommendation.csv"))
        background_data = df[
            ['N','P','K','temperature','humidity','ph','rainfall']
        ].values[:100]

    # 1) get mean & std of softmax via MC-Dropout
    mean_p, std_p     = predict_with_uncertainty(features, n_samples=50)

    # 2) pick predicted class
    pred_idx          = int(np.argmax(mean_p))
    crop_name         = label_encoder.inverse_transform([pred_idx])[0]

    # 3) get SHAP for that class only
    contrib           = shap_single_explanation(background_data,
                                               features,
                                               pred_idx)

    return crop_name, mean_p, std_p, contrib
