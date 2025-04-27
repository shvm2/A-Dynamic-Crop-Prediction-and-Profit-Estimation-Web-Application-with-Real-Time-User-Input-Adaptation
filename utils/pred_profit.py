import os
import pickle
import numpy as np
import pandas as pd

# ─── 1) Load your LabelEncoder to get the list of crops ─────────────────────
enc_path = os.path.join('model', 'pkl_files', 'encoder.pkl')
with open(enc_path, 'rb') as f:
    label_encoder = pickle.load(f)
CROP_LIST = list(label_encoder.classes_)  # e.g. ['rice','jute','blackgram',...]

# ─── 2) Explicit mapping from internal crop names to CSV commodity strings ─
MANUAL_MAP = {
    "blackgram":   "Black Gram (Urd Beans)(Whole)",
    "lentil":      "Lentil (Masur)(Whole)",
    "chickpea":    "Kabuli Chana(Chickpeas-White)",
    "coconut":     "Coconut",
    "coffee":      "Coffee",
    "cotton":      "Cotton",
    "jute":        "Jute",
    "maize":       "Maize",
    "mango":       "Mango",
    "orange":      "Orange",
    "pomegranate": "Pomegranate",
    "apple":       "Apple",
    "rice":        "Rice",
    "muskmelon":   "Karbuja(Musk Melon)",
    "papaya":      "Papaya",
    "grapes":      "Grapes",
    "banana":      "Banana",
    "watermelon":  "Watermelon",
    # add others as needed...
}

# ─── 3) Load only the two relevant CSV columns and clean ────────────────────
csv_path = os.path.join(
    'data',
    'Current_Daily_Price_of_Various_Commodities_from_Various_Markets.csv'
)
df = pd.read_csv(csv_path, usecols=['Commodity','Modal_x0020_Price'])
df = df.rename(columns={'Modal_x0020_Price': 'Modal_Price'})
df['Commodity'] = df['Commodity'].str.strip()
df['Modal_Price'] = pd.to_numeric(df['Modal_Price'], errors='coerce')
df = df.dropna(subset=['Modal_Price'])

# ─── 4) Compute per-commodity mean & std across all states ────────────────
_stats    = df.groupby('Commodity')['Modal_Price'].agg(['mean','std']).reset_index()
MEAN_MAP  = dict(zip(_stats['Commodity'], _stats['mean']))
STD_MAP   = dict(zip(_stats['Commodity'], _stats['std']))

# ─── 5) Overall fallback mean & std for unmapped crops ────────────────────
ALL_MEAN = float(df['Modal_Price'].mean())
ALL_STD  = float(df['Modal_Price'].std())

# ─── 6) (Optional) per-crop yield assumptions (tonnes/ha) ────────────────
YIELD_MAP = {crop: 1.0 for crop in CROP_LIST}


def get_profit_recommendation(
    pred_probs: np.ndarray,
    pred_uncerts: np.ndarray,
    n_samples: int = 500,
    risk_aversion: float = 1.0
):
    """
    Monte-Carlo profit sampling:
      - For each crop:
        * Look up mean/std price by MANUAL_MAP → MEAN_MAP/STD_MAP
        * If not found, use ALL_MEAN/ALL_STD
      - Sample predicted probability ~ N(pred, uncert)
      - Sample price ~ N(price_mu, price_sigma)
      - Profit = p * price * yield
      - Score = mean(profit) - λ*std(profit)
    Returns:
      profit_means: dict[crop->float]
      profit_stds:  dict[crop->float]
      best_crop:    str
    """
    profit_means = {}
    profit_stds  = {}
    scores       = {}

    for i, crop in enumerate(CROP_LIST):
        key = MANUAL_MAP.get(crop.lower())
        if key in MEAN_MAP:
            mu    = float(MEAN_MAP[key])
            sigma = float(STD_MAP.get(key, np.nan)) or mu * 0.05
        else:
            mu, sigma = ALL_MEAN, ALL_STD

        # 1) Sample model probability
        p_samps = np.random.normal(pred_probs[i], pred_uncerts[i], n_samples)
        p_samps = np.clip(p_samps, 0.0, 1.0)

        # 2) Sample market price
        price_samps = np.random.normal(mu, sigma, n_samples)
        price_samps = np.clip(price_samps, 0.0, None)

        # 3) Compute profit
        yld     = YIELD_MAP.get(crop, 1.0)
        profits = p_samps * price_samps * yld

        mean_p = float(np.mean(profits))
        std_p  = float(np.std(profits))

        profit_means[crop] = mean_p
        profit_stds[crop]  = std_p
        scores[crop]       = mean_p - risk_aversion * std_p

    best_crop = max(scores, key=scores.get)
    return profit_means, profit_stds, best_crop


def get_price_stats():
    """
    Returns:
      price_means: dict[crop->float]
      price_stds:  dict[crop->float]
    based purely on the government CSV (aggregated over all states).
    """
    price_means = {}
    price_stds  = {}

    for crop in CROP_LIST:
        key = MANUAL_MAP.get(crop.lower())
        if key in MEAN_MAP:
            mu    = float(MEAN_MAP[key])
            sigma = float(STD_MAP.get(key, np.nan)) or mu * 0.05
        else:
            mu, sigma = ALL_MEAN, ALL_STD

        price_means[crop] = mu
        price_stds[crop]  = sigma

    return price_means, price_stds
