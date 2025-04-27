# main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np                                      # for nan_to_num
from fastapi.responses import JSONResponse

from utils.pred_crop     import recommend_crop
from utils.pred_temp_hum import get_temp_hum
from utils.pred_rainfall import get_rainfall
from utils.pred_profit   import get_profit_recommendation

app = FastAPI()
# CORS must come first so all responses carry the header
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

class FeatureInput(BaseModel):
    nitrogen:    float
    phosphorous: float
    potassium:   float
    ph:          float
    state:       str
    district:    str
    month:       str

@app.post("/predict")
def predict_crop(inp: FeatureInput):
    # 1) Build feature vector
    temp, hum = get_temp_hum(inp.district, inp.state, inp.month)
    rain      = get_rainfall(inp.state, inp.district, inp.month)
    feats     = [
        inp.nitrogen, inp.phosphorous, inp.potassium,
        temp, hum, inp.ph, rain
    ]

    # 2) Core model + SHAP
    crop, raw_probs, raw_uncert, contrib = recommend_crop(feats)

    # 3) Economic profit layer
    pm, ps, best = get_profit_recommendation(
        pred_probs=raw_probs,
        pred_uncerts=raw_uncert,
        n_samples=500,
        risk_aversion=0.5
    )

    # 4) Sanitize all floats to be JSON-compliant
    probs   = np.nan_to_num(raw_probs,   nan=0.0, posinf=0.0, neginf=0.0).tolist()
    uncert  = np.nan_to_num(raw_uncert,  nan=0.0, posinf=0.0, neginf=0.0).tolist()
    contrib = np.nan_to_num(contrib,     nan=0.0, posinf=0.0, neginf=0.0).tolist()

    # Profit dicts: replace any NaN/Inf
    profit_means = {
        c: float(np.nan_to_num(pm[c],   nan=0.0, posinf=0.0, neginf=0.0))
        for c in pm
    }
    profit_stds  = {
        c: float(np.nan_to_num(ps[c],   nan=0.0, posinf=0.0, neginf=0.0))
        for c in ps
    }

    payload = {
      "predicted_crop":     crop,
      "probabilities":      probs,
      "uncertainty":        uncert,
      "shap_contributions": {
         "features": ['N','P','K','temperature','humidity','ph','rainfall'],
         "values":   contrib
      },
      "profit_means":       profit_means,
      "profit_stds":        profit_stds,
      "best_crop_profit":   best
    }

    # 5) Return via JSONResponse (now contains only finite floats)
    return JSONResponse(content=payload)
