import os

BASE_PATH = os.getcwd()+"/../../../"
BASE_DIR = BASE_PATH

CLASSICAL_FEATURES_FILE = os.path.join(BASE_PATH,"data/quick_brown_fox/classical_fox_features.csv")
IMAGEBIND_FEATURES_FILE = os.path.join(BASE_PATH,"data/quick_brown_fox/imagebind_fox_features.csv")
WAV2VEC_FEATURES_FILE = os.path.join(BASE_PATH,"data/quick_brown_fox/wav2vec_fox_features.csv")
WAVLM_FEATURES_FILE = os.path.join(BASE_PATH,"data/quick_brown_fox/wavlm_fox_features.csv")

MODEL_TAG = "best_auroc"
MODEL_BASE_PATH = os.path.join(BASE_PATH, f"models/fox_model_{MODEL_TAG}")
MODEL_PATH = os.path.join(MODEL_BASE_PATH,"predictive_model/model.pth")
SCALER_PATH = os.path.join(MODEL_BASE_PATH,"scaler/scaler.pth")
MODEL_CONFIG_PATH = os.path.join(MODEL_BASE_PATH,"predictive_model/model_config.json")