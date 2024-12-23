import os

#current dir: /localdisk1/PARK/park_multitask_fusion/code/unimodal_models/finger_tapping/
BASE_DIR = os.getcwd()+"/../../../"
#base dir: /localdisk1/PARK/park_multitask_fusion/

FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")

MODEL_TAG = "both_hand_fusion_baal"

MODEL_PATH = os.path.join(BASE_DIR,f"models/finger_model_{MODEL_TAG}","predictive_model/model.pth")
SCALER_PATH = os.path.join(BASE_DIR, f"models/finger_model_{MODEL_TAG}","scaler/scaler.pth")
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, f"models/finger_model_{MODEL_TAG}","predictive_model/model_config.json")
MODEL_BASE_PATH = os.path.join(BASE_DIR,f"models/finger_model_{MODEL_TAG}")