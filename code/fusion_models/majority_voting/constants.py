import os

BASE_DIR = os.getcwd()+"/../../../"

FINGER_FEATURES_FILE = os.path.join(BASE_DIR,"data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv")
AUDIO_FEATURES_FILE = os.path.join(BASE_DIR,"data/quick_brown_fox/wavlm_fox_features.csv")
FACIAL_FEATURES_FILE = os.path.join(BASE_DIR, "data/facial_expression_smile/facial_dataset.csv")

MODEL_BASE_PATH = os.path.join(BASE_DIR,"models")

FACIAL_EXPRESSIONS = {
    'smile': True,
    'surprise': False,
    'disgust': False
}

MODEL_SUBSETS = {
    0: ['finger_model_both_hand_fusion_no_baal', 'fox_model_best_auroc', 'facial_expression_smile_best_auroc'],
    1: ['finger_model_both_hand_fusion_no_baal', 'fox_model_best_auroc'],
    2: ['finger_model_both_hand_fusion_no_baal', 'facial_expression_smile_best_auroc'],
    3: ['fox_model_best_auroc', 'facial_expression_smile_best_auroc']
}