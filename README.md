# UFNet
Official Github repository for the paper "Accessible, At-Home Detection of Parkinson’s Disease via Multi-Task Video Analysis"

**Data Description**

Note: We are unable to share the raw patient videos to protect their identity and privacy. In this repository, we share the extracted features for the three tasks we experimented in our paper. If you need the frame-by-frame hand key-points, face mesh (facial key-points extracted with MediaPipe), facial action units (extracted using OpenFace), or raw audio files for the ``quick brown fox'' utterance, please reach out to us at mehoque@cs.rochester.edu. Requests will be evaluated case-by-case and we will be able to share those detailed data if the purpose of data use aligns with our IRB protocol.

To access the extracted features, please go to ```/data``` folder in this repository. 
* [Metadata](data/all_file_user_metadata.csv) contains demographic information about the user, unique Participant_ID, and whether the subject has Parkinson's disease (pd column: yes indicates the participant has PD, no indicates Non-PD)
* [List of all participants IDs](data/all_task_ids.txt) contains the list of unique ids of all participants. They are later divided into [Validation](data/dev_set_participants.txt) and [Test](data/test_set_participants.txt) sets. All other participants are considered part of the training set (IDs are not explicitly listed).
* For conformal prediction variants, a part of the training set participants is reserved as [Calibration](data/calib_set_participants.txt) set. None of the Validation, Test, and Calibration should be used for model training where conformal prediction or Platt scaling is involved.
* We also include extracted features from YoutubePD [1] dataset videos. As in the original paper, [test participants](data/test_set_participants_yt_pd.txt) are separated, and not used for training the models.
* Task-specific features are also organized in this folder: [Finger-tapping](data/finger_tapping/features_demography_diagnosis_Nov22_2023.csv), [Smile](data/facial_expression_smile/facial_dataset.csv), 
and [Speech](data/quick_brown_fox/wavlm_fox_features.csv).

**Code Description**

Note: The feature extraction codes will be added to the repository soon. Currently, the repository contains code for training and running the [task-specific models](code/unimodal_models) and [UFNet](code/fusion_models/ufnet), the [baselines](code/fusion_models/baselines) we experimented, and the [ablation studies](code/fusion_models/ufnet/ablations). 
We also provide code for [demographic information analysis](code/demographic_details/demography_summary_table.py).

* To run the UFNet model, you first need to run and save the task-specific models. Please make sure the hyper-parameters are the same as we report in our paper (the arxiv paper has the detailed hyper-parameters for both task-specific models and the UFNet: https://arxiv.org/abs/2406.14856).
  * [Finger-tapping model](/code/unimodal_models/finger_tapping/unimodal_finger_baal.py)
  * [Smile model](code/unimodal_models/facial_expression_smile/unimodal_smile_baal.py)
  * [Speech model](code/unimodal_models/quick_brown_fox/unimodal_fox_baal.py)
 
* 

**References**
1. Zhou, Andy, et al. "YouTubePD: A Multimodal Benchmark for Parkinson’s Disease Analysis." Advances in Neural Information Processing Systems 36 (2024).
