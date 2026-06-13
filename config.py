import numpy as np
import os

GMM_CHECKPOINT_DIR  = "gmm_model_hri"
GMM_MODEL_PATH      = os.path.join(GMM_CHECKPOINT_DIR, "gmm_models.pkl")
GMM_SCALER_PATH     = os.path.join(GMM_CHECKPOINT_DIR, "scaler.pkl")

SVGP_CHECKPOINT_DIR = "svgp_hri_m52"

TEST_FOLDER         = "Experiment_Test_Trajectory_HRI"
TEST_FILE_LIST      = "test_file_list.json"

#2 GMM 
GMM_WINDOW_SIZE = 5
TAU_SOFTMAX     = 8
SIGMA           = 0.2 

#3 SVGP 
SVGP_HISTORY_SIZE = 10

#4 FITTS' LAW
FITTS_A = 4.3939
FITTS_B = 0.5224
FITTS_W = 0.08    

#5 OUTER LOOP 
MJM_METHOD = "CURRENT"

USE_GT_TIME_CHEAT = True
P_LOW              = 0.6 
P_HIGH             = 0.8 
P_HYSTERESIS       = 0.05

# Conflict Detection
PHI_ANGLE          = 90
FORCE_DEADZONE     = 0.05
N_CONFLICT_REQUIRED = 5    
N_CONFIDENCE_REQUIRED = 5
LEADER_COOLDOWN        = 10 
BLEND_STEPS            = 10

# Time-step
DT = 12 / 191

# INNER LOOP — Low-Level Robot Dynamics

# Robot parameters
M_ROB = 1.0    
C_V   = 0.75   
F_C   = 3

# PD Controller
K_P   = 1000000
K_D   = 2.0 * 0.7 * np.sqrt(K_P) 

# Admittance Control
M_ADM = 1.0     
K_ADM = 100.0   
D_ADM = 20.0    

# Human stiffness
K_H   = 150
K_H_REDUCE_FACTOR = 0.5
K_H_REDUCE_STEPS  = 6

# Sub-stepping
N_SUB = 100

GOALS = {
    1: np.array([0.3796,  0.9779, 0.0169]),
    2: np.array([0.0409,  1.3684, 0.0052]),
    3: np.array([-0.0784, 1.0867, 0.0256]),
}

# ScenarioId → Target ID
GROUND_TRUTH = {
    1: 1,  4: 1,  9: 1, 11: 1, 15: 1, 17: 1,
    2: 2,  5: 2,  7: 2, 12: 2, 13: 2, 18: 2,
    3: 3,  6: 3,  8: 3, 10: 3, 14: 3, 16: 3,
}

# ScenarioId → Descriptive Name
SCENARIO_NAMES = {
    1: "Free_Initial_1_Target_1",
    2: "Free_Initial_2_Target_2",
    3: "Free_Initial_3_Target_3",
    4: "Obstacle_Initial_1_Target_1",
    5: "Obstacle_Initial_2_Target_2",
    6: "Obstacle_Initial_3_Target_3",
    7: "Change_Initial_1_Target_2",
    8: "Change_Initial_1_Target_3",
    9: "Change_Initial_2_Target_1",
    10: "Change_Initial_2_Target_3",
    11: "Change_Initial_3_Target_1",
    12: "Change_Initial_3_Target_2",
    13: "Change+Obstacle_Initial_1_Target_2",
    14: "Change+Obstacle_Initial_1_Target_3",
    15: "Change+Obstacle_Initial_2_Target_1",
    16: "Change+Obstacle_Initial_2_Target_3",
    17: "Change+Obstacle_Initial_3_Target_1",
    18: "Change+Obstacle_Initial_3_Target_2"
}

SAVE_DIR_OVERRIDE = None

def get_save_dir(base_dir=None):
    if SAVE_DIR_OVERRIDE:
        name = SAVE_DIR_OVERRIDE
    else:
        cheat_suffix = "_GTCHEAT" if USE_GT_TIME_CHEAT else ""
        if MJM_METHOD == "PAPER":
            name = (f"MJM_PAPER_{LEADER_COOLDOWN}_{BLEND_STEPS}_"
                    f"{PHI_ANGLE}_{N_CONFLICT_REQUIRED}_{N_CONFIDENCE_REQUIRED}_"
                    f"{C_V}_{F_C}_{K_H}_{K_P}_{K_D:.2f}{cheat_suffix}")
        else:
            name = (f"MJM_CURRENT_{LEADER_COOLDOWN}_{BLEND_STEPS}_"
                    f"{PHI_ANGLE}_{N_CONFLICT_REQUIRED}_{N_CONFIDENCE_REQUIRED}_"
                    f"{C_V}_{F_C}_{K_H}_{K_P}_{K_D:.2f}{cheat_suffix}")
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, name)
