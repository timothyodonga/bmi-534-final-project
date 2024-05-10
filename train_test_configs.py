train_defog_config = {
    "BATCH_SIZE": 64,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "train_valid_split": 0.8,
    "num_sensor_channels": 3,
    "class_weights": [0.85, 0.15],
    "k": 5,
}


train_finetune_defog_config = {
    "experiment_log_dir": r"./",
    "arch": "daily2defog",
    "training_mode": "fine_tune_test",
    "class_weights": [0.85, 0.15],
}

train_harth_config = {
    "BATCH_SIZE": 64,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    # "NUM_EPOCHS": 10,
    "train_valid_split": 0.8,
    # "data": "harth_preprocessed_data_206_window.csv",
    "num_sensor_channels": 6,
    "class_weights": [
        0.15076889696509452,
        0.01749342301537794,
        0.04228255268368094,
        0.011716525182392667,
        0.009465433538553334,
        0.12665780694855036,
        0.5197851970383228,
        0.07057036695505953,
        0.04228255268368094,
        0.004041116324482656,
        0.0047191559762414905,
        0.00021697268856282715,
    ],
    "k": 5,
    # "model_name": f"mlp",
}

train_finetune_harth_config = config = {
    "experiment_log_dir": r"./",
    "arch": "daily2harth",
    "training_mode": "fine_tune_test",
    "class_weights": [
        0.15076889696509452,
        0.01749342301537794,
        0.04228255268368094,
        0.011716525182392667,
        0.009465433538553334,
        0.12665780694855036,
        0.5197851970383228,
        0.07057036695505953,
        0.04228255268368094,
        0.004041116324482656,
        0.0047191559762414905,
        0.00021697268856282715,
    ],
}

config_pretrain_daily_harth = {
    "subset": True,
    "batch_size": 16,
    "training_mode": "pre_train",
    "experiment_log_dir": r"./",
}
