train_defog_config = {
    "BATCH_SIZE": 64,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "NUM_EPOCHS": 50,
    "train_valid_split": 0.8,
    "data": r"C:\Users\timot\OneDrive\Desktop\EMORY\Spring 2024\Research\code\fog_project\data\segmented_defog_data_no_overlap_normalized.csv",
    "num_sensor_channels": 3,
    "class_weights": [0.85, 0.15],
    "k": 5,
    "model_name": f"defog_cnn",
}


train_finetune_defog_config = {
    "pretrained_model": r"saved_models/cnn_defog_ckp_last.pt",
    "experiment_log_dir": r"./",
    "arch": "daily2defog",
    "training_mode": "fine_tune_test",
    "num_epochs": 100,
    "class_weights": [0.85, 0.15],
    "tfc_type": "cnn",
}
