# %%
# Import statements
import os
import numpy as np
from datetime import datetime
from model import *
from dataloader import generate_freq, Load_Dataset, Load_DatasetTwo
from configs import Config
from trainer import model_pretrain, model_test

from loss import *
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from model import *
from tfc_finetune_routines import *
import sys
import pandas as pd

# %%
df_performance = pd.DataFrame()
folds = ["0", "1", "2", "3", "4"]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for fold in folds:
    print(f"Fold: {fold}")
    training_mode = "fine_tune_test"

    # arch = "daily2harth"
    # configs.num_epoch = 50

    config = {
        "fine_tune_train": f"C:/Users/timot/OneDrive/Desktop/EMORY/Spring 2024/BMI-534/project-code/code/bmi-534-final-project/processed_data/harth_train_fold_{fold}.pt",
        "fine_tune_test": f"C:/Users/timot/OneDrive/Desktop/EMORY/Spring 2024/BMI-534/project-code/code/bmi-534-final-project/processed_data/harth_test_fold_{fold}.pt",
        "pretrained_model": r"C:\Users\timot\OneDrive\Desktop\EMORY\Spring 2024\BMI-534\project-code\code\bmi-534-final-project\saved_models\ckp_last.pt",
        "experiment_log_dir": r"C:\Users\timot\OneDrive\Desktop\EMORY\Spring 2024\BMI-534\project-code\code\bmi-534-final-project",
        "model_type": "cnn_small",
        "arch": "daily2harth",
        "training_mode": "fine_tune_test",
        "num_epochs": 1,
        "model_name": "harth_mlp_finetuned",
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
        "saved_models_classifier": {
            "0": r"experiments_logs\finetunemodel\daily2harth_fold_0_classifier.pt",
            "1": r"experiments_logs\finetunemodel\daily2harth_fold_1_classifier.pt",
            "2": r"experiments_logs\finetunemodel\daily2harth_fold_2_classifier.pt",
            "3": r"experiments_logs\finetunemodel\daily2harth_fold_3_classifier.pt",
            "4": r"experiments_logs\finetunemodel\daily2harth_fold_4_classifier.pt",
        },
        "saved_models_tfc": {
            "0": r"experiments_logs/finetunemodel/daily2harth_fold_0_model.pt",
            "1": r"experiments_logs/finetunemodel/daily2harth_fold_1_model.pt",
            "2": r"experiments_logs/finetunemodel/daily2harth_fold_2_model.pt",
            "3": r"experiments_logs/finetunemodel/daily2harth_fold_3_model.pt",
            "4": r"experiments_logs/finetunemodel/daily2harth_fold_4_model.pt",
        },
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # %%
    finetune_train = torch.load(config["fine_tune_train"])
    finetune_test = torch.load(config["fine_tune_test"])

    # %%
    configs = Config()
    print(configs.__dict__)
    training_mode = "pre_train"
    subset = False

    # %%

    test_dataset = Load_Dataset(
        finetune_test,
        configs,
        training_mode,
        target_dataset_size=configs.target_batch_size,
        subset=subset,
    )

    # %%

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=configs.target_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # %%
    with_gpu = torch.cuda.is_available()
    if with_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("We are using %s now." % device)

    TFC_model = TFC(configs)
    print("Loading the pre-trained model")
    TFC_model.load_state_dict(torch.load(config["pretrained_model"]))
    TFC_model = TFC_model.to(device)
    print("****** Loaded the pretrained model")
    # %%
    if config["model_type"] == "cnn_small":
        classifier = BaselineCNNSmall(
            num_sensor_channels=2, num_output_classes=12, embed_length=120
        ).to(device)
    elif config["model_type"] == "cnn":
        classifier = BaselineCNN(
            num_sensor_channels=2, num_output_classes=12, embed_length=112
        ).to(device)
    else:
        classifier = MLPTFC(num_sensor_channels=2, num_output_classes=12).to(device)

    print("Printing the classifier")
    print(classifier)

    print(f"Printing the learning rate: {configs.lr}")

    # %%
    model_optimizer = torch.optim.Adam(
        TFC_model.parameters(),
        lr=configs.lr,
        betas=(configs.beta1, configs.beta2),
        weight_decay=3e-4,
    )
    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=configs.lr,
        betas=(configs.beta1, configs.beta2),
        weight_decay=3e-4,
    )

    # criterion = nn.CrossEntropyLoss()
    tfc_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, "min")
    classifier_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        classifier_optimizer, "min"
    )

    # %%
    experiment_log_dir = config["experiment_log_dir"]
    gvalid_loss = np.inf

    class_weights = 1 / torch.tensor(config["class_weights"], dtype=torch.float)
    class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # %%
    # evaluate on the test set
    """Testing set"""
    # logger.debug('Test on Target datasts test set')
    # print("Test on Target datasts test set")

    # if config["model_type"] == "cnn":
    #     TFC_model.load_state_dict(torch.load())

    #     classifier.load_state_dict(torch.load())
    # elif config["model_type"] == "cnn_small":
    #     TFC_model.load_state_dict(torch.load())

    #     classifier.load_state_dict(torch.load())
    # else:
    #     TFC_model.load_state_dict(torch.load())

    #     classifier.load_state_dict(torch.load())

    TFC_model.load_state_dict(torch.load(config["saved_models_tfc"][fold]))
    classifier.load_state_dict(torch.load(config["saved_models_classifier"][fold]))

    if config["model_type"] == "cnn" or config["model_type"] == "cnn_small":
        test_loss, out_batch, out_label, out_pred, out_probabilties_ = model_test_cnn(
            model=TFC_model,
            test_dl=test_loader,
            config=configs,
            device=device,
            training_mode=training_mode,
            criterion=criterion,
            classifier=classifier,
            classifier_optimizer=classifier_optimizer,
        )
    else:
        test_loss, out_batch, out_label, out_pred, out_probabilties_ = model_test(
            model=TFC_model,
            test_dl=test_loader,
            config=configs,
            device=device,
            training_mode=training_mode,
            criterion=criterion,
            classifier=classifier,
            classifier_optimizer=classifier_optimizer,
        )

    accuracy = accuracy_score(
        y_true=out_label,
        y_pred=out_pred,
    )

    f1 = f1_score(
        y_true=out_label, y_pred=out_pred, average="macro", zero_division=np.nan
    )

    f1_per_class = f1_score(
        y_true=out_label, y_pred=out_pred, average=None, zero_division=np.nan
    )

    try:
        roc_value = roc_auc_score(
            y_true=out_label,
            y_score=out_probabilties_,
            multi_class="ovr",
            average="macro",
        )
    except:
        roc_value = np.nan

    new_data = pd.DataFrame(
        {
            "Test loss": [test_loss.float()],
            "Accuracy": [accuracy],
            "F1 (macro)": [f1],
            "F1 (per class)": [f1_per_class],
            "AUC": [roc_value],
            "model": [f'{config["arch"]}_fold_{fold}_classifier_{timestamp}'],
        }
    )

    df_performance = pd.concat(
        [df_performance, pd.DataFrame(new_data, index=[0])], ignore_index=True
    )


print("Printing the performance")
print(df_performance)
# print(df_performance.mean())
# print(df_performance.std())


# df_performance.to_csv("tfc_finetuned_mlp_05052024.csv", index=False)
