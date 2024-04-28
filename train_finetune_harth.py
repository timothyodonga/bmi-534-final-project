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
        "fine_tune_train": f"/opt/scratchspace/todonga/bmi-534-project/processed_data/harth_train_fold_{fold}.pt",
        "fine_tune_test": f"/opt/scratchspace/todonga/bmi-534-project/processed_data/harth_test_fold_{fold}.pt",
        "pretrained_model": "saved_models/ckp_last.pt",
        "experiment_log_dir": "./",
        "model_type": "cnn_small",
        "arch": "daily2harth",
        "training_mode": "fine_tune_test",
        "num_epochs": 500,
        "model_name": "harth_cnn_small_finetuned",
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
    finetune_dataset_train = Load_DatasetTwo(
        finetune_train,
        configs,
        training_mode,
        target_dataset_size=configs.target_batch_size,
        subset=subset,
        valid=False,
        train_valid_split=0.8,
    )

    finetune_dataset_valid = Load_DatasetTwo(
        finetune_train,
        configs,
        training_mode,
        target_dataset_size=configs.target_batch_size,
        subset=subset,
        valid=True,
        train_valid_split=0.8,
    )

    test_dataset = Load_Dataset(
        finetune_test,
        configs,
        training_mode,
        target_dataset_size=configs.target_batch_size,
        subset=subset,
    )

    # %%
    finetune_loader = torch.utils.data.DataLoader(
        dataset=finetune_dataset_train,
        batch_size=configs.target_batch_size,
        shuffle=True,
        drop_last=configs.drop_last,
        num_workers=0,
    )

    finetune_loader_valid = torch.utils.data.DataLoader(
        dataset=finetune_dataset_valid,
        batch_size=configs.target_batch_size,
        shuffle=True,
        drop_last=configs.drop_last,
        num_workers=0,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=configs.target_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )

    # %%
    print(finetune_dataset_train[:][0].shape)
    print(finetune_dataset_valid[:][0].shape)

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

    patience = 30
    pt_counter = 0
    max_f1 = 0

    # %%
    for epoch in range(1, config["num_epochs"] + 1):
        print(f"Epoch : {epoch}")
        if config["model_type"] != "cnn" and config["model_type"] != "cnn_small":
            train_loss = model_finetune(
                model=TFC_model,
                model_optimizer=model_optimizer,
                val_dl=finetune_loader,
                config=configs,
                device=device,
                training_mode=training_mode,
                criterion=criterion,
                classifier=classifier,
                classifier_optimizer=classifier_optimizer,
            )

            (valid_loss, out_batch, out_label, out_pred, out_probabilities) = (
                model_test(
                    model=TFC_model,
                    test_dl=finetune_loader_valid,
                    config=configs,
                    device=device,
                    training_mode=training_mode,
                    criterion=criterion,
                    classifier=classifier,
                    classifier_optimizer=classifier_optimizer,
                )
            )
        else:
            train_loss = model_finetune_cnn(
                model=TFC_model,
                model_optimizer=model_optimizer,
                val_dl=finetune_loader,
                config=configs,
                device=device,
                training_mode=training_mode,
                criterion=criterion,
                classifier=classifier,
                classifier_optimizer=classifier_optimizer,
            )

            (valid_loss, out_batch, out_label, out_pred, out_probabilities) = (
                model_test_cnn(
                    model=TFC_model,
                    test_dl=finetune_loader_valid,
                    config=configs,
                    device=device,
                    training_mode=training_mode,
                    criterion=criterion,
                    classifier=classifier,
                    classifier_optimizer=classifier_optimizer,
                )
            )

        tfc_scheduler.step(valid_loss)
        classifier_scheduler.step(valid_loss)

        valid_acc = accuracy_score(y_true=out_label, y_pred=out_pred)
        valid_f1 = f1_score(y_true=out_label, y_pred=out_pred, average="macro")
        print(f"Epoch: {epoch}")
        print(
            f"Validation loss: {valid_loss} | Valid acc: {valid_acc} | Valid F1: {valid_f1}"
        )
        print(f"Patience counter: {pt_counter}")

        if valid_loss < gvalid_loss or valid_f1 > max_f1:

            os.makedirs("experiments_logs/finetunemodel/", exist_ok=True)

            pt_counter = 0

            if valid_loss < gvalid_loss:
                print("**** Loss has dropped .... model saved")
                gvalid_loss = valid_loss

            if valid_f1 > max_f1:
                print("**** F1 score is greater than previous F1 .... model saved")
                max_f1 = valid_f1

            if config["model_type"] == "cnn_small":
                torch.save(
                    TFC_model.state_dict(),
                    "experiments_logs/finetunemodel/"
                    + config["arch"]
                    + f"_cnn_small_tfc_fold_{fold}_model_{timestamp}.pt",
                )
                torch.save(
                    classifier.state_dict(),
                    "experiments_logs/finetunemodel/"
                    + config["arch"]
                    + f"_cnn_small_fold_{fold}_classifier_{timestamp}.pt",
                )
            elif config["model_type"] == "cnn":
                torch.save(
                    TFC_model.state_dict(),
                    "experiments_logs/finetunemodel/"
                    + config["arch"]
                    + f"_cnn_tfc_fold_{fold}_model_{timestamp}.pt",
                )
                torch.save(
                    classifier.state_dict(),
                    "experiments_logs/finetunemodel/"
                    + config["arch"]
                    + f"_cnn_fold_{fold}_classifier_{timestamp}.pt",
                )
            else:
                torch.save(
                    TFC_model.state_dict(),
                    "experiments_logs/finetunemodel/"
                    + config["arch"]
                    + f"_fold_{fold}_model_{timestamp}.pt",
                )
                torch.save(
                    classifier.state_dict(),
                    "experiments_logs/finetunemodel/"
                    + config["arch"]
                    + f"_fold_{fold}_classifier_{timestamp}.pt",
                )

        else:
            pt_counter = pt_counter + 1
            if pt_counter == patience:
                print(
                    "====Patience for modeling training reached. Breaking from the train loop===="
                )
                break

    # %%
    # evaluate on the test set
    """Testing set"""
    # logger.debug('Test on Target datasts test set')
    print("Test on Target datasts test set")
    if config["model_type"] == "cnn":
        TFC_model.load_state_dict(
            torch.load(
                "experiments_logs/finetunemodel/"
                + config["arch"]
                + f"_cnn_tfc_fold_{fold}_model_{timestamp}.pt"
            )
        )

        classifier.load_state_dict(
            torch.load(
                "experiments_logs/finetunemodel/"
                + config["arch"]
                + f"_cnn_fold_{fold}_classifier_{timestamp}.pt"
            )
        )
    elif config["model_type"] == "cnn_small":
        TFC_model.load_state_dict(
            torch.load(
                "experiments_logs/finetunemodel/"
                + config["arch"]
                + f"_cnn_small_tfc_fold_{fold}_model_{timestamp}.pt"
            )
        )

        classifier.load_state_dict(
            torch.load(
                "experiments_logs/finetunemodel/"
                + config["arch"]
                + f"_cnn_small_fold_{fold}_classifier_{timestamp}.pt"
            )
        )
    else:
        TFC_model.load_state_dict(
            torch.load(
                "experiments_logs/finetunemodel/"
                + config["arch"]
                + f"_fold_{fold}_model_{timestamp}.pt"
            )
        )

        classifier.load_state_dict(
            torch.load(
                "experiments_logs/finetunemodel/"
                + config["arch"]
                + f"_fold_{fold}_classifier_{timestamp}.pt"
            )
        )

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

    f1 = f1_score(y_true=out_label, y_pred=out_pred, average="macro")

    try:
        roc_value = roc_auc_score(
            y_true=out_label,
            y_score=out_probabilties_,
            multi_class="ovr",
            average="macro",
        )
    except:
        roc_value = np.nan

    print("=" * 20)
    print(f"Test Loss: {test_loss}")
    print(f"Accuracy: {accuracy}")
    print(f"F1 score (macro): {f1}")
    print(f"AUC (macro): {roc_value}")

    new_data = pd.DataFrame(
        {
            "Test loss": [test_loss.float()],
            "Accuracy": [accuracy],
            "F1 (macro)": [f1],
            "AUC": [roc_value],
            "model": [f'{config["arch"]}_fold_{fold}_classifier_{timestamp}'],
        }
    )

    df_performance = pd.concat(
        [df_performance, pd.DataFrame(new_data, index=[0])], ignore_index=True
    )

    if config["model_name"] == "cnn_small":
        df_performance.to_csv(
            f"tfc_finetuned_small_cnn_performance.csv", index=False, mode="a"
        )
    elif config["model_name"] == "cnn":
        df_performance.to_csv(
            f"tfc_finetuned_cnn_performance.csv", index=False, mode="a"
        )
    else:
        df_performance.to_csv(
            f"tfc_finetuned_mlp_performance.csv", index=False, mode="a"
        )
