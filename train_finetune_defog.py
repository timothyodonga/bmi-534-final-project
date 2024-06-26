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
from torchsampler import ImbalancedDatasetSampler
from train_test_configs import train_finetune_defog_config
import argparse

# %%
df_performance = pd.DataFrame()
folds = ["0", "1", "2", "3", "4"]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

parser = argparse.ArgumentParser()
parser.add_argument(
    "-tfc_type",
    type=str,
    help="Type of the tfc encoder model",
    default="transformer",
)

parser.add_argument(
    "-model_type",
    type=str,
    help="Type of the downstream classifier model",
    default="mlp",
)

parser.add_argument(
    "-num_epochs",
    type=int,
    help="Number of epochs to run",
    default=20,
)


args = parser.parse_args()
print(args)
config = train_finetune_defog_config

config["model_type"] = args.model_type
config["tfc_type"] = args.tfc_type
config["pretrained_model"] = f"saved_models/{args.tfc_type}_defog_ckp_last.pt"
config["num_epochs"] = args.num_epochs
print(config)

for fold in folds:
    print(f"Fold: {fold}")
    training_mode = "fine_tune_test"
    config["fine_tune_train"] = f"defog_train_fold_{fold}.pt"
    config["fine_tune_test"] = f"defog_test_fold_{fold}.pt"

    print(config)

    # %%
    finetune_train = torch.load(config["fine_tune_train"])
    finetune_test = torch.load(config["fine_tune_test"])

    # %%
    configs = Config()
    print(configs.__dict__)
    training_mode = "pre_train"
    subset = False

    configs.tfc_type = config["tfc_type"]
    configs.TSlength_aligned = 300

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
        shuffle=False,
        drop_last=configs.drop_last,
        num_workers=0,
        # sampler=ImbalancedDatasetSampler(finetune_dataset_train),
    )

    finetune_loader_valid = torch.utils.data.DataLoader(
        dataset=finetune_dataset_valid,
        batch_size=configs.target_batch_size,
        shuffle=False,
        drop_last=configs.drop_last,
        num_workers=0,
        # sampler=ImbalancedDatasetSampler(finetune_dataset_valid),
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

    if config["tfc_type"] == "transformer":
        TFC_model = TFC(configs)
    else:
        TFC_model = TFCCNN(configs, embed_length=292)

    print("Loading the pre-trained model")
    TFC_model.load_state_dict(torch.load(config["pretrained_model"]))
    TFC_model = TFC_model.to(device)
    print("****** Loaded the pretrained model")
    # %%
    if config["model_type"] == "cnn_small":
        classifier = BaselineCNNSmall(
            num_sensor_channels=2, num_output_classes=2, embed_length=120
        ).to(device)
    elif config["model_type"] == "cnn":
        classifier = BaselineCNN(
            num_sensor_channels=2, num_output_classes=2, embed_length=112
        ).to(device)
    else:
        classifier = MLPTFC(num_sensor_channels=2, num_output_classes=2).to(device)

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
    # criterion = nn.CrossEntropyLoss()

    patience = 20
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
                tfc_type=config["tfc_type"],
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
                    tfc_type=config["tfc_type"],
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
                tfc_type=config["tfc_type"],
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
                    tfc_type=config["tfc_type"],
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

        # if valid_loss < gvalid_loss or valid_f1 > max_f1:
        if valid_loss < gvalid_loss:

            os.makedirs("experiments_logs/finetunemodel/", exist_ok=True)

            pt_counter = 0

            if valid_loss < gvalid_loss:
                print("**** Loss has dropped .... model saved")
                gvalid_loss = valid_loss

            # if valid_f1 > max_f1:
            #     print("**** F1 score is greater than previous F1 .... model saved")
            #     max_f1 = valid_f1

            if config["model_type"] == "cnn_small":
                torch.save(
                    TFC_model.state_dict(),
                    "experiments_logs/finetunemodel/"
                    + config["arch"]
                    + f"_{config['tfc_type']}_{config['model_type']}_fold_{fold}_model_{timestamp}.pt",
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
                    + f"_{config['tfc_type']}_{config['model_type']}_fold_{fold}_model_{timestamp}.pt",
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
                    + f"_{config['tfc_type']}_{config['model_type']}_fold_{fold}_model_{timestamp}.pt",
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

        if epoch % 5 == 0:
            print("Checking model generalizability every 5 epochs")
            # evaluate on the test set
            """Testing set"""
            # logger.debug('Test on Target datasts test set')
            print("Test on Target datasts test set")
            if config["model_type"] == "cnn":
                TFC_model.load_state_dict(
                    torch.load(
                        "experiments_logs/finetunemodel/"
                        + config["arch"]
                        + f"_{config['tfc_type']}_{config['model_type']}_fold_{fold}_model_{timestamp}.pt",
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
                        + f"_{config['tfc_type']}_{config['model_type']}_fold_{fold}_model_{timestamp}.pt",
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
                        + f"_{config['tfc_type']}_{config['model_type']}_fold_{fold}_model_{timestamp}.pt",
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
                test_loss, out_batch, out_label, out_pred, out_probabilties_ = (
                    model_test_cnn(
                        model=TFC_model,
                        test_dl=test_loader,
                        config=configs,
                        device=device,
                        training_mode=training_mode,
                        criterion=criterion,
                        classifier=classifier,
                        classifier_optimizer=classifier_optimizer,
                        tfc_type=config["tfc_type"],
                    )
                )
            else:
                test_loss, out_batch, out_label, out_pred, out_probabilties_ = (
                    model_test(
                        model=TFC_model,
                        test_dl=test_loader,
                        config=configs,
                        device=device,
                        training_mode=training_mode,
                        criterion=criterion,
                        classifier=classifier,
                        classifier_optimizer=classifier_optimizer,
                        tfc_type=config["tfc_type"],
                    )
                )

            accuracy = accuracy_score(
                y_true=out_label,
                y_pred=out_pred,
            )

            f1 = f1_score(
                y_true=out_label,
                y_pred=out_pred,
                average="macro",
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

            print("=" * 20)
            print(f"Test Loss: {test_loss}")
            print(f"Accuracy: {accuracy}")
            print(f"F1 score (macro): {f1}")
            print(f"AUC (macro): {roc_value}")

    # %%
    # evaluate on the test set
    """Testing set"""
    # logger.debug('Test on Target datasts test set')
    print("Test on Target dataset's test set")
    if config["model_type"] == "cnn":
        TFC_model.load_state_dict(
            torch.load(
                "experiments_logs/finetunemodel/"
                + config["arch"]
                + f"_{config['tfc_type']}_{config['model_type']}_fold_{fold}_model_{timestamp}.pt",
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
                + f"_{config['tfc_type']}_{config['model_type']}_fold_{fold}_model_{timestamp}.pt",
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
                + f"_{config['tfc_type']}_{config['model_type']}_fold_{fold}_model_{timestamp}.pt",
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
            tfc_type=config["tfc_type"],
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
            tfc_type=config["tfc_type"],
        )

    accuracy = accuracy_score(
        y_true=out_label,
        y_pred=out_pred,
    )

    f1 = f1_score(y_true=out_label, y_pred=out_pred, average="macro")

    try:
        roc_value = roc_auc_score(
            y_true=out_label,
            y_score=out_probabilties_[:, 1],
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
            "model_type": [config["model_type"]],
            "model": [
                f'{config["arch"]}_fold_{fold}_classifier_{config["model_type"]}_{timestamp}'
            ],
        }
    )

    df_performance = pd.concat(
        [df_performance, pd.DataFrame(new_data, index=[0])], ignore_index=True
    )

print(df_performance)

file_path = f'finetuned_{config["model_type"]}_{timestamp}_performance.csv'
if os.path.exists(file_path):
    df_performance.to_csv(file_path, index=False, mode="a", header=None)
else:
    df_performance.to_csv(
        file_path,
        index=False,
    )
