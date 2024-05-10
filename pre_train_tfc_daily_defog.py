# %%
# Import statements
import os
import numpy as np
from datetime import datetime
import argparse
from model import *
from dataloader import generate_freq, Load_Dataset, Load_DatasetTwo
from configs import Config
from trainer import model_pretrain, model_test
from loss import *
from model import *
from train_test_configs import config_pretrain_daily_defog

# %%
config = config_pretrain_daily_defog

parser = argparse.ArgumentParser()
parser.add_argument(
    "-tfc_type",
    type=str,
    help="Type of the tfc encoder model",
    default="transformer",
)


parser.add_argument(
    "-num_epochs",
    type=int,
    help="Number of epochs to run",
    default=20,
)

parser.add_argument(
    "-pretrain_data",
    type=str,
    help="Path to the preprocessed pretrain data",
)


args = parser.parse_args()
print(args)

config["pretrain_data"] = args.pretrain_data
config["tfc_type"] = args.tfc_type


# %%
pretrain_train = torch.load(config["pretrain_data"])

# %%
configs = Config()
print(configs.__dict__)
training_mode = config["training_mode"]
subset = config["subset"]

configs.TSlength_aligned = 300
configs.num_epoch = args.num_epochs

# %%
# Load the data
train_dataset = Load_Dataset(
    pretrain_train,
    configs,
    training_mode,
    target_dataset_size=config["batch_size"],
    subset=subset,
)

# %%
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=configs.batch_size,
    shuffle=True,
    drop_last=configs.drop_last,
    num_workers=0,
)

# %%
with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("We are using %s now." % device)

# TFC_model = TFC(configs).to(device)
if config["tfc_type"] == "transformer":
    TFC_model = TFC(configs).to(device)
else:
    TFC_model = TFCCNN(configs=configs, embed_length=292).to(device)

# %%
print("Printing the TFC model")
print(TFC_model)


model_optimizer = torch.optim.Adam(
    TFC_model.parameters(),
    lr=configs.lr,
    betas=(configs.beta1, configs.beta2),
    weight_decay=3e-4,
)

criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, "min")


# %%
# Routine to pretrain the tfc model
print("Pre training the model")
gtrain_loss = np.inf
experiment_log_dir = config["experiment_log_dir"]

for epoch in range(1, configs.num_epoch + 1):
    train_loss = model_pretrain(
        model=TFC_model,
        model_optimizer=model_optimizer,
        criterion=criterion,
        train_loader=train_loader,
        config=configs,
        device=device,
        training_mode=training_mode,
        tfc_type=config["tfc_type"],
    )
    print(f"\nPre-training Epoch : {epoch}", f"Train Loss : {train_loss:.4f}")

    if train_loss < gtrain_loss:
        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)

        torch.save(
            TFC_model.state_dict(),
            os.path.join(
                experiment_log_dir,
                "saved_models",
                f"{config['tfc_type']}_defog_ckp_last.pt",
            ),
        )

        print("*** TFC loss dropped model saved")

        gtrain_loss = train_loss
