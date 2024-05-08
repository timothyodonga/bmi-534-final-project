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

# %%
config = {
    "pretrain_data": r"C:\Users\timot\OneDrive\Desktop\EMORY\Spring 2024\BMI-534\project-code\code\bmi-534-final-project\saved_models\daily_living_sample.pt",
    "subset": True,
    "batch_size": 16,
    "training_mode": "pre_train",
    "experiment_log_dir": r"C:\Users\timot\OneDrive\Desktop\EMORY\Spring 2024\BMI-534\project-code\code\bmi-534-final-project",
    "tfc_type": "cnn",
}

# %%
pretrain_train = torch.load(config["pretrain_data"])

# %%
configs = Config()
print(configs.__dict__)
training_mode = config["training_mode"]
subset = config["subset"]

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
    TFC_model = TFCCNN(configs).to(device)

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
                experiment_log_dir, "saved_models", f"{config['tfc_type']}_ckp_last.pt"
            ),
        )

        print("*** TFC loss dropped model saved")

        gtrain_loss = train_loss
