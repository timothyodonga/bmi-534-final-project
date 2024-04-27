import torch


def validate_model(
    model,
    model_name,
    validation_dataloader,
    device,
    timestamp,
    loss_fn,
    best_vloss,
    epoch_number,
    key,
    k,
):
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    running_vloss = 0.0
    # Disable gradient computation and reduce memory consumption.
    print("Now computing the validation loss")
    with torch.no_grad():
        for i, (vinputs, vlabel0) in enumerate(validation_dataloader):
            vinputs = vinputs.to(device)
            vlabel0 = vlabel0.to(device)

            model.to(device)

            vout0 = model(vinputs)
            vloss0 = loss_fn(vout0, vlabel0.long())

            vloss = vloss0
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print("AVG-VALIDATION LOSS: {}".format(avg_vloss))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            torch.save(
                model.state_dict(),
                f"{model_name}_{key}_{k}_{timestamp}.pth",
            )
            print(
                f"****Model saved at Epoch {epoch_number} -----> AVG-LOSS: {best_vloss}"
            )

    return best_vloss
