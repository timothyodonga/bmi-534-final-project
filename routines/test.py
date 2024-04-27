import numpy as np
import torch
from sklearn.metrics import f1_score
import torch.nn.functional as F


def test_model(
    model,
    test_dataloader,
    criterion,
    device,
    f1_type="macro",
):

    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    accuracy = 0
    total = 0
    f1 = []

    loss_fn = criterion
    running_vloss = 0.0

    out_pred = []
    out_label = []
    out_subj = []
    out_batch = []
    out_probabilties = []

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        # for i, (vinputs, vlabel0, vsubj) in enumerate(test_dataloader):
        for i, (vinputs, vlabel0) in enumerate(test_dataloader):
            vinputs = vinputs.to(device)
            vlabel0 = vlabel0.to(device)

            model.to(device)

            vout0 = model(vinputs)
            vloss0 = loss_fn(vout0, vlabel0.long())

            vloss = vloss0
            running_vloss += vloss

            print(f"Printing the output from batch {i}")
            print(vout0.shape)
            vout = F.softmax(vout0, dim=1)

            _, pred_labels = torch.max(vout0, 1)
            pred_labels = pred_labels.view(-1)
            accuracy += torch.sum(torch.eq(pred_labels, vlabel0)).item()
            total += len(vlabel0)

            out_pred += pred_labels.cpu().int().tolist()
            out_label += vlabel0.cpu().int().tolist()
            out_batch += [i] * len(pred_labels)
            out_probabilties.append(vout.cpu())

        out_probabilties_ = np.vstack(out_probabilties)

        avg_vloss = running_vloss / (i + 1)
        print("LOSS valid {}".format(avg_vloss))
        f1.append(f1_score(y_true=out_label, y_pred=out_pred, average=f1_type))
        print(f"Average loss: {avg_vloss}")
        print(f"Acc: {accuracy}")
        print(f"Total: {total}")
        print(f"Accuracy: {accuracy/total}")
        print(f"f1 score: {np.mean(np.array(f1))}")

    return out_batch, out_label, out_pred, out_probabilties_
