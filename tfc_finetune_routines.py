# %%
# Import statements
import numpy as np
from model import *
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


# %%
def model_finetune(
    model,
    model_optimizer,
    val_dl,
    config,
    device,
    training_mode,
    criterion,
    classifier=None,
    classifier_optimizer=None,
):
    model.train()
    classifier.train()
    total_loss = []
    # criterion = nn.CrossEntropyLoss()

    for data, labels, aug1, data_f, aug1_f in val_dl:
        data, labels = data.float().to(device), labels.long().to(device)
        data_f = data_f.float().to(device)
        aug1 = aug1.float().to(device)
        aug1_f = aug1_f.float().to(device)

        """if random initialization:"""
        model_optimizer.zero_grad()  # The gradients are zero, but the parameters are still randomly initialized.
        classifier_optimizer.zero_grad()  # the classifier is newly added and randomly initialized

        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)
        nt_xent_criterion = NTXentLoss_poly(
            device,
            config.target_batch_size,
            config.Context_Cont.temperature,
            config.Context_Cont.use_cosine_similarity,
        )
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        l_TF = nt_xent_criterion(z_t, z_f)

        l_1, l_2, l_3 = (
            nt_xent_criterion(z_t, z_f_aug),
            nt_xent_criterion(z_t_aug, z_f),
            nt_xent_criterion(z_t_aug, z_f_aug),
        )
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)  #

        """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test."""
        fea_concat = torch.cat((z_t, z_f), dim=1)

        predictions = classifier(fea_concat)

        loss_p = criterion(predictions, labels)

        lam = 0.1
        loss = loss_p + l_TF + lam * (loss_t + loss_f)

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

    ave_loss = torch.tensor(total_loss).mean()

    print(" Finetune: Train loss = %.4f|" % (ave_loss,))

    return ave_loss


# %%
def model_finetune_cnn(
    model,
    model_optimizer,
    val_dl,
    config,
    device,
    training_mode,
    criterion,
    classifier=None,
    classifier_optimizer=None,
):
    # global labels, pred_numpy, fea_concat_flat
    model.train()
    classifier.train()
    total_loss = []
    # criterion = nn.CrossEntropyLoss()

    for data, labels, aug1, data_f, aug1_f in val_dl:
        data, labels = data.float().to(device), labels.long().to(device)
        data_f = data_f.float().to(device)
        aug1 = aug1.float().to(device)
        aug1_f = aug1_f.float().to(device)

        """if random initialization:"""
        model_optimizer.zero_grad()  # The gradients are zero, but the parameters are still randomly initialized.
        classifier_optimizer.zero_grad()  # the classifier is newly added and randomly initialized

        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)
        nt_xent_criterion = NTXentLoss_poly(
            device,
            config.target_batch_size,
            config.Context_Cont.temperature,
            config.Context_Cont.use_cosine_similarity,
        )
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        l_TF = nt_xent_criterion(z_t, z_f)

        l_1, l_2, l_3 = (
            nt_xent_criterion(z_t, z_f_aug),
            nt_xent_criterion(z_t_aug, z_f),
            nt_xent_criterion(z_t_aug, z_f_aug),
        )
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)  #

        """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test."""

        fea_concat__ = torch.stack((z_t, z_f))
        fea_concat__ = fea_concat__.permute(1, 2, 0)
        fea_concat__ = fea_concat__.unsqueeze(1)
        predictions = classifier(fea_concat__)
        loss_p = criterion(predictions, labels)

        lam = 0.1
        loss = loss_p + l_TF + lam * (loss_t + loss_f)

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

    ave_loss = torch.tensor(total_loss).mean()

    print(" Finetune: Train loss = %.4f|" % (ave_loss,))

    return ave_loss


# %%
def model_test_cnn(
    model,
    test_dl,
    config,
    device,
    training_mode,
    criterion,
    classifier=None,
    classifier_optimizer=None,
):
    model.eval()
    classifier.eval()

    accuracy = 0
    total = 0
    f1 = []

    total_loss = []
    total_acc = []

    # criterion = nn.CrossEntropyLoss()  # the loss for downstream classifier

    out_pred = []
    out_label = []
    out_subj = []
    out_batch = []
    out_probabilties = []

    with torch.no_grad():
        for i, (data, labels, _, data_f, _) in enumerate(test_dl):
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)

            """Add supervised classifier: 1) it's unique to finetuning. 
            2) this classifier will also be used in test"""
            h_t, z_t, h_f, z_f = model(data, data_f)
            fea_concat = torch.cat((z_t, z_f), dim=1)
            fea_concat__ = torch.stack((z_t, z_f))
            fea_concat__ = fea_concat__.permute(1, 2, 0)
            fea_concat__ = fea_concat__.unsqueeze(1)

            predictions_test = classifier(fea_concat__)
            loss = criterion(predictions_test, labels)

            vout = F.softmax(predictions_test, dim=1)

            _, pred_labels = torch.max(predictions_test, 1)
            pred_labels = pred_labels.view(-1)
            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            out_pred += pred_labels.cpu().int().tolist()
            out_label += labels.cpu().int().tolist()
            out_batch += [i] * len(pred_labels)
            out_probabilties.append(vout.cpu())

            total_loss.append(loss.item())

        out_probabilties_ = np.vstack(out_probabilties)

    total_loss = torch.tensor(total_loss).mean()
    acc = accuracy / total

    F1 = f1_score(
        y_true=out_label,
        y_pred=out_pred,
        average="macro",
    )

    roc_value = roc_auc_score(
        y_true=out_label, y_score=out_probabilties_, multi_class="ovr", average="macro"
    )
    # print(f"CNN Testing: Loss={total_loss} |  Acc={acc}| F1={F1} | AUROC={roc_value} |")

    return total_loss, out_batch, out_label, out_pred, out_probabilties_


# %%
def model_test(
    model,
    test_dl,
    config,
    device,
    training_mode,
    criterion,
    classifier=None,
    classifier_optimizer=None,
):
    model.eval()
    classifier.eval()

    total_loss = []
    total_acc = []

    accuracy = 0
    total = 0
    f1 = []

    # criterion = nn.CrossEntropyLoss()  # the loss for downstream classifier

    out_pred = []
    out_label = []
    out_subj = []
    out_batch = []
    out_probabilties = []

    with torch.no_grad():
        for i, (data, labels, _, data_f, _) in enumerate(test_dl):
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)

            """Add supervised classifier: 1) it's unique to finetuning. 
            2) this classifier will also be used in test"""
            h_t, z_t, h_f, z_f = model(data, data_f)
            fea_concat = torch.cat((z_t, z_f), dim=1)

            predictions_test = classifier(fea_concat)
            loss = criterion(predictions_test, labels)

            vout = F.softmax(predictions_test, dim=1)

            _, pred_labels = torch.max(predictions_test, 1)
            pred_labels = pred_labels.view(-1)
            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            out_pred += pred_labels.cpu().int().tolist()
            out_label += labels.cpu().int().tolist()
            out_batch += [i] * len(pred_labels)
            out_probabilties.append(vout.cpu())

            total_loss.append(loss.item())

        out_probabilties_ = np.vstack(out_probabilties)

    total_loss = torch.tensor(total_loss).mean()
    acc = accuracy / total

    F1 = f1_score(
        y_true=out_label,
        y_pred=out_pred,
        average="macro",
    )

    roc_value = roc_auc_score(
        y_true=out_label, y_score=out_probabilties_, multi_class="ovr", average="macro"
    )
    # print(f"CNN Testing: Loss={total_loss} |  Acc={acc}| F1={F1} | AUROC={roc_value} |")

    return total_loss, out_batch, out_label, out_pred, out_probabilties_
