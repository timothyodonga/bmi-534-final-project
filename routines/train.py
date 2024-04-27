def train_one_epoch(
    train_dataloader,
    optimizer,
    device,
    criterion,
    model,
):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, (inputs, label0) in enumerate(train_dataloader):
        # Every data instance is an input + label pair

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Put the data on the device
        inputs = inputs.to(device)
        label0 = label0.to(device)

        # print("Printing the shape of the inputs to the model")
        # print(inputs.shape)

        # Make predictions for this batch
        out0 = model(inputs)

        # Compute the loss and its gradients
        loss0 = criterion(out0, label0.long())
        loss0.backward()

        # It looks like the loss is added
        loss = loss0

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 50 == (50 - 1):
            last_loss = running_loss / 50  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            running_loss = 0.0
    return last_loss
