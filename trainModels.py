"""
Provides the main train loop for all models
"""
import torch


def train(model, optimizer, loss_fun, train_data_loader, val_data_loader, res_manager, scheduler=None, num_epochs=5,
          batch_size=16, device=torch.device('cuda'), train_whole_model=False, save_as="model.pkl",
          metric_file="STS_metric.pkl"):
    step = 0
    # if we want to train all the model (our added layers + RoBERTa)
    if train_whole_model:
        for param in model.base_model.parameters():
            param.requires_grad = True
    # in case we just want to train our added layer.
    else:
        for param in model.base_model.parameters():
            param.requires_grad = False

    model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        batch_count = 0
        for (input_ids, input_mask, y_true) in train_data_loader:
            y_pred, _ = model(input_ids=input_ids.to(device), attention_mask=input_mask.to(device))
            loss = loss_fun(y_pred, y_true.to(device))
            loss.backward()
            # Optimizer and scheduler step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # Update train loss and step
            train_loss += loss.item()
            step += batch_size
            batch_count += 1
        train_loss /= batch_count
        model.eval()
        with torch.no_grad():
            batch_count = 0
            for (input_ids, input_mask, y_true) in val_data_loader:
                y_pred, _ = model(input_ids=input_ids.to(device), attention_mask=input_mask.to(device))
                loss = loss_fun(y_pred, y_true.to(device))
                val_loss += loss.item()
                batch_count += 1
            val_loss /= batch_count
        res_manager.update_train_val_loss(model, train_loss, val_loss, step, epoch, num_epochs, save_as, metric_file)
        model.train()

    res_manager.save_metrics(metric_file)
