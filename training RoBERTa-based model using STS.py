"""
TODO: add file description
"""
from datetime import datetime
import os

import torch
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from STSDataHelper import STSBenchmark, FeatureExtractor
from utils import ModelManager
from models import ROBERTAOnSTS


def train(model, optimizer, train_data_loader, val_data_loader, res_manager, scheduler=None, num_epochs=5, batch_size=16,
          device=torch.device('cuda'), train_whole_model=False, save_as="model.pkl", metric_file="STS_metric.pkl"):
    step = 0
    # if we want to train all the model (our added layers + RoBERTa)
    if train_whole_model:
        for param in model.roberta.parameters():
            param.requires_grad = True
    # in case we just want to train our added layer.
    else:
        for param in model.roberta.parameters():
            param.requires_grad = False

    model.train()

    for epoch in range(num_epochs):
        train_loss = 0.0
        val_loss = 0.0
        batch_count = 0
        for input_ids, input_mask, sim_true in train_data_loader:
            sim_pred, _ = model(input_ids=input_ids.to(device), attention_mask=input_mask.to(device))
            loss = torch.nn.MSELoss()(sim_pred, sim_true.to(device).unsqueeze(1))
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
            for (input_ids, input_mask, sim_true) in val_data_loader:
                sim_pred, _ = model(input_ids=input_ids.to(device), attention_mask=input_mask.to(device))
                loss = torch.nn.MSELoss()(sim_pred, sim_true.to(device).unsqueeze(1))
                val_loss += loss.item()
                batch_count += 1
            val_loss /= batch_count
        res_manager.update_train_val_loss(model, train_loss, val_loss, step, epoch, num_epochs, save_as, metric_file)
        model.train()

    res_manager.save_metrics(metric_file)


def test_single_example(example, model, max_len):
    feature_extractor = FeatureExtractor(tokenizer, max_len)
    model.eval()
    input_ids,input_mask = feature_extractor.build_features(example)
    input_ids = input_ids.reshape(1, -1)
    input_mask = input_mask.reshape(1, -1)
    result = model(input_ids=input_ids.to(device), attention_mask=input_mask.to(device))
    result = result[0].detach().cpu()
    return result.numpy()[0] * 5 #The model is measuring similarity on a scale from 0 to 5


if __name__ == "__main__":
    # Read arguments
    parser = ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Set Hardware acceleration ('GPU' or 'CPU')")
    parser.add_argument("--train_src", default="./stsbenchmark/sts-train.csv", help="Path to train data")
    parser.add_argument("--val_src", default="./stsbenchmark/sts-dev.csv", help="Path to validation data")
    parser.add_argument("--output_dir", default="./models", help="Directory where output will be saved")
    parser.add_argument("--best_model", default="STS_model.pkl", help="Name of resulting model, model will be saved in "
                                                                      "'output_dir'")
    parser.add_argument("--mertic_file", default="STS_metric.pkl", help="Name of resulting model, model will be saved in "
                                                                        "'output_dir'")
    parser.add_argument("--from_file", default=False, help="True to continue model training, otherwise false"
                                                           "model name is specified in 'pretrained' param")
    parser.add_argument("--pretrained", default="STS_model.pkl", help="Pretrained model to continue its training")
    parser.add_argument("--batch_size", default=16, help="Batch size")
    parser.add_argument("--max_seq_length", default=128, help="Max sequence length")
    parser.add_argument("--lr1", default=1e-4, help="learning rate while training the additional layers")
    parser.add_argument("--lr2", default=1e-6, help="learning rate while training the whole model")
    parser.add_argument("--num_epochs1", default=25, help="Number of epochs used to train new layers")
    parser.add_argument("--num_epochs2", default=50, help="Number of epochs used to train the whole new layers")
    parser.add_argument("--dropout_rate", default=0.20, help="Dropout rate")
    args = parser.parse_args()

    # Create needed directories
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set device
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device('cuda')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            device = torch.device('cpu')
            print("CUDA is not available, using CPU...")
    else:
        device = torch.device('cpu')

    # Define Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # Define dataset and data loader
    train_dataset = STSBenchmark(args.train_src, args.max_seq_length, tokenizer)
    val_dataset = STSBenchmark(args.val_src, args.max_seq_length, tokenizer)

    train_data_loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)
    val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)

    # Define model manager
    manager = ModelManager(len(train_dataset), len(val_dataset), args.output_dir)

    # Define model
    if args.from_file:
        model = ROBERTAOnSTS(args.dropout_rate)
        manager.load_checkpoint(args.pretrained, model)
    else:
        model = ROBERTAOnSTS(args.dropout_rate)

    model = model.to(device)

    # Training new layers
    #       define optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr1)
    steps_per_epoch = len(train_dataset)
    #       define scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch * 1,
                                                num_training_steps=steps_per_epoch * args.num_epochs1)
    #       Start training
    print(f"[{datetime.now()}] -- Training new layers started")
    train(model=model, train_data_loader=train_data_loader, val_data_loader=val_data_loader, optimizer=optimizer,
          res_manager=manager, scheduler=scheduler, num_epochs=args.num_epochs1, device=device, train_whole_model=False,
          save_as=args.best_model, metric_file=args.mertic_file)
    print(f"[{datetime.now()}] -- Training new layers ended")

    # Clear cache
    torch.cuda.empty_cache()
    # Training whole model
    #       define optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr2)
    #       define scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch * 1,
                                                num_training_steps=steps_per_epoch * args.num_epochs2)
    #       Start training
    print(f"[{datetime.now()}] -- Training whole layers started")
    train(model=model, train_data_loader=train_data_loader, val_data_loader=val_data_loader, optimizer=optimizer,
          res_manager=manager, scheduler=scheduler, num_epochs=args.num_epochs2, device=device, train_whole_model=True,
          save_as=args.best_model, metric_file=args.mertic_file)
    print(f"[{datetime.now()}] -- Training whole layers ended")
