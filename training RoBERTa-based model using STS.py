"""
Read configs from command line arguments and train the model using STS benchmark
"""
from datetime import datetime
import os

import torch
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, AdamW, get_linear_schedule_with_warmup
from STSDataHelper import STSBenchmark, InputBuilder
from utils import ModelManager, set_device
from models import ROBERTAOnSTS
import trainModels

if __name__ == "__main__":
    # Read arguments
    parser = ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Set Hardware acceleration ('GPU' or 'CPU')")
    parser.add_argument("--train_src", default="./stsbenchmark/sts-train.csv", help="Path to train data")
    parser.add_argument("--val_src", default="./stsbenchmark/sts-dev.csv", help="Path to validation data")
    parser.add_argument("--output_dir", default="./models", help="Directory where output will be saved")
    parser.add_argument("--best_model", default="STS_model.pkl", help="Name of resulting model, model will be saved in "
                                                                      "'output_dir'")
    parser.add_argument("--mertic_file", default="STS_metric.pkl",
                        help="Name of resulting model, model will be saved in "
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
    device = set_device(args.device)

    # Define Tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # Define dataset and data loader
    train_dataset = STSBenchmark(args.train_src, args.max_seq_length, tokenizer)
    val_dataset = STSBenchmark(args.val_src, args.max_seq_length, tokenizer)

    train_data_loader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size)
    val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size)

    # Define model manager
    manager = ModelManager(args.output_dir)

    # Define model
    if args.from_file:
        model = ROBERTAOnSTS(args.dropout_rate)
        manager.load_checkpoint(args.pretrained, model)
    else:
        model = ROBERTAOnSTS(args.dropout_rate)

    model = model.to(device)

    # define loss function
    loss_fun = torch.nn.MSELoss()
    # Training new layers
    #       define optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr1)
    steps_per_epoch = len(train_dataset)
    #       define scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=steps_per_epoch * 1,
                                                num_training_steps=steps_per_epoch * args.num_epochs1)
    #       Start training
    print(f"[{datetime.now()}] -- Training new layers started")
    trainModels.train(model=model, loss_fun=loss_fun, train_data_loader=train_data_loader,
                      val_data_loader=val_data_loader, optimizer=optimizer, res_manager=manager, scheduler=scheduler,
                      num_epochs=args.num_epochs1, device=device, train_whole_model=False, save_as=args.best_model,
                      metric_file=args.mertic_file)
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
    print(f"[{datetime.now()}] -- Training entire model started")
    trainModels.train(model=model, loss_fun=loss_fun, train_data_loader=train_data_loader,
                      val_data_loader=val_data_loader, optimizer=optimizer, res_manager=manager, scheduler=scheduler,
                      num_epochs=args.num_epochs2, device=device, train_whole_model=True, save_as=args.best_model,
                      metric_file=args.mertic_file)
    print(f"[{datetime.now()}] -- Training entire model ended")
