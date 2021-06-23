"""
TODO: add file description
"""

import torch
from argparse import ArgumentParser

from transformers import RobertaTokenizer
from glueDataHelper import FeatureExtractor
from datasets import load_dataset
from utils import ModelManager
from models import ROBERTAOnSTS, ROBERTA_FT_MRPC


def test_single_example(example, model, max_len):
    feature_extractor = FeatureExtractor(tokenizer, max_len)
    model.eval()
    input_ids, input_mask, _ = feature_extractor.build_features(example)
    input_ids = input_ids.reshape(1, -1)
    input_mask = input_mask.reshape(1, -1)
    result = model(input_ids=input_ids.to(device), attention_mask=input_mask.to(device))
    result = result[0].detach().cpu()
    return result[0].detach().cpu().numpy(), torch.argmax(result).numpy()


if __name__ == "__main__":
    # Read arguments
    parser = ArgumentParser()
    parser.add_argument("--device", default="cuda", help="Set Hardware acceleration ('GPU' or 'CPU')")
    parser.add_argument("--models_dir", default="./models", help="Directory where model is saved")
    parser.add_argument("--model", default="MRPC_after_STS_model.pkl", help="Trained model")
    parser.add_argument("--max_seq_length", default=128, help="Max sequence length")
    args = parser.parse_args()

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
    raw_train_data = load_dataset('glue', 'mrpc', split='train')
    raw_val_data = load_dataset('glue', 'mrpc', split='validation')
    raw_test_data = load_dataset('glue', 'mrpc', split='test')

    # Define model manager
    manager = ModelManager(0, 0, args.models_dir)

    # Define model
    model = ROBERTA_FT_MRPC(ROBERTAOnSTS())
    manager.load_checkpoint(args.model, model)

    model = model.to(device)
    # Find test accuracy
    print("Summary: ")
    # Find train accuracy
    cnt = 0
    for i in range(len(raw_train_data)):
        example = raw_train_data[i]
        prob, result = test_single_example(example, model, args.max_seq_length)
        if example['label'] == result:
            cnt += 1

    print(f"Train Accuracy = {cnt / len(raw_train_data)}")

    # Find validation accuracy
    cnt = 0
    for i in range(len(raw_val_data)):
        example = raw_val_data[i]
        prob, result = test_single_example(example, model, args.max_seq_length)
        if example['label'] == result:
            cnt += 1

    print(f"Validtion Accuracy = {cnt / len(raw_val_data)}")

    # Find test accuracy
    cnt = 0
    for i in range(len(raw_test_data)):
        example = raw_test_data[i]
        _, result = test_single_example(example, model, args.max_seq_length)
        if example['label'] == result:
            cnt += 1

    print(f"Test Accuracy = {cnt / len(raw_test_data)}")

