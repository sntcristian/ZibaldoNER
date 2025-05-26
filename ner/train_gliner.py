import json
import random
import os
import torch
from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset
import argparse


def main(data_path, output_dir, modelname):
    train_path = os.path.join(data_path, "train.json")
    dev_path = os.path.join(data_path, "dev.json")

    with open(train_path, "r") as f:
        train_data = json.load(f)

    with open(dev_path, "r") as f:
        test_data = json.load(f)

    print('Train Dataset size:', len(train_data))
    print('Test Dataset size:', len(test_data))

    random.shuffle(train_data)
    random.shuffle(test_data)

    print('Dataset is shuffled...')


    train_dataset = train_data
    test_dataset = test_data


    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = GLiNER.from_pretrained(modelname)

    model.data_processor.config.max_len = 764

    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels=True)
    model.to(device)
    print("done")

    num_steps = len(train_dataset)
    batch_size = 4
    data_size = len(train_dataset)
    num_batches = data_size // batch_size
    num_epochs = max(1, num_steps // num_batches)
    print("Epochs: ", num_epochs)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-6,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear", #cosine
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        evaluation_strategy="steps",
        save_strategy="no",
        dataloader_num_workers = 0,
        use_cpu = False,
        report_to="none",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    trainer.model.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training GliNER model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to JSON NER dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="directory for storing trained model")
    parser.add_argument("--modelname", type=str, default="DeepMount00/GLiNER_ITA_LARGE", help="pre-trained GliNER model")
    args = parser.parse_args()
    params = vars(args)

    data_path = params["dataset_path"]
    output_dir = params["output_dir"]
    modelname = params["modelname"]
    main(data_path, output_dir, modelname)

