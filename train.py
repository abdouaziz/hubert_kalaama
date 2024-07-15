import warnings
import numpy as np
import re
import json
from datasets import load_dataset, load_metric, Audio
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import torch
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
    AutoProcessor,
    AutoModelForCTC,
    AutoFeatureExtractor,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
)
from transformers import AdamW, get_linear_schedule_with_warmup
from logger import logging
from huggingface_hub import login
import argparse
import os
from hubert import HubertForCTC

# change the projet wandb name
os.environ["WANDB_PROJECT"] = "wolof-asr"


warnings.filterwarnings("ignore")

login(token="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")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on a dataset")
    parser.add_argument(
        "--dataset",
        type=str,
        default="abdouaziiz/wolof_asr",
        help="Dataset to use",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Orange/SSA-HuBERT-base-60k",
        help="Model to use",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dataset",
        help="Output directory",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=3,
        help="Per device train batch size",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=3,
        help="Per device eval batch size",
    )
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        help="Evaluation strategy",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=12,
        help="Number of train epochs",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=40000,
        help="Save steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=1000,
        help="Eval steps",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1000,
        help="Logging steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.005,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Warmup steps",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Save total limit",
    )

    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="abdouaziiz/wolof_xls",
        help="Model ID on the Hub",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="wolof_xls",
        help="Run name",
    )

    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = load_dataset(args.dataset)

    if dataset["train"][0]["audio"]["sampling_rate"] != 16000:

        logger.info(
            f"Dataset {args.dataset} has a sampling rate of {dataset['train'][0]['audio']['sampling_rate']}. The sampling rate must be 16000."
        )

        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    logger.info(f"Dataset: {args.dataset} is loaded")

    chars_to_ignore_regex = '["\?\.\!\-\;\:\(\)\,]'

    def remove_special_characters(batch):
        batch["transcription"] = (
            re.sub(chars_to_ignore_regex, "", batch["transcription"]).lower() + " "
        )
        return batch

    dataset = dataset.map(remove_special_characters)

    def extract_all_chars(batch):
        all_text = " ".join(batch["transcription"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    vocabs = dataset.map(
        extract_all_chars,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=dataset.column_names["train"],
    )

    vocab_list = list(set(vocabs["train"]["vocab"][0]))

    vocab_dict = {v: k for k, v in enumerate(vocab_list)}

    vocab_dict["|"] = vocab_dict[" "]
    del vocab_dict[" "]

    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    print(f"vocab size: {len(vocab_dict)}")

    with open(args.output_dir + "/vocab.json", "w") as vocab_file:
        json.dump(vocab_dict, vocab_file)

    tokenizer = Wav2Vec2CTCTokenizer(
        args.output_dir + "/vocab.json",
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )

    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )

    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    

    def speech_file_to_array_fn(batch):
        # speech_array, sampling_rate = librosa.load(batch["file"], sr = 16000)
        batch["speech"] = batch["audio"]["array"]
        batch["sampling_rate"] = batch["audio"]["sampling_rate"]
        batch["target_text"] = batch["transcription"]
        return batch

    dataset = dataset.map(
        speech_file_to_array_fn,
        remove_columns=dataset.column_names["train"],
        num_proc=1,
    )

    def prepare_dataset(batch):
        # check that all files have the correct sampling rate
        assert (len(set(batch["sampling_rate"])) == 1), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."

        batch["input_values"] = processor(batch["speech"], sampling_rate=batch["sampling_rate"][0]).input_values

        with processor.as_target_processor():
            batch["labels"] = processor(batch["target_text"]).input_ids
            
        return batch

    dataset_prepared = dataset.map(
        prepare_dataset,
        remove_columns=dataset.column_names["train"],
        batch_size=32,
        batched=True,
    )

    @dataclass
    class DataCollatorCTCWithPadding:
        processor: Wav2Vec2Processor
        padding: Union[bool, str] = True
        max_length: Optional[int] = None
        max_length_labels: Optional[int] = None
        pad_to_multiple_of: Optional[int] = None
        pad_to_multiple_of_labels: Optional[int] = None

        def __call__(
            self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
        ) -> Dict[str, torch.Tensor]:
            # split inputs and labels since they have to be of different lenghts and need
            # different padding methods
            input_features = [
                {"input_values": feature["input_values"]} for feature in features
            ]
            label_features = [{"input_ids": feature["labels"]} for feature in features]

            batch = self.processor.pad(
                input_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
            with self.processor.as_target_processor():
                labels_batch = self.processor.pad(
                    label_features,
                    padding=self.padding,
                    max_length=self.max_length_labels,
                    pad_to_multiple_of=self.pad_to_multiple_of_labels,
                    return_tensors="pt",
                )

            # replace padding with -100 to ignore loss correctly
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )

            batch["labels"] = labels

            return batch

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    wer_metric = load_metric("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    model = HubertForCTC.from_pretrained(
        args.model_name_or_path,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    model.freeze_feature_extractor()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        group_by_length=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        evaluation_strategy=args.evaluation_strategy,
        num_train_epochs=args.num_train_epochs,
        fp16=True,
        run_name=args.run_name,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        save_total_limit=args.save_total_limit,
    )

    training_args.metric_for_best_model = "eval_loss"
    training_args.load_best_model_at_end = True
    training_args.push_to_hub = False
    # training_args.hub_model_id = args.hub_model_id

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 10% of training steps are warmup steps
    
    num_warmup_steps = int(0.1 * len(dataset_prepared["train"]) / args.per_device_train_batch_size)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(dataset_prepared["train"]) // args.per_device_train_batch_size * args.num_train_epochs,
    )

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_prepared["train"],
        eval_dataset=dataset_prepared["test"],
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        optimizers=(optimizer, scheduler),
    )

    logger.info(f"***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    logger.info(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {training_args.max_steps}")
    logger.info(f"  Eval steps = {training_args.eval_steps}")
    logger.info(f"  Logging steps = {training_args.logging_steps}")
    logger.info(f"  Save steps = {training_args.save_steps}")
    logger.info(f"  Learning rate = {training_args.learning_rate}")
    logger.info(f"  Weight decay = {training_args.weight_decay}")
    logger.info(f"  Warmup steps = {training_args.warmup_steps}")
    logger.info(f"  Save total limit = {training_args.save_total_limit}")

    trainer.train()

    if training_args.push_to_hub:
        trainer.push_to_hub()
        processor.push_to_hub(args.hub_model_id)

    trainer.model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
