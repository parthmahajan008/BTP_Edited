import argparse
import json
import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: str, tokenizer):
    """Load and prepare the augmented dataset for training."""
    with open(data_path, "r") as f:
        data = json.load(f)

    # Prepare examples in the format expected by the model
    examples = []
    for item in data:
        # Format the input text
        context = "\n\n".join(
            [
                f"Document {i+1} (Title: {ctx['title']}): {ctx['text']}"
                for i, ctx in enumerate(item["ctxs"])
            ]
        )
        input_text = f"Context:\n{context}\n\nQuestion: {item['question']}\n\nAnswer: {item['answers'][0]}\n\nRationale: {item['rationale']}"

        # Tokenize
        tokenized = tokenizer(
            input_text,
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            return_tensors="pt",
        )

        examples.append(
            {
                "input_ids": tokenized["input_ids"][0],
                "attention_mask": tokenized["attention_mask"][0],
                "labels": tokenized["input_ids"][0].clone(),
            }
        )

    return Dataset.from_list(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to augmented dataset"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/opt-350m",
        help="Base model to finetune",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the finetuned model",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=100, help="Number of warmup steps"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=10, help="Number of steps between logging"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Number of steps between model saves",
    )
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Prepare dataset
    dataset = load_and_prepare_data(args.data_path, tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
