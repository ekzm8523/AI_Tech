from transformers import AutoTokenizer
from data import KLUEDataset
from datasets import list_metrics, load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
import wandb

from kobert_transformers import get_kobert_lm

def model_init():
    return AutoModelForSequenceClassification.from_pretrained("klue/bert-base", num_labels=1)

if __name__ == "__main__":
    task = "sts"
    pretrained_model_name_or_path = "klue/bert-base"
    batch_size = 64

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=True)
    dataset = KLUEDataset(task)

    def preprocess_function(examples):
        return tokenizer(
            examples['sentence1'],
            examples['sentence2'],
            truncation=True,
            max_length=512,
            padding=True
        )

    encoded_dataset = dataset.get_dataset().map(preprocess_function, batched=True)
    num_labels = 1
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, num_labels=num_labels)

    model_name = pretrained_model_name_or_path.split("/")[-1]

    output_dir = os.path.join("test-klue", "sts")
    logging_dir = os.path.join(output_dir, 'logs')
    learning_rate = 2e-5
    epoch = 30
    weight_decay = 0.01
    warmup_steps = 200
    metric_name = "pearsonr"
    training_args = TrainingArguments(
        # checkpoint
        output_dir=output_dir,
        # overwrite_output_dir=True,

        # Model Save & Load
        save_strategy="epoch",  # 'steps'
        load_best_model_at_end=True,
        # save_steps = 500,


        # Dataset
        num_train_epochs=epoch,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,

        # Optimizer
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        # warmup_steps=warmup_steps,

        # Resularization
        # max_grad_norm=1.0,
        # label_smoothing_factor=0.1,

        # Evaluation
        metric_for_best_model='eval_' + metric_name,
        evaluation_strategy="epoch",

        # Huggingface Hub Upload
        # push_to_hub=True,
        # push_to_hub_model_id=f"{model_name}-finetuned-{task}",

        # Logging
        logging_dir=logging_dir,
        report_to="wandb",

        # Randomness
        seed=42
    )
    metric = load_metric(metric_name)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = predictions[:, 0]
        eval_result = metric.compute(predictions=predictions, references=labels)
        return eval_result

    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['validation'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


    wandb.login()
    wandb.init(project='klue-sts', entity='ekzm8523', name='sts')

    trainer.train()
    trainer.evaluate()
    wandb.finish()

    # trainer.push_to_hub()  # if you want to push hub please uncomment
    model = AutoModelForSequenceClassification.from_pretrained(
        'ekzm8523/bert-base-finetuned-sts',
        num_labels=num_labels
    )
