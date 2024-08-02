import numpy as np
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainer

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def train_model(train_data, val_data, tokenizer, data_collator, training_args, device, model_name="facebook/bart-base"):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device)
    model.gradient_checkpointing_enable()
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    trainer.train()
    #trainer.save_model('./model')