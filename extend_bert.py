import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Load model and tokenizer, and extend length
max_length = 1024 # BERT has 512 token max length as default
samples = 100
train = load_dataset('imdb', split='train').shuffle().select(range(samples))
test = load_dataset('imdb', split='test').shuffle().select(range(samples))
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer.model_max_length = max_length
model.config.max_position_embeddings = max_length
model.base_model.embeddings.position_ids = torch.arange(max_length).expand((1, -1))
model.base_model.embeddings.token_type_ids = torch.zeros(max_length).expand((1, -1))
orig_pos_emb = model.base_model.embeddings.position_embeddings.weight
model.base_model.embeddings.position_embeddings.weight = torch.nn.Parameter(torch.cat((orig_pos_emb, orig_pos_emb)))

# Tokenize IMDB dataset
def tokenize(examples):
    return tokenizer(examples['text'], truncation=True)
train_tokenized = train.map(tokenize)
test_tokenized = test.map(tokenize)

# Fine tune model
training_args = TrainingArguments(
    output_dir="./bert-imdb",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)
accuracy = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()