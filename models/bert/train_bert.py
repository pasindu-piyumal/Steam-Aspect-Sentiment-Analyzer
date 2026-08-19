import pandas as pd 
import os
from datasets import Dataset
from transformers import (AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer)
from sklearn.model_selection import train_test_split

MODEL_NAME = 'bert-base-uncased'
DATASET = "data/processed/steam_reviews_clean.csv"
OUTPUT_DIR = "models_saved/bert_sentiment"

def main():
    print('Loading dataset')
    df = pd.read_csv(DATASET)

    df = df[['clean_text','sentiment']].dropna()

    labels = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }

    df['label'] = df['sentiment'].map(labels)

    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    train_df, test_df = train_test_split(df,test_size=0.2,random_state=42,stratify=df["label"])
    train_dataset = Dataset.from_pandas(train_df[["clean_text", "label"]],preserve_index=False)
    test_dataset = Dataset.from_pandas(test_df[["clean_text", "label"]],preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch['clean_text'],padding='max_length',truncation=True,max_length=128)

    train_dataset = train_dataset.map(tokenize,batched=True)
    test_dataset = test_dataset.map(tokenize,batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    training_args = TrainingArguments(output_dir=OUTPUT_DIR, learning_rate=2e-5, num_train_epochs=3, per_device_eval_batch_size=16, per_device_train_batch_size=16, eval_strategy='epoch', save_strategy='epoch', logging_steps=50, load_best_model_at_end=True, report_to='none')
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, processing_class=tokenizer)

    print('Training BERT')
    trainer.train()

    print('Saving model')
    trainer.save_model(OUTPUT_DIR)

    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete.")

if __name__ == "__main__":
    main()