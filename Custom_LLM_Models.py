import pandas as pd
from datasets import Dataset, load_metric
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

class Transformer_LLM():
    def __init__(self, model_name, label_no, max_token_size=512, load_dir=""):
        self.model_name = model_name
        self.label_no = label_no
        self.max_token_size = max_token_size

        self.load_dir = load_dir

        # if no load path is provided for the finetuned model, init pretrained model and tokenizer
        if len(load_dir) == 0:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=self.label_no)

            self.trainer = None

        # if a model is loaded, also load finetuned trainer for testing and evaluation
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.load_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.load_dir, num_labels=self.label_no)

            self.trainer = Trainer(
                model=self.model,
                tokenizer=self.tokenizer,
            )

    # compute_metrics used for computing evaluation scores during training
    def compute_metrics(self, eval_pred):
        metric = load_metric('accuracy')
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        # return metric.compute(predictions=predictions, references=labels, average='macro')
        return metric.compute(predictions=predictions, references=labels)


    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], max_length=self.max_token_size, padding="max_length", truncation=True)

    def train(self, train_data, eval_data, lr=1e-5, epoch_no=5, batch_size=8, save_dir="model/"):
        train_dataset = Dataset.from_pandas(train_data)
        eval_dataset = Dataset.from_pandas(eval_data)

        tokenized_train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        tokenized_eval_dataset = eval_dataset.map(self.tokenize_function, batched=True)

        self.training_args = TrainingArguments(output_dir=save_dir,
                                          evaluation_strategy="epoch",
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          save_strategy='epoch',
                                          learning_rate=lr,
                                          num_train_epochs=epoch_no,
                                          )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            compute_metrics=self.compute_metrics,
        )

        self.trainer.train()

        self.trainer.save_model(save_dir)
        print("FINETUNING is complete. Model is saved to " + os.getcwd() + "\\" + save_dir)


    def predict_labels(self, test_data):
        test_dataset = Dataset.from_pandas(test_data)
        tokenized_test_dataset = test_dataset.map(self.tokenize_function, batched=True)

        predictions = self.trainer.predict(tokenized_test_dataset)
        predictions = predictions.predictions.argmax(axis=-1)

        return predictions

    def evaluate(self, test_labels, predictions):
        # you can add more customized metrics
        test_score = {'accuracy': accuracy_score(test_labels, predictions),
                      'f1': f1_score(test_labels, predictions, average="macro"),
                      'recall': recall_score(test_labels, predictions, average="macro"),
                      'precision': precision_score(test_labels, predictions, average="macro")}

        return test_score


if __name__ == "__main__":
    data = pd.read_csv('labeled_data.csv')
    data_all = data[['tweet', 'class']].copy()
    data_all.rename(columns={'tweet': 'text', 'class': 'label'}, inplace=True)

    skf = StratifiedKFold(n_splits=5)
    X = data_all.drop('label', axis=1)
    y = data_all.label

    for train_index, test_index in skf.split(X, y):
        data_train = data_all.iloc[train_index]
        data_test = data_all.iloc[test_index]
        break

    skf = StratifiedKFold(n_splits=5)
    X = data_train.drop('label', axis=1)
    y = data_train.label

    for train_index, test_index in skf.split(X, y):
        data_train = data_train.iloc[train_index]
        data_eval = data_train.iloc[test_index]
        break

    llm_model = Transformer_LLM('roberta-base', label_no=3, max_token_size=128)
    llm_model.train(data_train.iloc[:1000], data_eval.iloc[:200], lr=1e-5, epoch_no=1, batch_size=8,
                    save_dir="roberta_finetuned_model")










