import json
import torch
from transformers import MT5ForConditionalGeneration, AutoTokenizer
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from config import *
import re
import os

# use gpu 0
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# Load the data from JSON files
with open(bk_he_dir, 'r') as f:
    he = json.load(f)

with open(bk_en_dir, 'r') as f:
    en = json.load(f)

def remove_html_tags(text):
    clean_text = re.sub(r'<.*?>', '', text)
    return clean_text

def get_folders(path):
    folders = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            folders.append(item_path)
    return folders

['Beitzah','Chagigah','Eruvin','Moed Katan','Beitzah','Chagigah','Eruvin','Moed Katan','Pesachim','Rosh Hashanah','Shabbat','Yoma','Gittin','Kiddushin','Nazir','Nedarim','Sotah','Yevamot','Bava Batra','Bava Kamma','Bava Metzia']
complete = ['Megillah','Sukkah','Ketubot','Makkot']
total_he = []
total_en = []
for seder in ['Seder Kodashim','Seder Moed','Seder Nashim','Seder Nezikin','Seder Tahorot']:#,'Seder Zeraim'
    mesechtas = get_folders('/scratch/zceemsi/DLNLP_assignment_23/Datasets/Sefaria-Export/json/Talmud/Bavli/{seder}'.format(seder=seder))
    mesechtas.sort()
    for mesechta in mesechtas:
        # if mesechta.split('/')[-1] not in complete:
        #     continue
        #print(mesechta)
        with open(os.path.join(mesechta,'English','merged.json'), 'r') as f:
            en = json.load(f)
            
        with open(os.path.join(mesechta,'Hebrew','merged.json'), 'r') as f:
            he = json.load(f)

        en_text = [remove_html_tags(text) for chapter in en['text'] for text in chapter] #if text]
        he_text = [remove_html_tags(text) for chapter in he['text'] for text in chapter] #if text]
        total_en += en_text
        total_he += he_text

# Combine and preprocess the data
#he_texts = [text for chapter in he['text'] for text in chapter if text]
#en_texts = [text.replace('<b>', '').replace('</b>', '') for chapter in en['text'] for text in chapter if text]

data = list(zip(total_he, total_en))


# Load the model and tokenizer
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

# Send the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def preprocess_function(example):
    source_text, target_text = example
    inputs = tokenizer("translate Hebrew to English: " + source_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
    targets = tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)

    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    targets = {k: v.squeeze(0) for k, v in targets.items()}

    return {**inputs, "labels": targets["input_ids"]}



#pip install torch==2.1.0.dev20230328+cu117 -f https://download.pytorch.org/whl/nightly/cu117/torch_nightly.html transformers datasets pandas protobuf==3.20 sentencepiece

# Create dataset
preprocessed_data = [preprocess_function(example) for example in data]
dataset = Dataset.from_dict({k: [d[k] for d in preprocessed_data] for k in preprocessed_data[0]})

# Split dataset into train and validation
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./finetuned_mt5_he_en_whole",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    logging_dir="./finetuned_mt5_he_en_whole",

)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./finetuned_mt5_he_en_whole")
tokenizer.save_pretrained("./finetuned_mt5_he_en_whole")

# Load the fine-tuned model and tokenizer
finetuned_model = MT5ForConditionalGeneration.from_pretrained("./finetuned_mt5_he_en_whole")
finetuned_tokenizer = AutoTokenizer.from_pretrained("./finetuned_mt5_he_en_whole")

# Generate translation
hebrew_text = "רבי יהודה"
inputs = finetuned_tokenizer("Translate Hebrew to English: " + hebrew_text, return_tensors="pt")
translated_tokens = finetuned_model.generate(**inputs, num_beams=5)
translation = finetuned_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
print(translation)

