import pandas as pd
from config import *
# from transformers import T5ForConditionalGeneration, T5Tokenizer

# tokenizer = T5Tokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base")

# input_ids = tokenizer("translate English to German: The house is wonderful.", return_tensors="pt").input_ids
# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# from transformers import MT5Model, AutoTokenizer

# model = MT5Model.from_pretrained("google/mt5-small")
# tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
# article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
# summary = "Weiter Verhandlung in Syrien."
# inputs = tokenizer(article, return_tensors="pt")
# labels = tokenizer(text_target=summary, return_tensors="pt")

# outputs = model(input_ids=inputs["input_ids"], decoder_input_ids=labels["input_ids"])

# hidden_states = outputs.last_hidden_state
# print(hidden_states)

from transformers import MT5ForConditionalGeneration, AutoTokenizer

# Load the model and tokenizer
model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

# Define your text and target language
article = "UN Offizier sagt, dass weiter verhandelt werden muss in Syrien."
target_language = "en"

# Encode the text with the target language prefix
inputs = tokenizer("translate German to English: " + article, return_tensors="pt")

# Generate the translation
translated_tokens = model.generate(**inputs)

# Decode the translated tokens
translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
print(translation)
