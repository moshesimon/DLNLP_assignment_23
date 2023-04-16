from transformers import MarianMTModel, MarianTokenizer

src_text = [
    "היא שכחה לכתוב לו.",
    "אני רוצה לדעת מיד כשמשהו יקרה.",
    "שמי שרה ואני גרה בלונדון"

]

model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))

for t in translated:
    print( tokenizer.decode(t, skip_special_tokens=True) )

# expected output:
#     She forgot to write to him.
#     I want to know as soon as something happens.
