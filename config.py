import os

base_dir = os.path.dirname(os.path.abspath(__file__))

datasets_path = os.path.join(base_dir, 'Datasets')
figures_path = os.path.join(base_dir, 'Figures')
book1_path = os.path.join(datasets_path, 'Mesilat Yesharim/')
book2_path = os.path.join(datasets_path, 'Orchot Tzadikim/')

model_name = "Helsinki-NLP/opus-mt-tc-big-he-en"