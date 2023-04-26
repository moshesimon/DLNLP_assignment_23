from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
from transformers import MarianTokenizer, DataCollatorWithPadding
from config import *


def det_dataloaders(raw_datasets, batch_size=8):
    """
    Prepare data loaders for the given raw datasets.

    Args:
        raw_datasets (DatasetDict): A dictionary containing the raw datasets (train, eval, test).
        batch_size (int): The size of each batch in the data loaders.

    Returns:
        tuple: A tuple containing the train, validation, and test data loaders.
    """

    source_lang = "he"
    target_lang = "en"

    # Load the Marian tokenizer
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    def preprocess_function(examples):
        """
        Preprocess examples by tokenizing the source and target sequences.

        Args:
            examples (dict): A dictionary containing the source and target sequences.

        Returns:
            dict: A dictionary containing the tokenized inputs and labels.
        """

        # Ensure examples["translation"] is a list
        if type(examples["translation"]) is not list:
            examples["translation"] = [examples["translation"]]

        # Extract source and target sequences
        inputs = [ex[source_lang] for ex in examples["translation"]]
        targets = [ex[target_lang] for ex in examples["translation"]]

        # Tokenize the inputs
        model_inputs = tokenizer(inputs, return_tensors="pt", padding=True, max_length=512, truncation=True)

        # Tokenize the targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, return_tensors="pt", padding=True, max_length=512, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Parallelize preprocessing and apply it to raw_datasets
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["translation"])
    tokenized_datasets.set_format(type="torch")

    # Define a Data Collator with dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Use custom samplers for data loaders
    train_sampler = RandomSampler(tokenized_datasets['train'])
    eval_sampler = SequentialSampler(tokenized_datasets['eval'])
    test_sampler = SequentialSampler(tokenized_datasets['test'])

    # Create train, validation, and test data loaders
    train_dataloader = DataLoader(
        tokenized_datasets['train'], sampler=train_sampler, batch_size=batch_size, collate_fn=data_collator
    )

    val_dataloader = DataLoader(
        tokenized_datasets['eval'], sampler=eval_sampler, batch_size=batch_size, collate_fn=data_collator
    )

    test_dataloader = DataLoader(
        tokenized_datasets['test'], sampler=test_sampler, batch_size=batch_size, collate_fn=data_collator
    )

    return train_dataloader, val_dataloader, test_dataloader
