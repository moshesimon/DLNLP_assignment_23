import evaluate 
import torch
import numpy as np
from transformers import MarianTokenizer
from config import *

def eval_model(model, eval_dataloader):
    """
    Evaluate the model on the given evaluation dataset.

    Args:
        model (nn.Module): Pretrained model to evaluate.
        eval_dataloader (DataLoader): Evaluation data loader.

    Returns:
        float: Evaluation loss.
    """
    #Evaluate our model performance on the validation set:
    validation_loss = []
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    with torch.no_grad():
        for batch in eval_dataloader:
            #move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            validation_loss.append(outputs.loss.cpu().detach())

    val_loss = np.mean(np.array(validation_loss))
    print("Eval Loss:",val_loss)
    return val_loss

def test(model, test_dataloader):
    """
    Test the model on the given test dataset and compute the BLEU score.

    Args:
        model (nn.Module): Pretrained model to test.
        test_dataloader (DataLoader): Test data loader.

    Returns:
        float: BLEU score.
    """
    #Evaluate our model performance on the test set:
    validation_loss = []
    predictions1 = []
    labels1 = []
    
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    with torch.no_grad():
        for batch in test_dataloader:
            #move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits 
            predictions1.append(torch.argmax(logits, dim=-1))
            labels1.append(batch['labels'])
            validation_loss.append(outputs.loss.cpu().detach())

    val_loss = np.mean(np.array(validation_loss))
    print("Loss:",val_loss)
    predictions_list1 = []
    for p in predictions1:
        prediction = ""
        for t in p:
            prediction += tokenizer.decode(t, skip_special_tokens=True)
        predictions_list1.append(prediction)

    actual_list1 = []
    for l in labels1:
        actual = ""
        for t in l:
            actual += tokenizer.decode(t, skip_special_tokens=True)
        actual_list1.append([actual])

    metric = evaluate.load("sacrebleu")

    results = metric.compute(predictions=predictions_list1, references=actual_list1)
    bleu_score = round(results["score"], 1)
    print("BLEU SCORE:",bleu_score)
    return bleu_score