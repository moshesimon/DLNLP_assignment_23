from transformers import MarianMTModel
from modules.data_collection import get_datasets
from modules.data_process import det_dataloaders
from modules.evaluate_model import test
from modules.train_model import fine_tune
from config import *

run_experiment_1 = True
run_experiment_2 = True
run_experiment_3 = True
run_experiment_4 = True
num_epochs = 10
lr = 1e-5
batch_size = 16


print("Getting data...")
raw_datasets1, raw_datasets2, raw_datasets3, raw_datasets4 = get_datasets()

##############################################################################
# EXPERIMENT 1
##############################################################################
if run_experiment_1:
    # load data for experiment 1
    print("Loading data for experiment 1...")
    train_dataloader1, val_dataloader1, test_dataloader1 = det_dataloaders(raw_datasets1, batch_size=batch_size)

    # load model
    print("Loading model for experiment 1...")
    model = MarianMTModel.from_pretrained(model_name)

    #get benchmark score for experiment 1
    print("Getting benchmark score for experiment 1...")
    benchmark_score1 = test(model, test_dataloader1)

    #fine tune model for experiment 1
    print("Fine tuning model for experiment 1...")
    model = fine_tune(model, train_dataloader1, val_dataloader1, num_epochs=num_epochs, lr=lr, experiment=1)

    # test model for experiment 1
    print("Testing model for experiment 1...")
    test_score1 = test(model, test_dataloader1)

##############################################################################
# EXPERIMENT 2
##############################################################################
if run_experiment_2:
    # load data for experiment 2
    print("Loading data for experiment 2...")
    train_dataloader2, val_dataloader2, test_dataloader2 = det_dataloaders(raw_datasets2, batch_size=batch_size)

    # reset model
    print("Resetting model for experiment 2...")
    model = MarianMTModel.from_pretrained(model_name)

    #get benchmark score for experiment 2
    print("Getting benchmark score for experiment 2...")
    benchmark_score2 = test(model, test_dataloader2)

    #fine tune model for experiment 2
    print("Fine tuning model for experiment 2...")
    model = fine_tune(model, train_dataloader2, val_dataloader2, num_epochs=num_epochs, lr=lr,  experiment=2)

    # test model for experiment 2
    print("Testing model for experiment 2...")
    test_score2 = test(model, test_dataloader2)

##############################################################################
# EXPERIMENT 3
##############################################################################
if run_experiment_3:
    # load data for experiment 3
    print("Loading data for experiment 3...")
    train_dataloader3, val_dataloader3, test_dataloader3 = det_dataloaders(raw_datasets3, batch_size=batch_size)

    # reset model
    print("Resetting model for experiment 3...")
    model = MarianMTModel.from_pretrained(model_name)

    #get benchmark score for experiment 3
    print("Getting benchmark score for experiment 3...")
    benchmark_score3 = test(model, test_dataloader3)

    #fine tune model for experiment 3
    print("Fine tuning model for experiment 3...")
    model = fine_tune(model, train_dataloader3, val_dataloader3, num_epochs=num_epochs, lr=lr,  experiment=3)

    # test model for experiment 3
    print("Testing model for experiment 3...")
    test_score3 = test(model, test_dataloader3)

##############################################################################
# EXPERIMENT 4
##############################################################################
if run_experiment_4:
    # load data for experiment 4
    print("Loading data for experiment 4...")
    train_dataloader4, val_dataloader4, test_dataloader4 = det_dataloaders(raw_datasets4, batch_size=batch_size)

    # reset model
    print("Resetting model for experiment 4...")
    model = MarianMTModel.from_pretrained(model_name)

    #get benchmark score for experiment 4
    print("Getting benchmark score for experiment 4...")
    benchmark_score4 = test(model, test_dataloader4)
    
    #fine tune model for experiment 4
    print("Fine tuning model for experiment 4...")
    model = fine_tune(model, train_dataloader4, val_dataloader4, num_epochs=num_epochs, lr=lr,  experiment=4)
    
    # test model for experiment 4
    print("Testing model for experiment 4...")
    test_score4 = test(model, test_dataloader4)

# print results table
print("Results Table")
print("Experiment\tBenchmark Score\tTest Score")
print("1\t\t{}\t\t{}".format(benchmark_score1, test_score1))
print("2\t\t{}\t\t{}".format(benchmark_score2, test_score2))
print("3\t\t{}\t\t{}".format(benchmark_score3, test_score3))
print("4\t\t{}\t\t{}".format(benchmark_score4, test_score4))


