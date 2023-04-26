# DLNLP_assignment_23

Machine translation (MT) has made significant progress in recent years, primarily driven by advances in deep learning and neural networks. However, specialized domains and underrepresented language varieties, such as Rabbinic Hebrew, remain challenging to translate accurately. Rabbinic Hebrew is a variant of Hebrew used in rabbinic literature, which is of great importance to scholars, religious communities, and historians. This study aims to adapt an existing Hebrew-English MT model, the OPUS-MT, to better handle the translation of Rabbinic Hebrew texts by leveraging transfer learning and fine-tuning techniques.
## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

List the software and libraries required to run the project.

- Python 3.10+
- PyTorch 2.1.0
- Transformers 4.27

### Installation

Clone the repository to your local machine:

```
git clone https://github.com/moshesimon/DLNLP_assignment_23.git
```

Change to the project directory:

```
cd your_project
```

Create a virtual environment and activate it:

```
python -m venv venv
source venv/bin/activate
```
Install the required libraries:
```
pip install -r requirements.txt
```

## Usage

In the main.py file set the following variables:
```
run_experiment_1 = True
run_experiment_2 = True
run_experiment_3 = True
run_experiment_4 = True
```
The run the main.py file:
```
python main.py
```


## Project Structure

Briefly describe the structure of the project, including the purpose of each script and folder.

- Datasets/      # Contains the datasets files
- modules/       # Contains Python modules with functions and classes
- config.py      # Configuration file containing paths and other settings
- main.py        # Main script to run the project
- README.md      # This file