# UDBL-FR

## 1. Environment 

Clone the project:

bash

```
git clone https://github.com/chenYL-YL/UDBL-FR.git
cd UDBL-FR
```

Create a Conda environment:

```
conda create -n udbl-fr python=3.8 -y 
conda activate udbl-fr
```

Install dependencies:

```
pip install -r requirements.txt
```

## 2. Project Structure

```
UDBL-FR/
├── checkpoint/
├── dataset/
│   ├── lrDown2/
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   ├── lrDown4/
│   │   ├── test/
│   │   ├── train/
│   │   └── val/
│   └── lrDown8/
│       ├── test/
│       ├── train/
│       └── val/
├── model/
├── requirements.txt
├── test.py
└── train.py
```

## 3. Dataset Preparation

The datasets used in this work are publicly available from the following repository:

https://github.com/XylonXu01/TFS-Diff

The datasets are divided according to different downsampling rates, including lrDown2, lrDown4, and lrDown8. Each rate directory contains:

- train/: Training set
- val/: Validation set
- test/: Test set

Taking lrDown2 as an example, the dataset path should be:

```
dataset/lrDown2/train dataset/lrDown2/val dataset/lrDown2/test
```

If the dataset is placed in another location, please modify the corresponding data path in train.py or test.py.

## 4. Training

Run the training script

```
python train.py
```

Before training, please confirm that the dataset path in train.py is set correctly. For example, when using lrDown2, it should point to:

```
dataset/lrDown2
```

After training is completed, the model weights will be saved to the checkpoint/ directory or the save path specified in the script.

## 5. Testing

The test file in the project takes lrDown2 as an example.

Before testing, please confirm:

- The test data is located in dataset/lrDown2/test/
- The model weights have been placed in the checkpoint/ directory
- The data path and weight path in test.py are set correctly

Run the test:

```
python test.py
```

If you need to test lrDown4 or lrDown8, please modify the data path in test.py from lrDown2 to the corresponding directory.