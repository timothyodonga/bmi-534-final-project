# bmi-534-final-project

## Pretraining on source data
To pretrain the time-frequency encoder model on the daily living dataset
```{python}
python pre_train_tfc_daily.py -tfc_type=transformer -num_epochs=3 -pretrain_data="daily_living_sample.pt"
```
The encoder type can be changed to 'cnn' by changing the `tfc_type` parameter
The `pretrain_data` parameter contains a path to the preprocessed daily living data to use in pretraining

## Finetuning on the HARTH dataset
```{python}
python train_finetune_harth.py -tfc_type=transformer -model_type=mlp -num_epochs=3 -data_dir="./processed_data"
```
The `data_dir` parameter is the path to the folder containing the preprpocessed train and test data for the 5-folds used in training



## Requirements
To run the code, you mainly need to have PyTorch installed on your system which comes with most of the other dependencies such as numpy, pandas.
The latest Pytorch installation can be found no: https://pytorch.org/

The only other dependencies needed can installed using the script below
```{bash}
pip install sklearn
pip install torchsampler
```
