# ProteoRift
## End-to-end machine-learning pipeline for peptide database search. 

ProteoRift utlizes attention and multitask deep-network which can predict multiple peptide properties (length, missed cleavages, and modification status) directly from spectra. We demonstrate that ProteoRift can predict these properties with up to 97% accuracy resulting in search-space reduction by more than 90%. As a result, our end-to-end pipeline, utlizing Specollate as the underlying engine, is shown to exhibit 8x to 12x speedups with peptide deduction accuracy comparable to algorithmic techniques. 

## Citation
If you use ProteoRift in your work, please cite the following publications:


Full documentation and further functionality are still a work in progress. A step-by-step how-to for training or running our trained version of ProteoRift on your data is available below. Please check back soon for an updated tool!


<!-- # Step-by-Step HOW TO
The below sections explain the setup for running the database search (on already trained model) or retraining the model using your own data. -->

## System Requirements
- A Computer with Ubuntu 16.04 (or later) or CentOS 8.1 (or later).
- Cuda enabled GPU with at least 12 GBs of memory.
- OpenMS tool for creating custom peptide database. (Optional)

## Installation Guide

### Install Anaconda
[Step by Step Guide to Install Anaconda](https://docs.anaconda.com/anaconda/install/)

### Fork the repository
- Fork the repository to your own account.
- Clone your fork to your machine. 

### Create Conda Enviornment
`cd ProteoRift`

`conda env create --file proteorift_env.yml` (It would take some minutes to install dependencis)
### Activate Enviornment
`conda activate proteorift`

## Demo (Database Search)

Our end-to-end pipeline uses two models [Specollate](https://github.com/pcdslab/SpeCollate) and ProteoRift. 

1. Use mgf files for spectra in `sample_data/spectra`. Or you can use your own spectra files in mgf format.
2. Use human peptidome subset in `sample_data/peptide_database`. You can provide your own peptide database file created using the Digestor tool provided by [OpenMS](https://www.openms.de/download/openms-binaries/).
3. Download the weights for specollate and proteorift model [here](https://github.com/pcdslab/ProteoRift/releases/tag/V1.0.0) under the Assets section.
4. Set the following parameters in the [search] section of the `config.ini` file:
    - `model_name`: Absolute path to the proteorift model (called *proteorift_model_weights.pt* that you downloaded from [here](https://github.com/pcdslab/ProteoRift/releases/tag/V1.0.0) under the Assets section).
    - `specollate_model_path`:  Absolute path to the specollate model (called *specollate_model_weights.pt* that you downloaded from [here](https://github.com/pcdslab/ProteoRift/releases/tag/V1.0.0) under the Assets section). 
    - `mgf_dir`: Absolute path to the directory containing mgf files to be searched.
    - `prep_dir`: Absolute path to the directory where preprocessed mgf files will be saved.
    - `pep_dir`: Absolute path to the directory containing peptide database.
    - `out_pin_dir`: Absolute path to a directory where percolator pin files will be saved. The directory must exist; otherwise, the process will exit with an error.
    - Set database search parameters
5. Run `python read_spectra.py -t u`. It would preprocess the spectra files and place in the prep_dir.
6. Run `python run_search.py`. It would generate the embeddings for spectra and peptides and it would predict the filters for spectra and perform the search. It would generate the output(e.g target.pin, decoy.pin).

#### Expected Output
The database search would output two files (target.pin, decoy.pin). `target.pin` contains the information about Target Peptide Spectrum Match. `decoy.pin` contains the information about Decoy Peptide Spectrum Match. Both .pin file would have the features given below for Peptide-Spectrum Match.

![alt text](PSM.png)

 Once the search is complete and .pin are generated; you can analyze the percolator files using the crux percolator tool:
```shell
cd <out_pin_dir>
crux percolator target.pin decoy.pin --list-of-files T --overwrite T
```
##### Time of Execution
Execution time is dependent on many factors including your machines, size of the data, size of the spectra, and what kind of search-space reduction was achieved. As an example, for a 3.9GB database our proposed method completes the search in 1.65 hours with filters. 

## Uncertainty Analysis
To perform the uncertainty Analysis, open notebook `uncertainty_analysis/uncertainty-analysis-specs.ipynb`.

1. Set the following parameters in the notebook:

 - `model_path`: Absolute path to the specollate model weights (called *specollate_model_weights.pt* that you downloaded from [here](https://github.com/pcdslab/ProteoRift/releases/tag/V1.0.0) under the Assets section) 
 - `in_tensor_dir`: Path to your data folder, it should contain your data after preprocessing (Review the comments in the notebook to identify the files generated after preprocessing.)
2. Install dependencies specified in the notebook
3. Run the notebook 


## Retrain the Model 

You can retrain the ProteoRift model if you wish. 
1. Prepare the spectra data (mgf format).
2. Open the config.ini file in your favorite text editor and set the following parameters:
    - `mgf_dir`: Absolute path of the mgf files.
    - `prep_dir` Absolute path to the directory where preprocessed mgf files will be saved.
    - other parameters in the [ml] section: You can adjust different hyperparameters in the [ml] section, e.g., learning_rate, dropout, etc.
3. Setup the [wandb](https://wandb.ai/site) account. Create a project name `proteorift`. Then login to the project using `wandb login.` It would store the logs for training.
4. Run `python read_spectra.py -t l`. It would preprocess the spectra files and split them (training, validation, test) and place in the prep_dir.
5. Run the specollate_train file `python run_train.py`. The model weights would be saved in an output dir.



