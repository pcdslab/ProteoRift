# ProteoRift
## End-to-end machine-learning pipeline for peptide database search. 

ProteoRift utlizes attention and multitask deep-network which can predict multiple peptide properties (length, missed cleavages, and modification status) directly from spectra. We demonstrate that ProteoRift can predict these properties with up to 97% accuracy resulting in search-space reduction by more than 90%. As a result, our end-to-end pipeline, utlizing Specollate as the underlying engine, is shown to exhibit 8x to 12x speedups with peptide deduction accuracy comparable to algorithmic techniques. 

## Citation
If you use ProteoRift in your work, please cite the following publications:


Full documentation and further functionality are still a work in progress. A step-by-step how-to for training or running our trained version of ProteoRift on your data is available below. Please check back soon for an updated tool!


## Step-by-Step HOW TO
The below sections explain the setup for retraining the model.

### 1. Prerequisites

- A Computer with Ubuntu 16.04 (or later) or CentOS 8.1 (or later).
- At least 120GBs of system memory and 10 CPU cores.
- Cuda enabled GPU with at least 12 GBs of memory. Cuda Toolkit 10.0 (or later).
- OpenMS tool for creating custom peptide database. (Optional)
- Crux for FDR analysis using its percolator option.
#### 1.1 Create Conda Enviornment
`conda env create --file proteorift_env.yml`
#### 1.2 Activate Enviornment
`conda activate proteorift`

### 2. Retrain the Model
You can retrain the ProteoRift model if you wish. However a trained model is available and you can perform your database search by following [database-search section](#3.-Database-Search)
1. Prepare the spectra data (mgf format).
2. Open the config.ini file in your favorite text editor and set the following parameters:
    - `mgf_dir`: Absolute path of the mgf files.
    - `prep_dir` Absolute path to the directory where preprocessed mgf files will be saved.
    - other parameters in the [ml] section: You can adjust different hyperparameters in the [ml] section, e.g., learning_rate, dropout, etc.
3. Setup the [wandb](https://wandb.ai/site) account. Create a project name `proteorift`. Then login to the project using `wandb login.` It would store the logs for training.
4. Run `python read_spectra.py -t l`. It would preprocess the spectra files and split them (training, validation, test) and place in the prep_dir.
5. Run the specollate_train file `python run_train.py`. The model weights would be saved in an output dir.

### 3. Database Search
Our pipeline is using two models. You can train specollate model using [Specollate](https://github.com/pcdslab/SpeCollate). You can train the proteorift model using Section 2. Or you can download the weights for both models [here]().

1. Use mgf files for spectra in `sample_data/spectra`. Or you can use your own spectra files in mgf format.
2. Use human peptidome subset in `sample_data/peptide_database`. You can provide your own peptide database file created using the Digestor tool provided by [OpenMS](https://www.openms.de/download/openms-binaries/).
3. Download the weights for specollate and proteorift model [here]().
4. Set the following parameters in the [search] section of the `config.ini` file:
    - `model_name`: Absolute path to the proteorift model.
    - `specollate_model_path`:  Absolute path to the specollate model. 
    - `mgf_dir`: Absolute path to the directory containing mgf files to be searched.
    - `prep_dir`: Absolute path to the directory where preprocessed mgf files will be saved.
    - `pep_dir`: Absolute path to the directory containing peptide database.
    - `out_pin_dir`: Absolute path to a directory where percolator pin files will be saved. The directory must exist; otherwise, the process will exit with an error.
    - Set database search parameters
5. Run `python read_spectra.py -t u`. It would preprocess the spectra files and place in the prep_dir.
6. Run `python run_search.py`. It would generate the embeddings for spectra and peptides and it would predict the filters for spectra and perform the search. It would generate the output(e.g target.pin, decoy.pin).
7. Once the search is complete; you can analyze the percolator files using the crux percolator tool:
```shell
cd <out_pin_dir>
crux percolator target.pin decoy.pin --list-of-files T --overwrite T
```
