# Purpose of the new study

The neural model proposed in this work is reused in the new study: **Thinking Like a Developer? Comparing the Attention of Humans with Neural Models of Code**.

## Usage of the Model
We are interested in the attention weights produced by the neural model during inference, thus we use the model for inference and extract the weights for the entire testing dataset. This testing dataset used corresponds to the test split form the previously gathered Java dataset **A convolutional attention network for extreme summarization of source code**. We refer to this as the *Java Small* dataset.

------------------
# Reproducibility

## Computing Environment

We used the following setup:
- Operating System: Ubuntu 18.04.5 LTS
- Kernel: Linux 4.15.0-136-generic
- Architecture: x86-64
- conda 4.9.2
- x2 GPUs: Tesla T4 with 16 GB each
- RAM: 252 GB
- Nvidia driver (as reported by nvidia-smi command): NVIDIA-SMI 450.102.04   Driver Version: 450.102.04   CUDA Version: 11.0
- Nvidia drivers in folder: /usr/local/cuda-10.2

## Python Dependencies

We use conda as a package manager and we have two conda environments: one for train and one for test. We report the solution that worked for us after some tries, further tries are required to check if only one configuration is enough, we recommend to follow our setup as much as possible.

We dumped the requirement packages with the command: ```conda list -e > requirements_conda.txt```.

Make sure to have the two channels (conda-forge and anaconda) by checking ```conda config --show channels```. If you do not have them run the following:
- ```conda config --append channels conda-forge```
-  ```conda config --append channels anaconda```.

Our command run ```conda config --show channels``` returns:
```console
channels:
    - anaconda
    - defaults
    - conda-forge
```

### Conda - Train Environment

To create new predictions from pre-trained models create a conda environment with the following requirements:
```console
conda create --name condaenv_test --file requirements_conda_test.txt*
```
where **condaenv** is an arbitrary name for the folder with your dependencies for *training*.

### Conda - Test Environment

To create new predictions from pre-trained models create a conda environment with the following requirements:
```console
conda create --name condaenv_test --file requirements_conda_test.txt*
```
where **condaenv_test** is an arbitrary name for the folder with your dependencies for *testing*.

## Direct Data Download
You can download the data needed [here](https://figshare.com/s/f8b198bb271ed33e5920/articles/14431439), the package contains a readme with more information, but in synthesis it contains these data:
- **data** folder with data from Java Small dataset and the specific methods proposed to the human participants of the new study.
    The **data** folder is organized in subfolders (one per dataset) with the following structure, as expected by the NeuralCodeSum repository:
    - new_dataset_folder
        - train
        - dev
        - test

    (e.g. NeuralCodeSum/**data/elasticsearch_transformer/dev**)
- **tmp_finetuning** and **tmp_single_project** folders containing trained models and attention weights.

## Folder management
we assume that you downloaded the data from the original paper in the folder **dataset_convolutional-attention** as a sibling of the folder containing this project.
If you want to use the pretrained models we assume that they placed in the **tmp_finetuning** and **tmp_single_project** folders located in the main folder of this repository.
The **data** folder will instead contain the Java Small (Allamanis) dataset as prepared in the ``input preparation'' section, plus the **human_dataset_input** folder containing only the specific function for which we have human annotations and we have extracted the attention weights.

------------------
# Training Procedure


## Input preparation
We use the same token provided in the original dataset. To enforce the transformer model to use the same tokenization we feed to the transformer model the a stream of input tokens that are concatenated with a special character sequence **&*separator*&**. Then a modified version of the inputter of the transformer will parse the input and split it based on this custom token instead of using the space character.
You have two option: download the data we already preprocessed for you (check *Direct Data Download* paragraph above) or replicate also that part of the study, in case for example you want to force another tokenization.

### Prepare the data from the original dataset
To prepare the input sequences for the transformer we use the notebook **Create_Training_for_Transformer.ipynb** and we place it in the folder containing the json files of the Java Small dataset.

## Background run
For the training we use **screen** to run the training in the background and have the log on a file:
```
screen -L -Logfile long_training_first_try_via_screen bash
```

## Main files for training

The scripts to train the models are located in the folder **script/java**. We obtained three different models:

1. Model **tmp_finetuning/code2jdoc.mdl**: trained on the Java dataset of the this work (transformer paper). The task is to predict the java docstring from the method code (signature included). Command run:
    ```console
    bash transformer.sh 0,1 code2jdoc
    ```
    The run is logged in: *long_training_first_try_via_screen* file.
1. Model **tmp_finetuning/body2name.mdl**: trained starting from the previous **code2jdoc.mdl** model but the training dataset corresponds to the entire training dataset of the Java Small dataset (Allamanis). In this case, the task is to predict function name given only the method body's tokens (pure method naming task). Note that here we finetune on the a **cross-project dataset** (the entire Java Small training dataset).
Command run:
    ```console
    bash transformer_finetuning.sh 0,1 body2name
    ```
    The run is logged in: *long_training_finetuning* file.

1. Model **tmp_single_project/<project_name>_transformer.mdl**: trained starting from the previous **code2jdoc.mdl** model and finetune with only the **project-specific** part of the Java Small dataset (Allamanis). Note that this produces ten project-specific models specialized on the method naming task.
    Command run:
    ```console
    ./transformer_finetuning_single_project.sh 0,1
    ```
    The run is logged in: *long_training_finetuning_mono_project* file.
    **transformer_finetuning_single_project.sh** contains a hard reference to the code2doc.mdl model, so if you want a different model or no pretrained model change it here.


-----------------
# Testing procedure - Extract Attention Weights

To generate the prediction with the various models we use the following scripts in the folder **scripts**:
- **generate_pretrained_model.sh** to produce the prediction and attention weights of the specified model passed as argument. The output is saved in the**tmp_finetuning** folder.
- **generate_single_project.sh** to produce the prediction and attention weights for all the ten project-specific models. The output is saved in the **tmp_single_project** folder.

We test only on the methods for which we have human data.
These data are in NeuralCodeSum/**data/human_dataset_input/** folder and they should be organized in one file per project:
- data/human_dataset_input/cassandra_human_annotated.code
- data/human_dataset_input/elasticsearch_human_annotated.code
- etc...

Each of those files is generally small and contains the forced Allamanis tokenization:

```txt
{&*separator*&return&*separator*&completed&*separator*&;&*separator*&}
{&*separator*&return&*separator*&receiving&*separator*&Streams&*separator*&.&*separator*&get&*separator*&(&*separator*&plan&*separator*&Id&*separator*&)&*separator*&;&*separator*&}
{&*separator*&throw&*separator*&new&*separator*&unsupported&*separator*&Operation&*separator*&Exception&*separator*&(&*separator*&)&*separator*&;&*separator*&}
...
```

The ouptut contains the following files (where a prefix indicates the model). They are:
    - <model_prefix>_beam.json: it contains the predicted method names.
    - <model_prefix>_beam.json.attention_copy: it contains the copy attention weights produced during the prediction of each method name.
    - <model_prefix>_beam.json.attention_transformer_0..7: it contains the regular transformer multi-head attention for the eight heads.


# Modified files
Here we list the main modification done to the files in this repository and why:

- **main/test_allamanis.py**: script to compute the model prediction and extract the attention weights form the model. They weights are saved in the folder **tmp_single_project** and they start with a prefix that refers to the model. They are:
    - <model_prefix>_beam.json: it contains the predicted method names.
    - <model_prefix>_beam.json.attention_copy: it contains the copy attention weights produced during the prediction of each method name.
    - <model_prefix>_beam.json.attention_transformer_0..7: it contains the regular transformer multi-head attention for the eight heads.
- **c2nl/inputters/utils.py**: script to load the data. We ensure that every time that we load a dataset coming from the specified folder project it is split in tokens looking for the custom separator.
- **c2nl/inputters/constants.py**: it contains indications for the transformer model about which is the programming language of each dataset in the data folder. We modified it so that every folder name is present here.
- **c2nl/translator/beam.py**, **c2nl/translator/translation.py**, **c2nl/translator/translator.py**: they have been adopted to extract the attention weights. Note that the modifications might have changed the code in a way it is not compatible with other hyperparameters but the transformer with copy attention set to true (we did not test other configurations).







------------------
# Reproducibility Review
The project is well-organized and the files auto-documented. The major effort was put it the adaptation to extract the attention weights.


