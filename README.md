# Implementing ANNs with TensorFlow: Group 11

IANNwTF Seminar @uos Winter Term 2021/2022

## Table of Contents
* [General Info](#general-info)
* [Pipeline](#git-setup)

## General Info
This project is the reimplementation of ["Can GAN originate new electronic dance music genres?—Generating novel
rhythm patterns using GAN with Genre Ambiguity Loss"](https://arxiv.org/pdf/2011.13062.pdf) paper by Nao Tokui (2020, November 25).  

The Groove MIDI Dataset is available for download [here](https://magenta.tensorflow.org/datasets/groove#format).

## Pipeline
To run the pipeline, specific python libraries are required. You can find them
in `create_environment.sge`. Alternative just run the script from a shell. This 
will create `conda` environment with all necessary requirements.

To execute the pipeline, move to the `src/`. Here you have three different
options, to start the pipeline:
1. Run `../grid_search.sh local` for a local execution of the grid search. This
   will start all different hyperparameter trainings after another. This will
   take quite long. With 10 epochs each ~2h of computing.
2. Run `../grid_search.sh local` for a grid execution of the grid search. This
   will start the grid search. Here all hyperparameter train simultaneously when
   used on the grid.
3. The last method is for single parameter use. Just run the `main.py` file. You
   can run it with different flags, to specify the behaviour. However, it needs
   you to specify the optimiser. Flags:
    * `-e` or `--epochs` Number of epochs to run
    * `-v` or `--visulize` Visualize drum matrices between epochs
    * `-t` or `--second-training` Disables the second training 
    * `--RMSProp` Enables optimiser RMSProp followed by a learning rate
    * `--SGD` Enables optimiser SGD followed by a learning rate
    * `--adam` Enables optimiser Adam followed by a learning rate

## MLflow
To see the results of the grid computing run in the profect folder:
`mlflow ui --backend-store-uri data/mlflow` and navigate in your browser to 
`http://127.0.0.1:5000` to see the data.
