# Next Day Stock Market Price Direction Prediction Task

EE 559 - Machine Learning 1

By Sungje Park

*****Note***** Due to size limitations on github, the cleaned training data .pickle files in /datasets are not present. This dataset can be recreated using the data_prrocessing.ipynb file.

## Running the Code
To reproduce all results for training, simply run the data_processing.ipynb to create the cleaned dataset, then run the respective notebooks, Baseline Neural Network (baseline_nn.ipynb), decision tree model (decision_tree.ipynb), and attention-based models (transformer_model.ipynb).

The core model archetectures for the neural network based methods are located in models.py.

The codebase uses Python JAX at its core with various supporting libraries. A full list of requirements can be found in requirements.txt.