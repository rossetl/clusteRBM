# clusteRBM
[![PRE](https://img.shields.io/badge/PhysRevE-108,014110-red.svg)](https://scholar.google.com/citations?view_op=view_citation&hl=it&user=__OKD-kAAAAJ&sortby=pubdate&citation_for_view=__OKD-kAAAAJ:EPG8bYD4jVwC)

Code for the paper "Unsupervised hierarchical clustering using the learning dynamics of RBMs" by Aurélien Decelle, Lorenzo Rosset and Beatriz Seoane.

<p align="center">
<image src="/images/tree-MNIST.png" width=456 height=400/>
<p align="center">
<em>Example of hierarchical clustering obtained on the MNIST dataset. Image taken from the paper.</em>
</p>
</p>

## Installation
- Create the conda environment with all the dependencies: 
```
conda env create -f RBMenv.yml
```

- Include the main directory to your PATH environment variable by adding the following line to your ~/.bashrc file:
```
RBM_CLUSTERING=/path/to/RBM-hierarchical-clustering
```

- Create some repositories for storing the data and the programs' outputs
```
mkdir data models trees
```

## Usage

### Train an RBM model
The possible RBM models are denoted as:
- `BernoulliBernoulliRBM`: Bernoulli variables in both the visible and the hidden layer;
- `BernoulliBernoulliWeightedRBM`: Bernoulli variables in both the visible and the hidden layer. During the training, the averages over the data points are weighted;
- `PottsBernoulliRBM`: Potts variables in the visible layer and Bernoulli variables in the hidden layer;
- `PottsBernoulliWeightedRBM`: Potts variables in the visible layer and Bernoulli variables in the hidden layer. During the training, the averages over the data points are weighted.

To train a specific RBM model launch `rbm-train` followed by the proper arguments:
- *default*: trains a `BernoulliBernoulliRBM` model;
- `-V`: selects the visible layer variables type. Followed by `Bernoulli` trains a `BernoulliBernoulliRBM` model, while followed by `Potts` trains a `PottsBernoulliRBM` model;
- `-w`: trains the weighted version of the previous models;
- To open the help page type `rbm-train -h`.

Apart from the previous parameters, another series of parameters specifies the training specifics. To list all the possibilities, use the argument `-i` (e.g. `rbm-train -V -w -i`).

The script will ask to select the data file to be used for the training among those present in the folder `data/` (for details on the training data file format see the next section).

#### Data source format

## TreeRBM
Once you have a trained model in the folder `models/`, you can use the `rbm-maketree` command to generate the hierarchical tree for a dataset compatible with the one used for the training of the RBM. Use `rbm-maketree -h` to list all the optional arguments.

The script will output a repository in the `trees/` folder with a name referring to the model used for generating the tree. Inside the repository, there will be a file `tree.json` containing the information about the TreeRBM object created and (optionally) a `node_features.h5` file containing the features at each node of the tree.

To generate a newick file of the tree and the corresponding iTOL annotation files, the TreeRBM method `generate_tree` has to be called with a proper list of arguments. See the example notebook for the usage.

## Example data
