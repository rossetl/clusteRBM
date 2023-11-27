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

- Include the main directory to your PATH environment variable by adding the following line to your ~/.bashrc file:
```
RBM_CLUSTERING=/path/to/RBM-hierarchical-clustering
```

- Create some repositories for storing the data and the programs' outputs
```
mkdir data models trees
```

## Usage
#### Data source format

## TreeRBM
Once you have a trained model in the folder `models/`, you can use the `rbm-maketree` command to generate the hierarchical tree for a dataset compatible with the one used for the training of the RBM. Use `rbm-maketree -h` to list all the optional arguments.

The script will output a repository in the `trees/` folder with a name referring to the model used for generating the tree. Inside the repository, there will be a file `tree.json` containing the information about the TreeRBM object created and (optionally) a `node_features.h5` file containing the features at each node of the tree.

To generate a newick file of the tree and the corresponding iTOL annotation files, the TreeRBM method `generate_tree` has to be called with a proper list of arguments. See the example notebook for the usage.

## Example data
