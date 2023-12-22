# clusteRBM
[![PRE](https://img.shields.io/badge/PhysRevE-108,014110-red.svg)](https://scholar.google.com/citations?view_op=view_citation&hl=it&user=__OKD-kAAAAJ&sortby=pubdate&citation_for_view=__OKD-kAAAAJ:EPG8bYD4jVwC)

Code for the paper "Unsupervised hierarchical clustering using the learning dynamics of RBMs" by Aurélien Decelle, Lorenzo Rosset and Beatriz Seoane.

ClusteRBM generates a ralational tree of some input dataset using a previously trained RBM model.

<p align="center">
<image src="/images/tree-MNIST.png" width=456 height=400/>
<p align="center">
<em>Example of hierarchical clustering obtained on the MNIST dataset. Image taken from the paper.</em>
</p>
</p>

## Installation

- Include the main directory to your PATH environment variable by adding the following line to your ~/.bashrc file:
```
CLUSTERBM=/path/to/clusteRBM
```

- (Optional) Create some repositories for storing the data and the programs' outputs
```
mkdir data models trees
```

## Usage

The code for training an RBM and save the model in the correct format can be found in [TorchRBM](https://github.com/AurelienDecelle/TorchRBM.git). To obtain a proper tree reconstruction, it is important to save the parameters of the model many times during the training at evenly spaced time intervals. We therefore suggest to set `--n_save` $\geq$ 100 and `--spacing linear`.

Once you have a properly trained RBM model, to generate the tree enter:
```bash
./clusterbm -m <path_to_rbm_model> -d <path_to_data> -a <path_to_annotations> -o <output_folder>
```
This will create a folder called `output_folder` containing the tree in newick format and the annotation files to be given to [iTOL](https://itol.embl.de/).

The list of available argument can be printed using:
```bash
./clusterbm -h
```

### Options

- 

## TreeRBM
Once you have a trained model in the folder `models/`, you can use the `rbm-maketree` command to generate the hierarchical tree for a dataset compatible with the one used for the training of the RBM. Use `rbm-maketree -h` to list all the optional arguments.

The script will output a repository in the `trees/` folder with a name referring to the model used for generating the tree. Inside the repository, there will be a file `tree.json` containing the information about the TreeRBM object created and (optionally) a `node_features.h5` file containing the features at each node of the tree.

To generate a newick file of the tree and the corresponding iTOL annotation files, the TreeRBM method `generate_tree` has to be called with a proper list of arguments. See the example notebook for the usage.

## Example data
