#!/usr/bin/python3

import sys
import os
if os.getenv('RBM_CLUSTERING') != None:
    os.chdir(os.getenv('RBM_CLUSTERING'))
sys.path.append(os.getcwd() + '/src')
from typing import Tuple, Union
from mean_field_tools import *
from datasetRBM import DatasetRBM
from pathlib import Path
import logging
import random
import matplotlib.colors as plt_colors
import torch
import numpy as np
from h5py import File
from sklearn.cluster import DBSCAN
from ete3 import Tree
import time
import json
from tqdm import tqdm
import argparse
import matplotlib
from matplotlib.colors import to_hex

Tensor = torch.Tensor
Array = np.ndarray

# define logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
    )

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_params(filename : str, stamp : Union[str, int], device : torch.device=torch.device("cpu")) -> Tuple[Tensor, Tensor, Tensor]:
    """Returns the parameters of the model at the selected time stamp.

    Args:
        filename (str): filename of the model.
        stamp (Union[str, int]): Epoch.
        device (torch.device): device.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Parameters of the model (vbias, hbias, weigth_matrix).
    """
    stamp = str(stamp)
    key = f"epoch_{stamp}"
    f = File(filename, "r")
    vbias = torch.tensor(f[key]["vbias"][()], device=device)
    hbias = torch.tensor(f[key]["hbias"][()], device=device)
    weight_matrix = torch.tensor(f[key]["weight_matrix"][()], device=device)
    return (vbias, hbias, weight_matrix)

def get_epochs(filename : str) -> Array:
    """Returns the epochs at which the model has been saved.

    Args:
        filename (str): filename of the model.
        stamp (Union[str, int]): Epoch.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: Parameters of the model (vbias, hbias, weigth_matrix).
    """
    f = File(filename, 'r')
    alltime = []
    for key in f.keys():
        if "epoch" in key:
            alltime.append(int(key.replace("epoch_", "")))
    f.close()
    # Sort the results
    alltime = np.sort(alltime)
    return alltime
    
class TreeRBM():
    """
    Class that implements the construction of a tree exploiting the training hystory of an RBM.
    """
    
    def __init__(self, device : torch.device=torch.device('cpu')) -> None:
        """
        Args:
            fname_tree (str) : Path of the RBM model to load.
            device (torch.device) : device where to mount the TreeRBM model.
        """

        self.fname_model = ""
        self.device = device
        self.tree_codes = []
        self.t_ages = []
        self.tree = Tree()
        self.num_labels_families = 1
        self.labels = {}       # dictionary of {label_family : all unique labels}
        self.leaves_names = []
        self.labels_dict = {}
        self.colors_dict = {}
        self.max_depth = 1
        # fit parameters
        self.eps = 0
        self.order = 0
        self.batch_size = 0
        self.alpha = 0

    def save_tree(self, fname : Path) -> None:
        """Save the treeRBM model.

        Args:
            fname (Path): Path of the folder where to save the treeRBM model
        """

        print('\nSaving the tree...\n')

        tree_dict = {
            'fname_model' : str(self.fname_model),
            'tree_codes' : self.tree_codes.tolist(),
            't_ages' : self.t_ages.tolist(),
            'tree' : self.tree.write(),
            'num_labels_families' : self.num_labels_families,
            'labels' : self.labels,
            'leaves_names' : self.leaves_names.tolist(),
            'labels_dict' : self.labels_dict,
            'colors_dict' : self.colors_dict,
            'max_depth' : self.max_depth,
            'folder_name' : str(fname),
            'eps' : self.eps,
            'alpha' : self.alpha,
            'batch_size' : self.batch_size,
            'order' : self.order
        }

        tree_filename = fname / 'tree.json'
        with open(tree_filename, 'w') as f:
            json.dump(tree_dict, f)
    
    def load_tree(self, fname : str) -> None:
        """Load treeRBM model from a file.

        Args:
            fname (str): Path of the treeRBM model to be loaded.
        """

        with open(fname, 'r') as f:
            jf = json.load(f)

        self.fname_model = jf['fname_model']
        f_model = File(self.fname_model, 'r')
        f_model.close()
        self.tree_codes = np.array(jf['tree_codes'])
        self.t_ages = np.array(jf['t_ages'])
        self.tree = Tree(jf['tree'], format=1)
        self.num_labels_families = jf['num_labels_families']
        self.labels = jf['labels']
        self.leaves_names = jf['leaves_names']
        self.labels_dict = jf['labels_dict']
        self.colors_dict = jf['colors_dict']
        self.max_depth = jf['max_depth']
        self.folder_name = jf['folder_name']
        self.eps = jf['eps']
        self.alpha = jf['alpha']
        self.batch_size = jf['batch_size']
        self.order = jf['order']
        
    def get_tree_ages(self, X : Tuple[Tensor, Tensor], min_increase : float=0.1, alpha : float=1e-4) -> Array:
        """Scans all t_ages and returns those that bring an increase in the
        mean-field estimate of the number of fixed points greather than
        the proportion min_increase.

        Args:
            X (Tuple[Tensor, Tensor]): Initial conditions (visible and hidden magnetizations).
            min_increase (float, optional): Fraction of fixed points increase to consider for choosing the ages of the RBM. Defaults to 0.1.
            eps (float, optional): Convergence threshold of the algorithm. Defaults to 1e-4.

        Returns:
            np.ndarray: Array of t_ages for constructing the tree.
        """

        # Get all t_ages saved
        alltime = get_epochs(self.fname_model)
        tree_ages = []
        prev_num_fps = 1

        pbar = tqdm(alltime, colour='red', leave=False, ascii="-#")
        pbar.set_description('Filtering the ages')
        for stamp in pbar:
            params = get_params(self.fname_model, stamp=stamp, device=self.device)            
            X_mf, _ = iterate_mean_field(X=X, params=params, order=1, batch_size=256, alpha=alpha, verbose=False)
            fps = torch.unique(torch.argmax(X_mf, -1), dim=0).cpu().numpy()
            num_fps = len(fps)
            if (len(tree_ages) > 0) and (num_fps == prev_num_fps):
                # keep all t_ages with the same number of fixed points for tracking the dragging of the fixed points
                tree_ages.append(stamp)
            elif (num_fps - prev_num_fps) / num_fps >= min_increase:
                tree_ages.append(stamp)
                prev_num_fps = num_fps
        return np.array(tree_ages)
    
    def fit(self, model : str, data : Tensor, leaves_names : Array, t_ages : Array=None, batch_size : int=500, min_increase : float=0.1, eps : float=1.,
            alpha : float=1e-4, save_node_features : bool=False, order : int=2, max_iter : int=10000) -> dict:
        """Fits the treeRBM model on the data.

        Args:
            model (str): Path to the RBM model.
            data (Tensor): Data to fill the treeRBM model.
            leaves_names (Array): Names of the leaves.
            t_ages (Array, optional): Ages of the RBM at which compute the branches of the tree. If None, t_ages are chosen automatically. Defaults to None.
            batch_size (int, optional): Batch size, to tune based on the memory availability. Defaults to 128.
            min_increase (float, optional): Relative fixed points number that has to change for saving one age. Used only if t_ages is None. Defaults to 0.1.
            eps (float, optional): Epsilon parameter of the DBSCAN. Defaults to 1..
            alpha (float, optional): Convergence threshold of the TAP equations. Defaults to 1e-4.
            save_features_depth (int, optional): Depth below which saving the states at the tree nodes.
            order (int, optional): Order of the mean-field free energy approximation. Defaults to 2.
            max_iter (int, optional): Maximum number of TAP iterations. Defaults to 10000.
        Returns:
            dict : dictionary that associates tree nodes to the fixed points.
        """
        
        self.eps = eps
        self.order = order
        self.alpha = alpha
        self.batch_size = batch_size
        self.leaves_names = leaves_names
        self.fname_model = str(model)

        # get t_ages
        if np.sum(t_ages):
            self.t_ages = t_ages
        else:
            self.t_ages = self.get_tree_ages(data, min_increase=min_increase)

        # initialize the RBM
        params = get_params(self.fname_model, stamp=t_ages[-1])

        # generate tree_codes
        _, mh = sample_hiddens(data, params[1], params[2])
        _, mv = sample_visibles(mh, params[0], params[2])
        mag_state = (mv, mh)
            
        n_data = data.shape[0]
        n_levels = len(self.t_ages)

        tree_codes = np.zeros(shape=(n_data, n_levels), dtype=np.int32)
        scan = DBSCAN(eps=eps, min_samples=1, n_jobs=-1, metric='euclidean')
        old_fixed_points_number = np.inf
        unused_levels = n_levels
        level = n_levels - 1
        
        def evaluate_mask_pipe(mask_pipe, last_classification):
            """Function that is used to propagate the classification of the representative fixed points at a certain
            age to all the data points.
            """
            classification1 = last_classification
            for mask_matrix in reversed(mask_pipe):
                classification2 = np.zeros(mask_matrix.shape[1])
                for i, mask_row in enumerate(mask_matrix):
                    classification2[mask_row] = classification1[i]
                classification1 = classification2

            return classification1
        
        mask_pipe = []
        levels_temp = []
        labels_temp = []
        fps_temp = []

        pbar = tqdm(total=len(self.t_ages), colour='red', leave=False, dynamic_ncols=True, ascii='-#')
        pbar.set_description('Generating tree codes')
        for t_age in reversed(self.t_ages):
            pbar.update(1)

            # load the rbm
            params = get_params(self.fname_model, stamp=t_age)

            # Iterate TAP equations until convergence
            n = len(mag_state[0])
            mag_state = iterate_mean_field(X=mag_state, params=params, order=order, batch_size=batch_size, alpha=alpha, max_iter=max_iter, verbose=False, device=self.device)

            # Clustering with DBSCAN
            scan.fit(mag_state[1].cpu())
            unique_labels = np.unique(scan.labels_)
            new_fixed_points_number = len(unique_labels)
            
            # select only a representative for each cluster and propagate the new classification up to the first layer
            mask_matrix = np.ndarray((0, n))
            representative_list = [[], []]
            for l in unique_labels:
                mask = (l == scan.labels_)
                for i, mag in enumerate(mag_state):
                    representative_list[i].append(mag[mask][0].unsqueeze(0))
                mask_matrix = np.append(mask_matrix, [mask], axis=0)
            for i in range(len(representative_list)):
                representative_list[i] = torch.cat(representative_list[i], dim=0).to(self.device)
            mask_pipe.append(mask_matrix.astype(np.bool_))
            mag_state = representative_list
            level_classification = evaluate_mask_pipe(mask_pipe, unique_labels)

            # add the new classification only if the number of TAP fixed points has decreased
            if new_fixed_points_number < old_fixed_points_number:
                tree_codes[:, level] = level_classification
                unused_levels -= 1
                old_fixed_points_number = new_fixed_points_number
                
                if save_node_features:
                    for lab, fp in zip(unique_labels, mag_state[0]):
                        labels_temp.append(lab)
                        levels_temp.append(level)
                        fps_temp.append(fp.cpu().numpy())
                level -= 1       
        pbar.close()
        
        # Subtract from the level index the number of unused levels
        levels_temp = np.array(levels_temp) - unused_levels
        # Construct the node features dictionary
        node_features_dict = {f'I{level}-{lab}' : fp for level, lab, fp in zip(levels_temp, labels_temp, fps_temp)}

        tree_codes = tree_codes[:, unused_levels:]
        self.tree_codes = tree_codes
        self.max_depth = tree_codes.shape[1]
        return node_features_dict

    def generate_tree(self, labels_dict : list=None, colors_dict : list=None, depth : int=None, write_to_file : bool=False):
        """Constructs an ete3.Tree objects with the previously fitted data.

        Args:
            labels_dict (list, optional): Dictionaries of the kind {leaf_name : leaf_label} with the labels to assign to the leaves. Defaults to None.
            colors_dict (list, optional): Dictionaries with a mapping {label : colour}. Defaults to None.
            depth (int, optional): Maximum depth of the tree. If None, all levels are used. Defaults to None.
            write_to_file (bool, optional): Wheather to save the results on a file.
        """
        
        self.labels_dict = labels_dict
        self.colors_dict = colors_dict

        # Validate input arguments
        if labels_dict:
            if (type(labels_dict) != list) and (type(labels_dict) != tuple):
                labels_dict = [labels_dict]

            if colors_dict:
                if (type(colors_dict) != list) and (type(colors_dict) != tuple):
                    colors_dict = [colors_dict]
                assert(len(colors_dict) == len(labels_dict)), 'colors_dict must have the same length of labels_dict'

        if depth:
            assert(depth <= self.tree_codes.shape[1]), 'depth parameter should be <= than the tree depth'
            n_levels = depth
        else:
            n_levels = self.tree_codes.shape[1]

        # Initialize the tree with the proper number of initial branches
        n_start_branches = np.max(self.tree_codes[:, 0]) + 1
        init_tree = '('
        for i in range(n_start_branches):
            init_tree += 'R{0}-:1,'.format(i)
        init_tree = init_tree[:-1] + ')Root:1;'
        t = Tree(init_tree, format=1)
        for n in t.traverse():
            t.add_feature('level', 0)

        # Build the tree structure
        for level in range(2, n_levels + 1):
            tree_lvl = np.unique(self.tree_codes[:, :level], axis=0)
            for lvl in tree_lvl:
                leaf_name = 'R' + ''.join([str(aa) + '-' for aa in lvl])
                mother_name = 'R' + ''.join([str(aa) + '-' for aa in lvl[:-1]])
                M = t.search_nodes(name=mother_name)[0]
                M.add_child(name=leaf_name)
                C = M.search_nodes(name=leaf_name)[0]
                M.add_feature('level', level - 1)
                C.add_feature('level', level)

        # Add all leaves to the tree
        for tree_node, leaf_name in zip(self.tree_codes, self.leaves_names):
            mother_name = 'R' + ''.join([str(aa) + '-' for aa in tree_node[:n_levels]])
            M = t.search_nodes(name=mother_name)[0]
            M.add_child(name=leaf_name)

        # add labels to the leaves
        if labels_dict:
            for i, ld_raw in enumerate(labels_dict):
                # remove '-1' if present
                ld = {k : v for k, v in ld_raw.items() if v != '-1'}
                
                if colors_dict:
                    colors_dict[i] = {l : c for l, c in colors_dict[i].items() if l != '-1'} # remove '-1' if present
                    leaves_colors = [colors_dict[i][label] for label in ld.values()]
                    # create annotation file for iTOL
                    f = open(f'{self.folder_name}/leaves_colours{str(i)}{str(i)}.txt', 'w')
                    f.write('DATASET_COLORSTRIP\nSEPARATOR TAB\nDATASET_LABEL\tLabel family ' + str(i) + '\nCOLOR\tred\n')
                    f.write('LEGEND_TITLE\tLabel family {0}\nSTRIP_WIDTH\t75'.format(i))
                    f.write('\nLEGEND_SHAPES')
                    for _ in colors_dict[i].keys():
                        f.write('\t2')
                    f.write('\nLEGEND_LABELS')
                    for l in colors_dict[i].keys():
                        l = l
                        f.write(f'\t{l}')
                    f.write('\nLEGEND_COLORS')
                    for c in colors_dict[i].values():
                        f.write(f'\t{c}')
                    f.write('\nDATA\n')
                    
                    for leaf_name, leaf_color, label in zip(ld.keys(), leaves_colors, ld.values()):
                        leaf = t.search_nodes(name=leaf_name)[0]
                        rgba_colors_norm = plt_colors.to_rgba(leaf_color, 1.)
                        rgba_colors = tuple(int(nc * 255) if i != 3 else nc for i, nc in enumerate(rgba_colors_norm))
                        f.write(leaf.name + '\trgba' + str(rgba_colors).replace(' ', '') + '\t' + str(label) + '\n')
                    f.close()
        
        # rename the nodes of the tree
        def get_node_name(node_id):
            split = [s for s in node_id.split('-') if s != '']
            level = len(split) - 1
            label = split[-1].replace('R', '')

            return f'I{level}-{label}'
                    
        for node in t.traverse():
            if not node.is_leaf():
                if node.name != 'Root':
                    node.name = get_node_name(node.name)

        self.tree = t
        
        # generate nw file
        if write_to_file:
            self.tree.write(format=1, outfile=f'{self.folder_name}/tree.nw')

def create_parser():
    parser = argparse.ArgumentParser(description='Generates the hierarchical tree of a dataset using the specified RBM model.')
    parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')
    
    required.add_argument('-m', '--model',         type=Path, help='Path to RBM model.', required=True)
    required.add_argument('-o', '--output',        type=Path, help='Path to output directory.', required=True)
    required.add_argument('-d', '--data',          type=Path, help='Path to data.', required=True)
    required.add_argument('-a', '--annotations',   type=Path, help='Path to the csv annotation file.', required=True)
    
    optional.add_argument('--n_data',                 type=int,   default=500,        help='(Defaults to 500). Number of data to put in the tree.')
    optional.add_argument('--batch_size',             type=int,   default=500,        help='(Defaults to 500). Batch size.')
    optional.add_argument('--filter_ages', '-f',      action='store_true',   default=False,      help='If specified, filters the ages with the naive MF equations.')
    optional.add_argument('--max_age',                type=int,   default=np.inf,     help='(Defaults to inf). Maximum age to consider for the tree construction.')
    optional.add_argument('--save_node_features',     action='store_true',  default=False,      help='If specified, saves the states corresponding to the tree nodes.')
    optional.add_argument('--max_iter',               type=int,   default=10000,      help='(Defaults to 10000). Maximum number of TAP iterations.')
    optional.add_argument('--max_depth',              type=int,   default=50,         help='(Defaults to 50). Maximum depth to visualize in the generated tree.')
    optional.add_argument('--order_mf',               type=int,   default=2,          help='(Defaults to 2). Mean-field order of the Plefka expansion.', choices=[1, 2, 3])
    optional.add_argument('--min_increase',           type=float, default=0.1,        help='(Defaults to 0.1). Relative fixed points number that has to change for saving one age. Used only if `filter_ages` is specified.')
    optional.add_argument('--eps',                    type=float, default=1.,         help='(Defaults to 1.). Epsilon parameter of the DBSCAN.')
    optional.add_argument('--alpha',                  type=float, default=1e-4,       help='(Defaults to 1e-4). Convergence threshold of the TAP equations.')
    optional.add_argument('--colormap',               type=str,   default='tab20',   help='(Defaults to `Paired`). Name of the colormap to use for the labels.')
    optional.add_argument('--no_newick',              action='store_false',   default=True,      help='If specified, suppresses the automatic generation of the newick file.')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    args.output.mkdir(exist_ok=True)

    # Select device and RBM model
    start = time.time()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")
    
    # Set random seeds
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # Initialize the tree
    tree = TreeRBM(device=device)
    
    # Load the data
    logger.info('Loading the data')
    data_type = torch.float32
    dataset = DatasetRBM(data_path=args.data, ann_path=args.annotations)
    data = torch.tensor(dataset.data[:args.n_data], device=device).type(data_type)
    leaves_names = dataset.names[:args.n_data]
    #print([{n : l for n, l in dl.items()} for dl in dataset.labels])
    labels_dict = [{n : l for n, l in dl.items() if n in leaves_names} for dl in dataset.labels]
    
    # Fit the tree to the data
    if args.filter_ages:
        t_ages = None
    else:
        alltime = get_epochs(args.model)
        t_ages = alltime[alltime <= args.max_age]
    logger.info('Fitting the model')
    node_features_dict = tree.fit(args.model, data, leaves_names, batch_size=args.batch_size, t_ages=t_ages, save_node_features=args.save_node_features, min_increase=args.min_increase, eps=args.eps, alpha=args.alpha, max_iter=args.max_iter, order=args.order_mf)

    # Save the tree codes
    logger.info(f'Saving the model in {args.output}')
    tree.folder_name = str(args.output)
    tree.save_tree(args.output)
    
    # Create the file with the node states
    if args.save_node_features:
        f_nodes = File(args.output / 'node_features.h5', 'w')
        for state_name, state in node_features_dict.items():
            level = int(state_name.split('-')[0].replace('I', ''))
            if level < args.max_depth:
                f_nodes[state_name] = state
        f_nodes.close()
    
    # Generate the tree
    if args.no_newick:
        logger.info(f'Generating a tree of depth {min(args.max_depth, tree.max_depth)}. Maximum depth is {tree.max_depth}')
        colors = matplotlib.colormaps[args.colormap]
        colors_dict = [{l : to_hex(colors(i)) for i, l in enumerate(np.unique(list(labels.values())))} for labels in labels_dict]
        if args.max_depth > tree.max_depth:
            args.max_depth = tree.max_depth
        tree.generate_tree(labels_dict=labels_dict, colors_dict=colors_dict, depth=args.max_depth, write_to_file=True)

    stop = time.time()
    logger.info(f'Process completed, elapsed time: {round((stop - start) / 60, 1)} minutes')
    sys.exit(0)