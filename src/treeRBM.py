#!/usr/bin/python3

import sys
import os
if os.getenv('CLUSTERBM') != None:
    os.chdir(os.getenv('CLUSTERBM'))
sys.path.append(os.getcwd() + '/src')
from typing import Tuple, Union
import matplotlib
import matplotlib.colors as plt_colors
from matplotlib.colors import to_hex
from pathlib import Path
import random
import importlib
import numpy as np
import time
import argparse

import torch
from h5py import File
from sklearn.cluster import DBSCAN
from ete3 import Tree
from tqdm import tqdm
import logging
from dataset import DatasetRBM

Tensor = torch.Tensor
Array = np.ndarray

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
    
def fit(module,
        fname : str,
        data : Tensor,
        t_ages : Array,
        batch_size : int=500,
        eps : float=1.,
        alpha : float=1e-4,
        save_node_features : bool=False,
        order : int=2,
        max_iter : int=10000,
        device : torch.device=torch.device("cpu")) -> Tuple[Array, dict]:
    """Fits the treeRBM model on the data.
    
    Args:
        module: module containing the mean-field methods.
        fname (str): Path to the RBM model.
        data (Tensor): Data to fill the treeRBM model.
        t_ages (Array): Ages of the RBM at which compute the branches of the tree.
        batch_size (int, optional): Batch size, to tune based on the memory availability. Defaults to 128.
        eps (float, optional): Epsilon parameter of the DBSCAN. Defaults to 1..
        alpha (float, optional): Convergence threshold of the TAP equations. Defaults to 1e-4.
        save_node_features (bool, optional): Wheather to save the states (fixed points) at the tree nodes.
        order (int, optional): Order of the mean-field free energy approximation. Defaults to 2.
        max_iter (int, optional): Maximum number of TAP iterations. Defaults to 10000.
        device (torch.device): Device.
        
    Returns:
        Tuple[Array, dict] : Array with the encoded tree structure, dictionary that associates tree nodes to the fixed points.
    """
        
    # initialize the RBM
    params = get_params(fname, stamp=t_ages[-1], device=device)
    
    # generate tree_codes
    mh = module.profile_hiddens(data, params[1], params[2])
    mv = module.profile_visibles(mh, params[0], params[2])
    mag_state = (mv, mh)
        
    n_data = data.shape[0]
    n_levels = len(t_ages) # depth of the tree
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
    pbar = tqdm(total=n_levels, colour='red', leave=False, dynamic_ncols=True, ascii='-#')
    pbar.set_description('Generating tree codes')
    for t_age in reversed(t_ages):
        pbar.update(1)
        # load the rbm parameters
        params = get_params(fname, stamp=t_age, device=device)
        # Iterate mean field equations until convergence
        n = len(mag_state[0])
        mag_state = module.iterate_mean_field(X=mag_state, params=params,
                                              order=order, batch_size=batch_size,
                                              alpha=alpha, rho=0., max_iter=max_iter, verbose=False, device=device)
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
            representative_list[i] = torch.cat(representative_list[i], dim=0).to(device)
        mask_pipe.append(mask_matrix.astype(np.bool_))
        mag_state = representative_list
        level_classification = evaluate_mask_pipe(mask_pipe, unique_labels)
        
        # Add the new classification only if the number of TAP fixed points has decreased
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
    return (tree_codes, node_features_dict)

def generate_tree(tree_codes : Array,
                  folder : str,
                  leaves_names : Array,
                  legend : list=None,
                  labels_dict : list=None,
                  colors_dict : list=None,
                  depth : int=None) -> None:
    """Constructs an ete3.Tree objects with the previously fitted data.
    
    Args:
        tree_codes (Array): Array encoding the tree.
        folder (str): Path to the folder where to store the data.
        leaves_names (Array): List of names of all the leaves in the tree.
        legend (list, optional): List with the names to assign to the legend titles. Defaults to None.
        labels_dict (list, optional): Dictionaries of the kind {leaf_name : leaf_label} with the labels to assign to the leaves. Defaults to None.
        colors_dict (list, optional): Dictionaries with a mapping {label : colour}. Defaults to None.
        depth (int, optional): Maximum depth of the tree. If None, all levels are used. Defaults to None.
    """
    # Validate input arguments
    if labels_dict:
        if (type(labels_dict) != list) and (type(labels_dict) != tuple):
            labels_dict = [labels_dict]
        if colors_dict:
            if (type(colors_dict) != list) and (type(colors_dict) != tuple):
                colors_dict = [colors_dict]
            assert(len(colors_dict) == len(labels_dict)), 'colors_dict must have the same length of labels_dict'
    max_depth = tree_codes.shape[1]
    if depth:
        assert(depth <= max_depth), 'depth parameter should be <= than the tree depth'
        n_levels = depth
    else:
        n_levels = max_depth
        
    # Initialize the tree with the proper number of initial branches
    n_start_branches = np.max(tree_codes[:, 0]) + 1
    init_tree = '('
    for i in range(n_start_branches):
        init_tree += 'R{0}-:1,'.format(i)
    init_tree = init_tree[:-1] + ')Root:1;'
    t = Tree(init_tree, format=1)
    for n in t.traverse():
        t.add_feature('level', 0)
        
    # Build the tree structure
    for level in range(2, n_levels + 1):
        tree_lvl = np.unique(tree_codes[:, :level], axis=0)
        for lvl in tree_lvl:
            leaf_name = 'R' + ''.join([str(aa) + '-' for aa in lvl])
            mother_name = 'R' + ''.join([str(aa) + '-' for aa in lvl[:-1]])
            M = t.search_nodes(name=mother_name)[0]
            M.add_child(name=leaf_name)
            C = M.search_nodes(name=leaf_name)[0]
            M.add_feature('level', level - 1)
            C.add_feature('level', level)
    # Add all leaves to the tree
    for tree_node, leaf_name in zip(tree_codes, leaves_names):
        mother_name = 'R' + ''.join([str(aa) + '-' for aa in tree_node[:n_levels]])
        M = t.search_nodes(name=mother_name)[0]
        M.add_child(name=leaf_name)
    # Add labels to the leaves
    if labels_dict:
        for i, ld_raw in enumerate(labels_dict):
            # Remove '-1' if present
            ld = {k : v for k, v in ld_raw.items() if v != '-1'}
            
            if colors_dict:
                colors_dict[i] = {l : c for l, c in colors_dict[i].items() if l != '-1'} # remove '-1' if present
                leaves_colors = [colors_dict[i][label] for label in ld.values()]
                # Create annotation file for iTOL
                if legend is not None:
                    f = open(f'{folder}/leaves_colours_{legend[i]}.txt', 'w')
                    f.write('DATASET_COLORSTRIP\nSEPARATOR TAB\nDATASET_LABEL\t{0}'.format(legend[i]) + '\nCOLOR\tred\n')
                    f.write('LEGEND_TITLE\t{0}\nSTRIP_WIDTH\t75'.format(legend[i]))
                else:
                    f = open(f'{folder}/leaves_colours_{str(i)}.txt', 'w')
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
    
    # Rename the nodes of the tree
    def get_node_name(node_id):
        split = [s for s in node_id.split('-') if s != '']
        level = len(split) - 1
        label = split[-1].replace('R', '')
        return f'I{level}-{label}'
                
    for node in t.traverse():
        if not node.is_leaf():
            if node.name != 'Root':
                node.name = get_node_name(node.name)
    
    # Generate newick file
    t.write(format=1, outfile=f'{folder}/tree.nw')

def create_parser():
    parser = argparse.ArgumentParser(description='Generates the hierarchical tree of a dataset using the specified RBM model.')
    parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')
    
    required.add_argument('-m', '--model',         type=Path, help='Path to RBM model.', required=True)
    required.add_argument('-o', '--output',        type=Path, help='Path to output directory.', required=True)
    required.add_argument('-d', '--data',          type=Path, help='Path to data.', required=True)
    required.add_argument('-a', '--annotations',   type=Path, help='Path to the csv annotation file.', required=True)
    
    optional.add_argument('-c', '--colors',           type=Path,  default=None,       help='Path to the csv color mapping file.')
    optional.add_argument('--n_data',                 type=int,   default=500,        help='(Defaults to 500). Number of data to put in the tree.')
    optional.add_argument('--batch_size',             type=int,   default=500,        help='(Defaults to 500). Batch size.')
    optional.add_argument('--max_age',                type=int,   default=np.inf,     help='(Defaults to inf). Maximum age to consider for the tree construction.')
    optional.add_argument('--save_node_features',     action='store_true',  default=False,      help='If specified, saves the states corresponding to the tree nodes.')
    optional.add_argument('--max_iter',               type=int,   default=10000,      help='(Defaults to 10000). Maximum number of TAP iterations.')
    optional.add_argument('--max_depth',              type=int,   default=50,         help='(Defaults to 50). Maximum depth to visualize in the generated tree.')
    optional.add_argument('--order_mf',               type=int,   default=2,          help='(Defaults to 2). Mean-field order of the Plefka expansion.', choices=[1, 2, 3])
    optional.add_argument('--eps',                    type=float, default=1.,         help='(Defaults to 1.). Epsilon parameter of the DBSCAN.')
    optional.add_argument('--alpha',                  type=float, default=1e-4,       help='(Defaults to 1e-4). Convergence threshold of the TAP equations.')
    optional.add_argument('--colormap',               type=str,   default='tab20',    help='(Defaults to `tab20`). Name of the colormap to use for the labels.')
    return parser

if __name__ == '__main__':
    # Define logger
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

    parser = create_parser()
    args = parser.parse_args()
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    args.output.mkdir(exist_ok=True)

    # Select device
    start = time.time()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")
    
    # Set random seeds
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    # Load the data and the module
    logger.info('Loading the data')
    dataset = DatasetRBM(data_path=args.data, ann_path=args.annotations, colors_path=args.colors)
    
    if dataset.get_num_states() > 2:
        data_type = torch.int64
        module = importlib.import_module("mf_tools.mf_tools_categorical")
    else:
        data_type = torch.float32
        module = importlib.import_module("mf_tools.mf_tools_binary")
        
    data = torch.tensor(dataset.data[:args.n_data], device=device).type(data_type)
    leaves_names = dataset.names[:args.n_data]
    labels_dict = [{n : l for n, l in dl.items() if n in leaves_names} for dl in dataset.labels]
    
    # Fit the tree to the data
    alltime = get_epochs(args.model)
    t_ages = alltime[alltime <= args.max_age]
    logger.info('Fitting the model')
    tree_codes, node_features_dict = fit(module=module, fname=args.model, data=data, batch_size=args.batch_size,
                                        t_ages=t_ages, save_node_features=args.save_node_features,
                                        eps=args.eps, alpha=args.alpha, max_iter=args.max_iter,
                                        order=args.order_mf, device=device)
    max_depth = tree_codes.shape[1]
    # Save the tree codes
    logger.info(f'Saving the model in {args.output}')
    
    # Create the file with the node states
    if args.save_node_features:
        f_nodes = File(args.output / 'node_features.h5', 'w')
        for state_name, state in node_features_dict.items():
            level = int(state_name.split('-')[0].replace('I', ''))
            if level < args.max_depth:
                f_nodes[state_name] = state
        f_nodes.close()
    
    # Generate the tree
    logger.info(f'Generating a tree of depth {min(args.max_depth, max_depth)}. Maximum depth is {max_depth}.')
    if args.colors is not None:
        colors_dict = dataset.colors
    else:
        colors = matplotlib.colormaps[args.colormap]
        colors_dict = [{l : to_hex(colors(i)) for i, l in enumerate(np.unique(list(labels.values())))} for labels in labels_dict]
    if args.max_depth > max_depth:
        args.max_depth = max_depth
    generate_tree(tree_codes=tree_codes, leaves_names=leaves_names, legend=dataset.legend, folder=args.output, labels_dict=labels_dict, colors_dict=colors_dict, depth=args.max_depth)

    stop = time.time()
    logger.info(f'Process completed, elapsed time: {round((stop - start) / 60, 1)} minutes')
    sys.exit(0)