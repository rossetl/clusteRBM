#!/usr/bin/python3

import sys
import matplotlib
from matplotlib.colors import to_hex
from pathlib import Path
import random
import importlib
import numpy as np
import time
import argparse
import torch
from h5py import File
import logging

from treerbm.dataset import DatasetRBM
from treerbm.ioRBM import get_epochs
from treerbm.treeRBM import fit, generate_tree
from treerbm.branch_metrics import l2_dist


def create_parser():
    parser = argparse.ArgumentParser(description='Generates the hierarchical tree of a dataset using the specified RBM model.')
    parser._action_groups.pop()
    required = parser.add_argument_group('Required arguments')
    optional = parser.add_argument_group('Optional arguments')
    
    required.add_argument('-m', '--model',         type=Path, help='Path to RBM model.', required=True)
    required.add_argument('-o', '--output',        type=Path, help='Path to output directory.', required=True)
    required.add_argument('-d', '--data',          type=Path, help='Path to data.', required=True)
    
    optional.add_argument('-a', '--annotations',      type=Path,            default=None,             help='Path to the csv annotation file.',)
    optional.add_argument('-c', '--colors',           type=Path,            default=None,             help='Path to the csv color mapping file.')
    optional.add_argument('-f', '--filter',           type=float,           default=None,             help='(defaults to None). Selects a subset of epochs such that the acceptance rate of swapping two adjacient configurations is the one specified.')
    optional.add_argument("--alphabet",               type=str,             default="protein",        help="(Defaults to protein). Type of encoding for the sequences. Choose among ['protein', 'rna', 'dna'] or a user-defined string of tokens.")
    optional.add_argument('--n_data',                 type=int,             default=500,              help='(Defaults to 500). Number of data to put in the tree.')
    optional.add_argument('--batch_size',             type=int,             default=500,              help='(Defaults to 500). Batch size.')
    optional.add_argument('--max_age',                type=int,             default=np.inf,           help='(Defaults to inf). Maximum age to consider for the tree construction.')
    optional.add_argument('--save_node_features',     action='store_true',  default=False,            help='If specified, saves the states corresponding to the tree nodes.')
    optional.add_argument('--max_iter',               type=int,             default=10000,            help='(Defaults to 10000). Maximum number of TAP iterations.')
    optional.add_argument('--max_depth',              type=int,             default=50,               help='(Defaults to 50). Maximum depth to visualize in the generated tree.')
    optional.add_argument('--order_mf',               type=int,             default=2,                help='(Defaults to 2). Mean-field order of the Plefka expansion.', choices=[1, 2, 3])
    optional.add_argument('--eps',                    type=float,           default=1.,               help='(Defaults to 1.). Epsilon parameter of the DBSCAN.')
    optional.add_argument('--alpha',                  type=float,           default=1e-4,             help='(Defaults to 1e-4). Convergence threshold of the TAP equations.')
    optional.add_argument('--colormap',               type=str,             default='tab20',          help='(Defaults to `tab20`). Name of the colormap to use for the labels.')
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
    dataset = DatasetRBM(data_path=args.data, ann_path=args.annotations, colors_path=args.colors, alphabet=args.alphabet)
    num_states = int(dataset.get_num_states())
    
    if num_states > 2:
        data_type = torch.int64
        module = importlib.import_module("tools.tools_categorical")
    else:
        data_type = torch.float32
        module = importlib.import_module("tools.tools_binary")
        
    data = torch.tensor(dataset.data[:args.n_data], device=device).type(data_type)
    leaves_names = dataset.names[:args.n_data]
    labels_dict = [{n : l for n, l in dl.items() if n in leaves_names} for dl in dataset.labels]
    
    # Fit the tree to the data
    alltime = get_epochs(args.model)
    t_ages = alltime[alltime <= args.max_age]
    logger.info('Fitting the model')
    tree_codes, node_features_dict = fit(
        module=module,
        fname=args.model,
        data=data,
        batch_size=args.batch_size,
        t_ages=t_ages,
        num_states=num_states,
        eps=args.eps,
        alpha=args.alpha,
        max_iter=args.max_iter,
        order=args.order_mf,
        filter_ages=args.filter,
        device=device
        )
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
        
    generate_tree(
        tree_codes=tree_codes,
        leaves_names=leaves_names,
        legend=dataset.legend,
        folder=args.output,
        labels_dict=labels_dict,
        colors_dict=colors_dict,
        depth=args.max_depth,
        node_features_dict=node_features_dict,
        dist_fn=l2_dist
    )

    stop = time.time()
    logger.info(f'Process completed, elapsed time: {round((stop - start) / 60, 1)} minutes')
    sys.exit(0)