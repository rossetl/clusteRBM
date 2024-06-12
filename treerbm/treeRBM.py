from typing import Tuple
import matplotlib.colors as plt_colors
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from ete3 import Tree
from tqdm import tqdm

from treerbm.ioRBM import get_params
from treerbm.branch_metrics import l2_dist

Tensor = torch.Tensor
Array = np.ndarray

# Function to find a node by name
def find_node_by_name(tree, name):
    for node in tree.traverse():
        if node.name == name:
            return node
    return None

def set_tree_lengths(
    t : Tree,
    nodes_list : list,
    fp : dict,
    dist_fn
):
    new_nodes_list = []
    for p_name in nodes_list:
        p = find_node_by_name(t, p_name)
        children = p.get_children()
        for c in children:
            if c.is_leaf():
                return t
            c_name = c.name
            new_nodes_list.append(c_name)
            branch_length = dist_fn(fp[p_name], fp[c_name])
            c.dist = branch_length
    return set_tree_lengths(t, new_nodes_list, fp, dist_fn)
    
def fit(module,
        fname : str,
        data : Tensor,
        t_ages : Array,
        num_states : int,
        batch_size : int=500,
        eps : float=1.,
        alpha : float=1e-4,
        order : int=2,
        max_iter : int=10000,
        filter_ages : float=None,
        device : torch.device=torch.device("cpu")) -> Tuple[Array, dict]:
    """Fits the treeRBM model on the data.
    
    Args:
        module: module containing the mean-field methods.
        fname (str): Path to the RBM model.
        data (Tensor): Data to fill the treeRBM model.
        t_ages (Array): Ages of the RBM at which compute the branches of the tree.
        num_states (int): Number of categories of the categorical variables.
        batch_size (int, optional): Batch size, to tune based on the memory availability. Defaults to 128.
        eps (float, optional): Epsilon parameter of the DBSCAN. Defaults to 1..
        alpha (float, optional): Convergence threshold of the TAP equations. Defaults to 1e-4.
        save_node_features (bool, optional): Wheather to save the states (fixed points) at the tree nodes.
        order (int, optional): Order of the mean-field free energy approximation. Defaults to 2.
        max_iter (int, optional): Maximum number of TAP iterations. Defaults to 10000.
        filter_ages (float, optional): Selects a subset of epochs such that the acceptance rate of swapping two adjacient configurations is the one specified.'
        device (torch.device): Device.
        
    Returns:
        Tuple[Array, dict] : Array with the encoded tree structure, dictionary that associates tree nodes to the fixed points.
    """

    # filter the ages
    if filter_ages:
        print(f'Filtering the ages with mutual acceptance rate of {filter_ages}:')
        chains = module.init_sampling(fname=fname, n_gen=1000, it_mcmc=100, epochs=t_ages, num_states=num_states, device=device)
        t_ages = module.filter_epochs(fname=fname, chains=chains, target_acc_rate=filter_ages, device=device)
        
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

def generate_tree(
    tree_codes : Array,
    folder : str,
    leaves_names : Array,
    legend : list=None,
    labels_dict : list=None,
    colors_dict : list=None,
    depth : int=None,
    node_features_dict : dict=None,
    dist_fn = l2_dist
) -> None:
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
        
    # Leaves names cannot contain the characters '(', ')', which are reserved for the newick sy>
    leaves_names = np.array([str(ln).replace('(', '').replace(')', '') for ln in leaves_names])
        
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
            ld = {k.replace('(', '').replace(')', '') : v for k, v in ld_raw.items() if v != '-1'}
            
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
            
    # Set the length of the first nodes wrt a uniform profile distribution
    init_node_list = []
    for nn in node_features_dict.keys():
        if "I0" in nn:
            init_node_list.append(nn)
            
    if len(node_features_dict[init_node_list[0]].shape) == 2:
        norm_factor = node_features_dict[init_node_list[0]].shape[1] # Potts
    else:
        norm_factor = 2 # Binary
    ref_profile = np.ones_like(node_features_dict[init_node_list[0]]) / norm_factor
    for nn in init_node_list:
        n = find_node_by_name(t, nn)
        n.dist = dist_fn(ref_profile, node_features_dict[nn])
    
    # Set the lengths of the internal branches
    t = set_tree_lengths(t=t, nodes_list=init_node_list, fp=node_features_dict, dist_fn=dist_fn)
    
    # Set the length of the leafs as the average tree branch length
    total_branch_length = 0
    num_branches = 0
    for node in t.traverse():
        if (not node.is_root()) and (not node.is_leaf()):
            total_branch_length += node.dist
            num_branches += 1
    if num_branches > 0:
        average_branch_length = total_branch_length / num_branches
    else:
        average_branch_length = 0
    for leaf in t.iter_leaves():
        leaf.dist = average_branch_length
    
    # Generate newick file
    t.write(format=1, outfile=f'{folder}/tree.nw')