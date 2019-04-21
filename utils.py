import numpy as np
from Bio import Phylo
from cStringIO import StringIO
import phyloinfer as pinf
from ete3 import Tree
from bitarray import bitarray
from dendropy import TreeList
import copy
from collections import defaultdict
EPS = 1e-100
import pdb


class BitArray(object):
    def __init__(self, taxa):
        self.taxa = taxa
        self.ntaxa = len(taxa)
        self.map = {taxon: i for i, taxon in enumerate(taxa)}
        
    def combine(self, arrA, arrB):
        if arrA < arrB:
            return arrA + arrB
        else:
            return arrB + arrA 
        
    def merge(self, key):
        return bitarray(key[:self.ntaxa]) | bitarray(key[self.ntaxa:])
        
    def decomp_minor(self, key):
        return min(bitarray(key[:self.ntaxa]), bitarray(key[self.ntaxa:]))
        
    def minor(self, arrA):
        return min(arrA, ~arrA)
        
    def from_clade(self, clade):
        bit_list = ['0'] * self.ntaxa
        for taxon in clade:
            bit_list[self.map[taxon]] = '1'
        return bitarray(''.join(bit_list))
    

class ParamParser(object):
    def __init__(self):
        self.start_and_end = {}
        self.num_params = 0
        self.num_params_in_dicts = 0
        self.dict_len = []
        
    def add_item(self, name):
        start = self.num_params
        self.num_params += 1
        self.start_and_end[name] = (start, self.num_params)
        
    def check_item(self, name):
        return (name in self.start_and_end)
        
    def add_dict(self, name):
        start = self.num_params_in_dicts
        self.num_params_in_dicts = self.num_params
        self.start_and_end[name] = (start, self.num_params)
        self.dict_len.append(self.num_params - start)
        
    def get(self, vect, name):
        if name in self.start_and_end:
            start, end = self.start_and_end[name]
            if end - start == 1:
                return vect[start]
            else:
                return vect[start:end]
        else:
            return 0.0
    
    def assign(self, vect, name, value):
        start, end = self.start_and_end[name]
        vect[start:end] = value
        
    def get_indexes(self, name):
        return self.start_and_end[name]
        
    def get_lengths(self, name):
        start, end = self.start_and_end[name]
        return end - start
        



def taxa2num(taxa):
    taxa2numMap = {}
    for i, taxon in enumerate(taxa):
        taxa2numMap[taxon] = i    
    return taxa2numMap 
    
    
def logmeanexp(arrA, axis=0, exclude=False, keepdims=False):
    max_arrA = np.amax(arrA, axis=axis, keepdims=keepdims)
    is_not_neginf_mask = ~np.isneginf(max_arrA)
    if not exclude:
        res = np.zeros(max_arrA.shape)
        res[is_not_neginf_mask] =np.log(np.mean(np.exp(arrA[:,is_not_neginf_mask]-max_arrA[is_not_neginf_mask]), axis=axis, keepdims=keepdims))
        return  res + max_arrA
    else:
        K = len(arrA)
        matA = np.tile(arrA, reps=(K,1)).T
        np.fill_diagonal(matA, 0.0)
        matA += np.diag(np.sum(matA, axis=0)/(K-1.))
        return logmeanexp(matA)    
    

def softmax(arrA):
    max_arrA = np.max(arrA)
    if max_arrA != -np.inf:
        normalizing_constant = np.sum(np.exp(arrA - max_arrA))
        return np.exp(arrA - max_arrA) / normalizing_constant
    else:
        return np.ones(arrA.shape)/len(arrA)
        

def dict_sum(arrA, dict_length):
    cumsum_arrA = np.zeros(len(arrA) + 1)
    cumsum_arrA[1:] = np.cumsum(arrA)
    cumsum_dict_arrA = np.zeros(len(dict_length) + 1)
    cumsum_dict_arrA[1:] = cumsum_arrA[np.cumsum(dict_length)]
    sum_dict_arrA = np.repeat(np.diff(cumsum_dict_arrA), dict_length)
    return sum_dict_arrA
        

def softmax_parser(arrA, parser, dict_names):
    dict_length = parser.dict_len
    max_arrA = np.repeat([np.max(parser.get(arrA, name)) for name in dict_names], dict_length)
    exp_arrA = np.exp(arrA - max_arrA)
    normal_arrA = dict_sum(exp_arrA, dict_length)
    return exp_arrA / normal_arrA
        

def upper_clip(z, ub):
    return z if not np.isposinf(z) else ub
        
        
# generate full tree space    
def generate(taxa):
    if len(taxa)==3:
        return [pinf.Tree('('+','.join(taxa)+');')]
    else:
        res = []
        sister = pinf.Tree('('+taxa[-1]+');')
        for tree in generate(taxa[:-1]):
            for node in tree.traverse('preorder'):
                if not node.is_root():
                    node.up.add_child(sister)
                    node.detach()
                    sister.add_child(node)
                    res.append(copy.deepcopy(tree))
                    node.detach()
                    sister.up.add_child(node)
                    sister.detach()
        
        return res
                   
                       
def clade_table(probs, trees, model):
    table = defaultdict(list)
    for i, tree in enumerate(trees):
        prob = probs[i]
        for node in tree.traverse('preorder'):
            node_bitarr = model.clade_to_bitarr(node.get_leaf_names())
            table[min(node_bitarr,~node_bitarr).to01()].append(prob)
    return table
    
    
def saveTree(sampled_tree, filename, tree_format=5):
    if type(sampled_tree) is not list:
        sampled_tree = [sampled_tree]
        
    with open(filename,'w') as output_file:
        for tree in sampled_tree:
            tree_newick = tree.write(format=tree_format)
            output_file.write(tree_newick + '\n') 
            

def consensus(path, schema='nexus', rooting='force-unrooted', min_freq=0.5):
    tree_list = TreeList.get(path=path, schema=schema, rooting=rooting)
    consensus_tree = tree_list.consensus(min_freq=min_freq)
    consensus_tree_str = consensus_tree.as_string(schema='newick', suppress_rooting=True)
    consensus_tree_newick = ''.join(c for c in consensus_tree_str if c not in ['\'', '\n'])
    
    output_tree = Tree(consensus_tree_newick, format=9)
    return output_tree
     
    
def summary(dataset, file_path, truncate=None):
    tree_dict_total = {}
    tree_dict_map_total = defaultdict(float)
    tree_names_total = []
    tree_wts_total = []
    n_samp_tree = 0
    for i in range(1,11):
        tree_dict_rep, tree_name_rep, tree_wts_rep = pinf.result.mcmc_treeprob(file_path + dataset + '/rep_{}/'.format(i) + dataset + '.trprobs', 'nexus', truncate=truncate, taxon='keep')
        tree_wts_rep = np.round(np.array(tree_wts_rep)*750001)
 
        for j, name in enumerate(tree_name_rep):
            tree_id = tree_dict_rep[name].get_topology_id()
            if tree_id not in tree_dict_map_total:
                n_samp_tree += 1
                tree_names_total.append('tree_{}'.format(n_samp_tree))
                tree_dict_total[tree_names_total[-1]] = tree_dict_rep[name]

            tree_dict_map_total[tree_id] += tree_wts_rep[j]
    
    for key in tree_dict_map_total:
        tree_dict_map_total[key] /= 10*750001

    for name in tree_names_total:
        tree_wts_total.append(tree_dict_map_total[tree_dict_total[name].get_topology_id()])  
        
    return tree_dict_total, tree_names_total, tree_wts_total
    
    
def get_tree_list(filename, data_type, burnin=0, truncate=None):
    samp_tree_stats = Phylo.parse(filename, data_type)
    tree_dict = {}
    tree_wts_dict = defaultdict(float)
    tree_names = []
    i, num_trees = 0, 0
    for tree in samp_tree_stats:
        num_trees += 1
        if num_trees < burnin:
            continue
        handle = StringIO()
        Phylo.write(tree, handle, 'newick')
        ete_tree = Tree(handle.getvalue().strip())
        handle.close()
        tree_id = ete_tree.get_topology_id()
        if tree_id not in tree_wts_dict:
            tree_name = 'tree_{}'.format(i)
            tree_dict[tree_name] = ete_tree
            tree_names.append(tree_name)
            i += 1
        tree_wts_dict[tree_id] += 1.0
                
        if truncate and num_trees == truncate+burnin:
            break
    
    tree_wts = [tree_wts_dict[tree_dict[tree_name].get_topology_id()]/(num_trees-burnin) for tree_name in tree_names]
        
    return tree_dict, tree_names, tree_wts
    

def get_tree_list_raw(filename, burnin=0, truncate=None, hpd=0.95):
    tree_dict = {}
    tree_wts_dict = defaultdict(float)
    tree_names = []
    i, num_trees = 0, 0
    with open(filename, 'r') as input_file:
        while True:
            line = input_file.readline()
            if line == "":
                break
            num_trees += 1
            if num_trees < burnin:
                continue
            tree = Tree(line.strip())
            tree_id = tree.get_topology_id()
            if tree_id not in tree_wts_dict:
                tree_name = 'tree_{}'.format(i)
                tree_dict[tree_name] = tree
                tree_names.append(tree_name)
                i += 1            
            tree_wts_dict[tree_id] += 1.0
            
            if truncate and num_trees == truncate + burnin:
                break
    tree_wts = [tree_wts_dict[tree_dict[tree_name].get_topology_id()]/(num_trees-burnin) for tree_name in tree_names]
    if hpd < 1.0:
        ordered_wts_idx = np.argsort(tree_wts)[::-1]
        cum_wts_arr = np.cumsum([tree_wts[k] for k in ordered_wts_idx])
        cut_at = next(x[0] for x in enumerate(cum_wts_arr) if x[1] > hpd)
        tree_wts = [tree_wts[k] for k in ordered_wts_idx[:cut_at]]
        tree_names = [tree_names[k] for k in ordered_wts_idx[:cut_at]]
        
    return tree_dict, tree_names, tree_wts
    

def summary_raw(dataset, file_path, truncate=None, hpd=0.95):
    tree_dict_total = {}
    tree_id_set_total = set()
    tree_names_total = []
    n_samp_tree = 0
    
    for i in range(1, 11):
        tree_dict_rep, tree_names_rep, tree_wts_rep = get_tree_list_raw(file_path + dataset + '/' + dataset + '_ufboot_rep_{}'.format(i), truncate=truncate, hpd=hpd)
        for j, name in enumerate(tree_names_rep):
            tree_id = tree_dict_rep[name].get_topology_id()
            if tree_id not in tree_id_set_total:
                n_samp_tree += 1
                tree_names_total.append('tree_{}'.format(n_samp_tree))
                tree_dict_total[tree_names_total[-1]] = tree_dict_rep[name]
                tree_id_set_total.add(tree_id)
    
    return tree_dict_total, tree_names_total
        
            
def stepping_stone_stats(filename):
    ss_stats = []
    with open(filename, 'r') as readin_file:
        while True:
            line = readin_file.readline()
            if line == "":
                break
            if line[0] == '[':
                continue
            
            if line[0] == 'S':
                names = line.strip('\n').split('\t')
            else:
                stats = line.strip('\n').split('\t')
                ss_stats.append([float(stat) for stat in stats[2:-1]])
    return np.asarray(ss_stats)
    

def get_support_from_samples(taxa, tree_dict_total, tree_names_total, tree_wts_total=None):
    rootsplit_supp_dict = defaultdict(float)
    subsplit_supp_dict = defaultdict(lambda: defaultdict(float))
    toBitArr = BitArray(taxa)
    for i, tree_name in enumerate(tree_names_total):
        tree = tree_dict_total[tree_name]
        wts = tree_wts_total[i] if tree_wts_total else 1.0
        nodetobitMap = {node:toBitArr.from_clade(node.get_leaf_names()) for node in tree.traverse('postorder') if not node.is_root()}
        for node in tree.traverse('levelorder'):
            if not node.is_root():
                rootsplit = toBitArr.minor(nodetobitMap[node]).to01()
                rootsplit_supp_dict[rootsplit] += wts
                if not node.is_leaf():
                    child_subsplit = min([nodetobitMap[child] for child in node.children]).to01()
                    for sister in node.get_sisters():
                        parent_subsplit = (nodetobitMap[sister] + nodetobitMap[node]).to01()
                        subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                    if not node.up.is_root():
                        parent_subsplit = (~nodetobitMap[node.up] + nodetobitMap[node]).to01()
                        subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                        
                    parent_subsplit = (~nodetobitMap[node] + nodetobitMap[node]).to01()
                    subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                
                if not node.up.is_root():
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()] + [~nodetobitMap[node.up]])
                else:
                    bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])
                child_subsplit = bipart_bitarr.to01()
                if not node.is_leaf():
                    for child in node.children:
                        parent_subsplit = (nodetobitMap[child] + ~nodetobitMap[node]).to01()
                        subsplit_supp_dict[parent_subsplit][child_subsplit] += wts
                
                parent_subsplit = (nodetobitMap[node] + ~nodetobitMap[node]).to01()
                subsplit_supp_dict[parent_subsplit][child_subsplit] += wts

    return rootsplit_supp_dict, subsplit_supp_dict                           
    
    
def load_params(filename):
    params = {}
    with open(filename + '__CPDs.txt') as _CPDs_file:
        _CPDs_list = _CPDs_file.readlines()
        params['_CPDs'] = [np.array([float(item) for item in _CPDs.strip().split()[1:]]) for _CPDs in _CPDs_list]
    with open(filename + '_CPDs.txt') as CPDs_file:
        CPDs_list = CPDs_file.readlines()
        params['CPDs'] = [np.array([float(item) for item in CPDs.strip().split()[1:]]) for CPDs in CPDs_list]
    with open(filename + '_loc.txt') as loc_file:
        loc_list = loc_file.readlines()
        params['loc'] = [np.array([float(item) for item in loc.strip().split()[1:]]) for loc in loc_list]
    with open(filename + '_shape.txt') as shape_file:
        shape_list = shape_file.readlines()
        params['shape'] = [np.array([float(item) for item in shape.strip().split()[1:]]) for shape in shape_list]
    return params