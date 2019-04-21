import numpy as np
from bitarray import bitarray
from ete3 import Tree
from collections import defaultdict
import phyloinfer as pinf
from optimizers import SGD_Server
from utils import softmax, softmax_parser, dict_sum, upper_clip, logmeanexp, BitArray, ParamParser
import distributions
import time
from copy import deepcopy
from joblib import Parallel, delayed
import multiprocessing
import pdb


class SBN(object):
    """
    Subsplit Bayesian Networks (SBNs) for distributions over 
    phylogenetic tree topologies. SBNs utilize the similarity 
    among tree topologies to provide a familty of flexibile 
    distributions over the entire tree topology space. 
    
    Parameters
    ----------
    taxa : ``list``, a list of the labels of the sequences.
    rootsplit_supp_dict : ``dict``, a dictionary of rootsplit support, 
                           usually obtained from some bootstrap run.
    subsplit_supp_dict: ``dict``, a dictionary of subsplit support,
                         obtained similarly as above.
    
    References
    ----------
    .. [1] Cheng Zhang and Frederick A. Matsen IV. "Generalizing Tree 
    Probability Estimation via Bayesian Networks", Advances in Neural 
    Information Processing Systems 32, 2018. (https://papers.nips.cc/
    paper/7418-generalizing-tree-probability-estimation-via-bayesian-
    networks)
    
    """
    
    def __init__(self, taxa, rootsplit_supp_dict, subsplit_supp_dict, support_only=True):
        self.taxa, self.ntaxa = taxa, len(taxa)
        self.toBitArr = BitArray(taxa)
        self.CPDs_parser = ParamParser()
        self.dict_names = []
        self.rootsplit_supp_dict = rootsplit_supp_dict
        self.subsplit_supp_dict = subsplit_supp_dict
        if not support_only:
            init_CPDs = []
        for split in rootsplit_supp_dict:
            self.CPDs_parser.add_item(split)
            if not support_only:
                init_CPDs.append(rootsplit_supp_dict[split])
        self.CPDs_parser.add_dict('rootsplits')
        self.dict_names.append('rootsplits')
        for parent in subsplit_supp_dict:
            for child in subsplit_supp_dict[parent]:
                self.CPDs_parser.add_item(parent + child)
                if not support_only:
                    init_CPDs.append(subsplit_supp_dict[parent][child])
            self.CPDs_parser.add_dict(parent)
            self.dict_names.append(parent)
        
        self.num_CPDs = self.CPDs_parser.num_params
        self._CPDs = np.zeros(self.num_CPDs)
        if support_only:
            self.CPDs = softmax_parser(self._CPDs, self.CPDs_parser, self.dict_names)
        else:
            self.CPDs = init_CPDs
        
    def get_CPDs(self, name):
        return self.CPDs_parser.get(self.CPDs, name)
        
    def assign_CPDs(self, vect, name, value):
        self.CPDs_parser.assign(vect, name, value)
        
    @property
    def rootsplit_CPDs(self):
        return {split: self.get_CPDs(split) for split in self.rootsplit_supp_dict}
        
    def subsplit_CPDs(self, parent):
        return {child: self.get_CPDs(parent+child) for child in self.subsplit_supp_dict[parent]}
        
    
    def check_item(self, name):
        return self.CPDs_parser.check_item(name)
        
    def node2bitMap(self, tree, bit_type='split'):
        if bit_type == 'split':
            return {node: self.toBitArr.minor(self.toBitArr.from_clade(node.get_leaf_names())).to01() for node in tree.traverse('postorder') if not node.is_root()}
        elif bit_type == 'clade':
            return {node: self.toBitArr.from_clade(node.get_leaf_names()) for node in tree.traverse('postorder') if not node.is_root()}
    
    def check_subsplit_pair(self, subsplit_pair):
        # if self.CPDs_parser.get(self.CPDs, subsplit_pair) == 0.0:
        if self.get_CPDs(subsplit_pair) == 0.0:
            return False
        return True
    
    def rooted_tree_probs(self, tree, nodetobitMap=None):
        """
        Compute the logprobs of all compatible rooted tree topologies
        of the unrooted tree via a two pass algorithm. 
        The overall computational cost is O(N) where N is the number of
        species. 
        """
        sbn_est_up = {node: 0.0 for node in tree.traverse('postorder') if not node.is_root()}
        Up = {node: 0.0 for node in tree.traverse('postorder') if not node.is_root()}

        if not nodetobitMap:
            nodetobitMap = self.node2bitMap(tree, 'clade')
            
        bipart_bitarr_up = {}
        bipart_bitarr_prob = {}
        bipart_bitarr_node = {}
        zero_bubble_up = {node:0 for node in tree.traverse('postorder') if not node.is_root()}
        zero_bubble_Up = {node:0 for node in tree.traverse('postorder') if not node.is_root()}
        
        for node in tree.traverse('postorder'):
            if not node.is_leaf() and not node.is_root():
                for child in node.children:
                    Up[node] += sbn_est_up[child]
                    zero_bubble_Up[node] += zero_bubble_up[child]
                    
                bipart_bitarr = min(nodetobitMap[child] for child in node.children)
                bipart_bitarr_up[node] = bipart_bitarr
                if not node.up.is_root():
                    sbn_est_up[node] += Up[node]
                    zero_bubble_up[node] += zero_bubble_Up[node]
                    comb_parent_bipart_bitarr = nodetobitMap[node.get_sisters()[0]] + nodetobitMap[node]
                    item_name = comb_parent_bipart_bitarr.to01() + bipart_bitarr.to01()
                    if not self.check_subsplit_pair(item_name):
                        zero_bubble_up[node] += 1
                    else:
                        sbn_est_up[node] += np.log(self.get_CPDs(item_name))
        
        sbn_est_down = {node: 0.0 for node in tree.traverse('preorder') if not node.is_root()}
        zero_bubble_down = {node: 0.0 for node in tree.traverse('preorder') if not node.is_root()}
        zero_bubble = defaultdict(int)
        bipart_bitarr_down = {}
        
        for node in tree.traverse('preorder'):
            if not node.is_root():
                if node.up.is_root():
                    parent_bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])
                    bipart_bitarr_down[node] = parent_bipart_bitarr
                    
                    for sister in node.get_sisters():
                        if not sister.is_leaf():
                            sbn_est_down[node] += Up[sister]
                            zero_bubble_down[node] += zero_bubble_Up[sister]
                            comb_parent_bipart_bitarr = ((~nodetobitMap[node]) ^ nodetobitMap[sister]) + nodetobitMap[sister]
                            item_name = comb_parent_bipart_bitarr.to01() + bipart_bitarr_up[sister].to01()
                            if not self.check_subsplit_pair(item_name):
                                zero_bubble_down[node] += 1
                            else:
                                sbn_est_down[node] += np.log(self.get_CPDs(item_name))
                else:
                    sister = node.get_sisters()[0]
                    parent_bipart_bitarr = min([nodetobitMap[sister], ~nodetobitMap[node.up]])
                    bipart_bitarr_down[node] = parent_bipart_bitarr
                    sbn_est_down[node] += sbn_est_down[node.up]
                    zero_bubble_down[node] += zero_bubble_down[node.up]
                    comb_parent_bipart_bitarr = nodetobitMap[sister] + ~nodetobitMap[node.up]
                    item_name = comb_parent_bipart_bitarr.to01() + bipart_bitarr_down[node.up].to01()
                    if not self.check_subsplit_pair(item_name):
                        zero_bubble_down[node] += 1
                    else:
                        sbn_est_down[node] += np.log(self.get_CPDs(item_name))
                    
                    if not sister.is_leaf():
                        sbn_est_down[node] += Up[sister]
                        zero_bubble_down[node] += zero_bubble_Up[sister]
                        comb_parent_bipart_bitarr = ~nodetobitMap[node.up] + nodetobitMap[sister]
                        item_name = comb_parent_bipart_bitarr.to01() + bipart_bitarr_up[sister].to01()
                        if not self.check_subsplit_pair(item_name):
                            zero_bubble_down[node] += 1
                        else:
                            sbn_est_down[node] += np.log(self.get_CPDs(item_name))
                
                parent_bipart_bitarr = self.toBitArr.minor(nodetobitMap[node])
                bipart_bitarr_node[node] = parent_bipart_bitarr
                if parent_bipart_bitarr.to01() not in self.rootsplit_supp_dict or self.get_CPDs(parent_bipart_bitarr.to01()) == 0.0:
                    zero_bubble[parent_bipart_bitarr.to01()] += 1
                    bipart_bitarr_prob[parent_bipart_bitarr.to01()] = 0.0
                else:
                    bipart_bitarr_prob[parent_bipart_bitarr.to01()] = np.log(self.get_CPDs(parent_bipart_bitarr.to01()))
                    
                if not node.is_leaf():
                    bipart_bitarr_prob[parent_bipart_bitarr.to01()] += Up[node]
                    zero_bubble[parent_bipart_bitarr.to01()] += zero_bubble_Up[node]
                    comb_parent_bipart_bitarr = ~nodetobitMap[node] + nodetobitMap[node]
                    item_name = comb_parent_bipart_bitarr.to01() + bipart_bitarr_up[node].to01()
                    if not self.check_subsplit_pair(item_name):
                        zero_bubble[parent_bipart_bitarr.to01()] += 1
                    else:
                        bipart_bitarr_prob[parent_bipart_bitarr.to01()] += np.log(self.get_CPDs(item_name))
                
                bipart_bitarr_prob[parent_bipart_bitarr.to01()] += sbn_est_down[node]
                zero_bubble[parent_bipart_bitarr.to01()] += zero_bubble_down[node]
                comb_parent_bipart_bitarr = nodetobitMap[node] + ~nodetobitMap[node]
                item_name = comb_parent_bipart_bitarr.to01() + bipart_bitarr_down[node].to01()
                if not self.check_subsplit_pair(item_name):
                    zero_bubble[parent_bipart_bitarr.to01()] += 1
                else:
                    bipart_bitarr_prob[parent_bipart_bitarr.to01()] += np.log(self.get_CPDs(item_name))
                
        bipart_bitarr_prob_real = {key: value if zero_bubble[key]==0 else -np.inf for key, value in bipart_bitarr_prob.iteritems()}
        bipart_bitarr_prob_mask = {key: value if zero_bubble[key]<2 else -np.inf for key, value in bipart_bitarr_prob.iteritems()}
        return bipart_bitarr_prob_real, bipart_bitarr_prob_mask, bipart_bitarr_node, zero_bubble
        
    
    def cum_root_probs(self, tree, bipart_bitarr_prob, bipart_bitarr_node, max_bipart_bitarr_prob=None, log=False, normalized=True):
        """
        Compute the cumulative sums of the rooted tree probabilities
        from the leaves to the root.
        """
        root_prob = {}
        cum_root_prob = defaultdict(float)
        if max_bipart_bitarr_prob is None:
            max_bipart_bitarr_prob = np.max(bipart_bitarr_prob.values())
        for node in tree.traverse('postorder'):
            if not node.is_root():
                bipart_bitarr = bipart_bitarr_node[node]
                cum_root_prob[bipart_bitarr.to01()] += np.exp(bipart_bitarr_prob[bipart_bitarr.to01()] - max_bipart_bitarr_prob)
                if not node.is_leaf():
                    for child in node.children:
                        cum_root_prob[bipart_bitarr.to01()] += cum_root_prob[bipart_bitarr_node[child].to01()]
    
        root_prob_sum = 0.0
        for child in tree.children:
            root_prob_sum += cum_root_prob[bipart_bitarr_node[child].to01()]
    
        if normalized:
            if log:
                root_prob = {key: bipart_bitarr_prob[key] - max_bipart_bitarr_prob - np.log(root_prob_sum) for key in bipart_bitarr_prob}
                cum_root_prob = {key: np.log(cum_root_prob[key]) - np.log(root_prob_sum) if cum_root_prob[key] != 0.0 else -np.inf for key in bipart_bitarr_prob}
            else:
                root_prob = {key: np.exp(bipart_bitarr_prob[key] - max_bipart_bitarr_prob)/root_prob_sum for key in bipart_bitarr_prob}
                cum_root_prob = {key: cum_root_prob[key]/root_prob_sum for key in bipart_bitarr_prob}
    
        return root_prob, cum_root_prob, root_prob_sum
        
      
    def tree_loglikelihood(self, tree, nodetobitMap=None, grad=False, entry_ub=10.0, value_and_grad=False):
        """ Compute the SBN loglikelihood and gradient. """
         
        if not nodetobitMap:
            nodetobitMap = self.node2bitMap(tree, bit_type='clade')
        
        bipart_bitarr_prob_real, bipart_bitarr_prob, bipart_bitarr_node, zero_bubble = self.rooted_tree_probs(tree, nodetobitMap)
        bipart_bitarr_prob_real_vec = np.array(bipart_bitarr_prob_real.values())
        max_bipart_bitarr_prob_real = np.max(bipart_bitarr_prob_real_vec)
        if max_bipart_bitarr_prob_real != -np.inf:
            loglikelihood = np.log(np.sum(np.exp(bipart_bitarr_prob_real_vec - max_bipart_bitarr_prob_real))) + max_bipart_bitarr_prob_real
        else:
            loglikelihood = -np.inf
        max_bipart_bitarr_prob = np.max(bipart_bitarr_prob.values())    
        if not grad:
            return loglikelihood
        
        CPDs_grad = np.ones(self.num_CPDs) * (-np.inf)
        root_prob_real, cum_root_prob_real, _ = self.cum_root_probs(tree, bipart_bitarr_prob_real, bipart_bitarr_node, max_bipart_bitarr_prob_real, log=True)
        _, cum_root_prob, root_prob_sum = self.cum_root_probs(tree, bipart_bitarr_prob, bipart_bitarr_node, max_bipart_bitarr_prob, normalized=False)
        
        for node in tree.traverse('postorder'):
            if not node.is_root():
                bipart_bitarr = bipart_bitarr_node[node]
                if bipart_bitarr.to01() in self.rootsplit_supp_dict:
                    if self.get_CPDs(bipart_bitarr.to01()) == 0.0:
                        self.assign_CPDs(CPDs_grad, bipart_bitarr.to01(), upper_clip(bipart_bitarr_prob[bipart_bitarr.to01()] - loglikelihood, entry_ub))
                    else:
                        self.assign_CPDs(CPDs_grad, bipart_bitarr.to01(), root_prob_real[bipart_bitarr.to01()] - np.log(self.get_CPDs(bipart_bitarr.to01())))
                
                if not node.is_leaf():
                    children_bipart_bitarr = min([nodetobitMap[child] for child in node.children])
                    if not node.up.is_root():
                        parent_bipart_bitarr = bipart_bitarr_node[node.up]
                        comb_parent_bipart_bitarr = nodetobitMap[node.get_sisters()[0]] + nodetobitMap[node]
                        item_name = comb_parent_bipart_bitarr.to01() + children_bipart_bitarr.to01()
                        if self.check_item(item_name):
                            if self.get_CPDs(item_name) == 0.0:
                                self.assign_CPDs(CPDs_grad, item_name, np.log(root_prob_sum - cum_root_prob[parent_bipart_bitarr.to01()] + \
                                    np.exp(bipart_bitarr_prob[parent_bipart_bitarr.to01()] - max_bipart_bitarr_prob)) + upper_clip(max_bipart_bitarr_prob - loglikelihood, entry_ub))
                            else:
                                cum_node_prob = 1.0 - np.exp(cum_root_prob_real[parent_bipart_bitarr.to01()]) + np.exp(root_prob_real[parent_bipart_bitarr.to01()])
                                if cum_node_prob == 0.0:
                                    self.assign_CPDs(CPDs_grad, item_name, -np.inf)
                                else:
                                    self.assign_CPDs(CPDs_grad, item_name, np.log(cum_node_prob) - np.log(self.get_CPDs(item_name)))
                                
                        comb_parent_bipart_bitarr = ~nodetobitMap[node.up] + nodetobitMap[node]
                        item_name = comb_parent_bipart_bitarr.to01() + children_bipart_bitarr.to01()
                        if self.check_item(item_name):
                            if self.get_CPDs(item_name) == 0.0:
                                self.assign_CPDs(CPDs_grad, item_name, np.log(cum_root_prob[bipart_bitarr_node[node.get_sisters()[0]].to01()]) + \
                                                upper_clip(max_bipart_bitarr_prob - loglikelihood, entry_ub))
                            else:
                                cum_node_prob = cum_root_prob_real[bipart_bitarr_node[node.get_sisters()[0]].to01()]
                                self.assign_CPDs(CPDs_grad, item_name, cum_node_prob - np.log(self.get_CPDs(item_name)))
                    else:
                        for sister in node.get_sisters():
                            comb_parent_bipart_bitarr = nodetobitMap[sister] + nodetobitMap[node]
                            item_name = comb_parent_bipart_bitarr.to01() + children_bipart_bitarr.to01()
                            if self.check_item(item_name):
                                if self.get_CPDs(item_name) == 0.0:
                                    self.assign_CPDs(CPDs_grad, item_name, np.log(cum_root_prob[self.toBitArr.minor((~nodetobitMap[node]) ^ nodetobitMap[sister]).to01()]) + \
                                             upper_clip(max_bipart_bitarr_prob - loglikelihood, entry_ub))
                                else:
                                    cum_node_prob = cum_root_prob_real[self.toBitArr.minor(nodetobitMap[sister] ^ (~nodetobitMap[node])).to01()]
                                    self.assign_CPDs(CPDs_grad, item_name, cum_node_prob - np.log(self.get_CPDs(item_name)))
                    
                    comb_parent_bipart_bitarr = ~nodetobitMap[node] + nodetobitMap[node]
                    item_name = comb_parent_bipart_bitarr.to01() + children_bipart_bitarr.to01()
                    if self.check_item(item_name):
                        if self.get_CPDs(item_name) == 0.0:
                            self.assign_CPDs(CPDs_grad, item_name, upper_clip(bipart_bitarr_prob[bipart_bitarr.to01()] - loglikelihood, entry_ub))
                        else:
                            self.assign_CPDs(CPDs_grad, item_name, root_prob_real[bipart_bitarr.to01()] - np.log(self.get_CPDs(item_name)))
                
                if not node.up.is_root():
                    children_bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()] + [~nodetobitMap[node.up]])
                else:
                    children_bipart_bitarr = min([nodetobitMap[sister] for sister in node.get_sisters()])
                    
                if not node.is_leaf():
                    for child in node.children:
                        comb_parent_bipart_bitarr = nodetobitMap[child] + ~nodetobitMap[node]
                        item_name = comb_parent_bipart_bitarr.to01() + children_bipart_bitarr.to01()
                        if self.check_item(item_name):
                            if self.get_CPDs(item_name) == 0.0:
                                self.assign_CPDs(CPDs_grad, item_name, np.log(cum_root_prob[self.toBitArr.minor(nodetobitMap[node] ^ nodetobitMap[child]).to01()]) + \
                                        upper_clip(max_bipart_bitarr_prob - loglikelihood, entry_ub))
                            else:
                                cum_node_prob = cum_root_prob_real[self.toBitArr.minor(nodetobitMap[node] ^ nodetobitMap[child]).to01()]
                                self.assign_CPDs(CPDs_grad, item_name, cum_node_prob - np.log(self.get_CPDs(item_name)))
                
                comb_parent_bipart_bitarr = nodetobitMap[node] + ~nodetobitMap[node]
                item_name = comb_parent_bipart_bitarr.to01() + children_bipart_bitarr.to01()
                if self.check_item(item_name):
                    if self.get_CPDs(item_name) == 0.0:
                        self.assign_CPDs(CPDs_grad, item_name, upper_clip(bipart_bitarr_prob[bipart_bitarr.to01()] - loglikelihood, entry_ub))
                    else:
                        self.assign_CPDs(CPDs_grad, item_name, root_prob_real[bipart_bitarr.to01()] - np.log(self.get_CPDs(item_name)))
        
        CPDs_grad = np.exp(CPDs_grad)
        dict_length = self.CPDs_parser.dict_len
        dict_sum_grad = dict_sum(self.CPDs*CPDs_grad, dict_length)
        CPDs_grad = (CPDs_grad - dict_sum_grad) * self.CPDs
        
        if not value_and_grad:
            return CPDs_grad
        else:
            return loglikelihood, CPDs_grad
        

    def sample_tree(self, rooted=False):
        """ Sampling from SBN (ancestral sampling) """
        
        root = Tree()
        node_split_stack = [(root, '0'*self.ntaxa + '1'*self.ntaxa)]
        for i in range(self.ntaxa-1):
            node, split_bitarr = node_split_stack.pop()
            parent_clade_bitarr = bitarray(split_bitarr[self.ntaxa:])
            if node.is_root():
                split_candidate, split_prob = zip(*self.rootsplit_CPDs.iteritems())
            else:
                split_candidate, split_prob = zip(*self.subsplit_CPDs(split_bitarr).iteritems())
            
            split = np.random.choice(split_candidate, p=split_prob)                
            comp_split = (parent_clade_bitarr ^ bitarray(split)).to01()
            
            c1 = node.add_child()
            c2 = node.add_child()
            if split.count('1') > 1:
                node_split_stack.append((c1, comp_split + split))
            else:
                c1.name = self.taxa[split.find('1')]
            if comp_split.count('1') > 1:
                node_split_stack.append((c2, split + comp_split))
            else:
                c2.name = self.taxa[comp_split.find('1')]
        
        if not rooted:
            root.unroot()
        
        return root          
    
 
    @staticmethod  
    def aggregate_CPDs_grad(wts, CPDs_grad, clip=None, choose_one=False):
        """
        Aggregate gradients from multiple sampled tree topologies.
        
        Parameters
        ----------
        wts : ``np.array``, the weight vector for sampled tree topologies.
        CPDs_grad : ``np.ndarray``, the gradient matrix of CPDs.
        clip : ``float``, the bound for clipping the gradient.
        choose_one: ``boolean``, optional for reweighted weak sleep (RWS).
        """
        
        n_particles = len(CPDs_grad)
        if not choose_one:
            agg_CPDs_grad = np.sum(wts.reshape(n_particles, 1) * CPDs_grad, 0)
        else:
            if choose_one == 'sample':
                samp_index = np.random.choice(np.arange(n_particles), p=wts)
            elif choose_one == 'max':
                samp_index = wts.argmax()
            else:
                raise NotImplementedError
            agg_CPDs_grad = CPDs_grad[samp_index]
        if clip:
            agg_CPDs_grad = np.clip(agg_CPDs_grad, -clip, clip)
        return agg_CPDs_grad
            

class SBN_VI_EMP(SBN):
    """
    Training SBNs for empirical distributions over tree topologies only.
        
    Parameters
    ----------
    taxa : ``list``, a list of the labels of the sequences.
    emp_tree_freq : ``dict``, a dictionary of the empirical probabilities
                     of tree topologies.
    rootsplit_supp_dict : ``dict``, a dictionary of rootsplit support,
                          usually obtained from some bootstrap run.
    subsplit_supp_dict: ``dict``, a dictionary of subsplit support,
                         obtained similarly as above.
    
    References
    ----------
    .. [1] Cheng Zhang and Frederick A. Matsen IV. "Variational Bayesian Phylogenetic
           Inference",  In Proceedings of the 7th International Conference on Learning
           Representations (ICLR), 2019. (https://openreview.net/forum?id=SJVmjjR9FX)
    
    """
    
    EPS = 1e-100
    def __init__(self, taxa, emp_tree_freq, rootsplit_supp_dict, subsplit_supp_dict):
        super(SBN_VI_EMP, self).__init__(taxa, rootsplit_supp_dict, subsplit_supp_dict)
        self.trees , self.emp_freqs = zip(*emp_tree_freq.iteritems())
        self.emp_freqs = np.array(self.emp_freqs)
        self.emp_tree_freq = emp_tree_freq
        self.negDataEnt = np.sum(self.emp_freqs * np.log(np.maximum(self.emp_freqs, self.EPS)))
        self._data_loglikelihood = defaultdict(lambda: -np.inf)
        
        for tree, value in emp_tree_freq.iteritems():
            if value > 0.0:
                self._data_loglikelihood[tree.get_topology_id()] = np.log(value)
        
        
        
    
    # compute the KL divergence from SBN to the target empirical distribution.   
    def kl_div(self):
        kl_div = 0.0
        for tree, wt in self.emp_tree_freq.iteritems():
            kl_div += wt * np.log(max(np.exp(self.tree_loglikelihood(tree)), self.EPS))
        kl_div = self.negDataEnt - kl_div
        return kl_div

    
    def data_loglikelihood(self, tree):
        return self._data_loglikelihood[tree.get_topology_id()]
    
    # n-sample lower bound estimates   
    def lower_bound_estimate(self, n_particles, rooted=False, sample_sz=1000):
        lower_bound = np.empty(sample_sz)
        for sample in range(sample_sz):
            samp_tree_list = [self.sample_tree(rooted=rooted) for particle in range(n_particles)]
            loglikelihood = np.array([self.data_loglikelihood(samp_tree) for samp_tree in samp_tree_list])
            log_prob_ratios = loglikelihood - np.array([self.tree_loglikelihood(samp_tree) for samp_tree in samp_tree_list])  
            
            lower_bound[sample] = logmeanexp(log_prob_ratios)
        
        return np.mean(lower_bound[~np.isinf(lower_bound)])
        

    def rws(self, stepsz, maxiter=1000000, test_freq=1000, lb_test_freq=1000, lb_test_sampsz=1000, rooted=False, anneal_freq=20000, anneal_rate=0.75, alpha=1.0,
            n_particles=20, clip=100., momentum=0.9, decay=0.0, sgd_solver='adam', sample_particle=False):
        test_kl_div = []
        test_lower_bound = []
        
        lbs, lls = [], []
        infer_opt = SGD_Server({'CPDs':self.num_CPDs}, momentum=momentum, decay=decay)
        if not isinstance(stepsz, dict):
            stepsz = {'CPDs': stepsz}
            
        run_time = -time.time()
        for it in range(1, maxiter+1):
            samp_tree_list = [self.sample_tree(rooted) for particle in range(n_particles)]
            loglikelihood = np.array([self.data_loglikelihood(samp_tree) for samp_tree in samp_tree_list])
            logq_tree, CPDs_grad = zip(*[self.tree_loglikelihood(samp_tree, grad=True, value_and_grad=True) for samp_tree in samp_tree_list])
            logq_tree, CPDs_grad = np.array(logq_tree), np.concatenate(CPDs_grad).reshape(n_particles, -1)
            # log_prob_ratios = loglikelihood - np.array([self.tree_loglikelihood(samp_tree) for samp_tree in samp_tree_list])
            log_prob_ratios = loglikelihood - logq_tree
            particle_wts = softmax(alpha*log_prob_ratios)
            lower_bound = logmeanexp(log_prob_ratios)
            
            # CPDs_grad = self.aggregate_CPDs_grad(particle_wts, samp_tree_list, clip=clip, choose_one=sample_particle)
            CPDs_grad = self.aggregate_CPDs_grad(particle_wts, CPDs_grad, clip=clip, choose_one=sample_particle)
            self._CPDs = self._CPDs + getattr(infer_opt, sgd_solver)(stepsz, {'CPDs': self._CPDs}, {'CPDs': CPDs_grad})['CPDs']
            self.CPDs = softmax_parser(self._CPDs, self.CPDs_parser, self.dict_names)
            
            lls.append(np.max(loglikelihood))
            lbs.append(lower_bound)
            
            if it % anneal_freq == 0:
                # stepsz *= anneal_rate
                for var in stepsz:
                    stepsz[var] *= anneal_rate
            
            if it % test_freq == 0:
                run_time += time.time()
                test_kl_div.append(self.kl_div())
                print 'Iter {} ({:.1f}s): Lower Bound {:.4f} | Loglikelihood {:.4f} | KL {:.6f}'.format(it, run_time, np.mean(lbs), np.max(lls), test_kl_div[-1])
                if it % lb_test_freq == 0:
                    run_time = -time.time()
                    test_lower_bound.append(self.lower_bound_estimate(n_particles, sample_sz=lb_test_sampsz)) 
                    run_time += time.time()  
                    print '>>> Iter {} ({:.1f}s): Test Lower Bound {:.4f}'.format(it, run_time, test_lower_bound[-1])
                                 
                lbs, lls = [], []
                run_time = -time.time()
            
        return test_kl_div, test_lower_bound
    

    def vimco(self, stepsz, maxiter=1000000, test_freq=1000, lb_test_freq=1000, lb_test_sampsz=1000, rooted=False, anneal_freq=20000, anneal_rate=0.75,
              n_particles=20, clip=100., momentum=0.9, decay=0.0, sgd_solver='adam'):
        test_kl_div = []
        test_lower_bound = []
        
        lbs, lls = [], []
        infer_opt = SGD_Server({'CPDs': self.num_CPDs}, momentum=momentum, decay=decay)
        if not isinstance(stepsz, dict):
            stepsz = {'CPDs': stepsz}
            
        run_time = -time.time()
        for it in range(1, maxiter+1):
            samp_tree_list = [self.sample_tree(rooted=rooted) for particle in range(n_particles)]
            loglikelihood = np.array([self.data_loglikelihood(samp_tree) for samp_tree in samp_tree_list])
            logq_tree, CPDs_grad = zip(*[self.tree_loglikelihood(samp_tree, grad=True, value_and_grad=True) for samp_tree in samp_tree_list])
            logq_tree, CPDs_grad = np.array(logq_tree), np.concatenate(CPDs_grad).reshape(n_particles, -1)
            # log_prob_ratios = loglikelihood - np.array([self.tree_loglikelihood(samp_tree) for samp_tree in samp_tree_list])
            log_prob_ratios = loglikelihood - logq_tree
            
            particle_wts = softmax(log_prob_ratios)
            lower_bound = logmeanexp(log_prob_ratios)
            lower_bound_approx = logmeanexp(log_prob_ratios, exclude=True)
            
            if lower_bound == -np.inf:
                update_wts = -particle_wts
            else:
                update_wts = lower_bound - lower_bound_approx - particle_wts
                update_wts[np.isposinf(update_wts)] = 20.
            
            # CPDs_grad = self.aggregate_CPDs_grad(update_wts, samp_tree_list, clip=clip)
            CPDs_grad = self.aggregate_CPDs_grad(update_wts, CPDs_grad, clip=clip)
            self._CPDs = self._CPDs + getattr(infer_opt, sgd_solver)(stepsz, {'CPDs': self._CPDs}, {'CPDs': CPDs_grad})['CPDs']
            self.CPDs = softmax_parser(self._CPDs, self.CPDs_parser, self.dict_names)
            
            lls.append(np.max(loglikelihood))
            lbs.append(lower_bound)
            
            if it % anneal_freq == 0:
                for var in stepsz:
                    stepsz[var] *= anneal_rate
            
            if it % test_freq == 0:
                run_time += time.time()
                test_kl_div.append(self.kl_div())                   
                print 'Iter {} ({:.1f}s): Lower Bound {:.4f} | Loglikelihood {:.4f} | KL {:.6f}'.format(it, run_time, np.mean(lbs), np.max(lls), test_kl_div[-1])
                if it % lb_test_freq == 0:
                    run_time = -time.time()
                    test_lower_bound.append(self.lower_bound_estimate(n_particles, sample_sz=lb_test_sampsz))
                    run_time += time.time()
                    print '>>> Iter {} ({:.1f}s): Test Lower Bound {:.4f}'.format(it, run_time, test_lower_bound[-1])
                    
                lbs, lls = [], []
                run_time = -time.time()
                    
        return test_kl_div, test_lower_bound


class PHY(object):
    """
    A phylogenetic loglikelihood wrapper based on the 
    Phyloinfer package developed by Cheng Zhang.
    
    Parameters
    ----------
    data : ``list``, the molecular sequence data.
    taxa : ``list``, a list of the labels of the sequences.
    pden : ``np.array``, the stationary probability vector for the evolution model.
    subModel : ``tuple``, parameters for the evolution model.
    
    Github
    ------
    .. [1] https://github.com/zcrabbit/PhyloInfer
    
    """
    
    def __init__(self, data, taxa, pden, subModel, unique_site=True, scale=0.1):
        self.pden = pden  # stationary distribution of the continuous time Markov chain evolution model.
        Qmodel, Qpara = subModel # parameters for the evolution model 
        if Qmodel == "JC":
            self.D, self.U, self.U_inv, self.rateM = pinf.rateM.decompJC()
        if Qmodel == "HKY":
            self.D, self.U, self.U_inv, self.rateM = pinf.rateM.decompHKY(pden, Qpara)
        if Qmodel == "GTR":
            AG, AC, AT, GC, GT, CT = Qpara
            self.D, self.U, self.U_inv, self.rateM = pinf.rateM.decompGTR(pden, AG, AC, AT, GC, GT, CT)
        
        # initialize the conditional likelihood vector
        if unique_site:
            self.L, self.site_counts = pinf.Loglikelihood.initialCLV(data, unique_site)
        else: 
            self.L, self.site_counts = pinf.Loglikelihood.initialCLV(data, unique_site), 1.0
        self.nsites = len(data[0])
        
        self.scale = scale  # the branch length prior(exponential) hyperparameter
        self.ntips = len(data)  # number of tips 
        self.taxa = taxa
        
        
    def init_tree(self, branch='random'):
        tree = pinf.Tree()
        tree.populate(self.ntips)
        tree.unroot()
        pinf.tree.init(tree, branch=branch)
        return tree
    
    def logprior(self, branch, grad=False):
        if not grad:
            return pinf.Logprior.phyloLogprior_exp(np.exp(branch), scale=self.scale) + np.sum(branch)
        else:
            return pinf.Logprior.phyloLogprior_exp(np.exp(branch), scale=self.scale, grad=True) * np.exp(branch) + 1.0
                
    def loglikelihood(self, tree, branch, grad=False, value_and_grad=False, batch_size=None):
        if batch_size:
            subset = np.random.choice(np.arange(self.L.shape[2]), size=batch_size, replace=False)
            L = self.L[:,:,subset]
            rescale = self.L.shape[2]/(batch_size*1.)
        else:
            L, rescale = self.L, 1.0
            
        if not grad:
            return rescale * pinf.Loglikelihood.phyloLoglikelihood(tree, np.exp(branch), self.D, self.U, self.U_inv, self.pden, L, site_counts=self.site_counts) 
        elif not value_and_grad:           
            return rescale * pinf.Loglikelihood.phyloLoglikelihood(tree, np.exp(branch), self.D, self.U, self.U_inv, self.pden, L, site_counts=self.site_counts, grad=True) * np.exp(branch) 
        else:
            Loglikelihood, gradient = pinf.Loglikelihood.phyloLoglikelihood(tree, np.exp(branch), self.D, self.U, self.U_inv, self.pden, L, site_counts=self.site_counts, grad=True, value_and_grad=True)
            return rescale * Loglikelihood, rescale * gradient * np.exp(branch)
                        

class SBN_VBPI(SBN):
    """
    Variational Bayesian Phylogenetic Inference (VBPI) via SBN. VBPI is the first
    variational framework for Bayesian phylogenetic inference that learns both 
    tree topologies and branch lengthes. To deal with the discrete phylogenetic
    trees, we use VIMCO and RWS (see the references below).
    
    Parameters
    ----------
    taxa : ``list``, a list of the labels of the sequences.    
    rootsplit_supp_dict : ``dict``, a dictionary of rootsplit support, usually 
                          obtained from some bootstrap run.
    subsplit_supp_dict: ``dict``, a dictionary of subsplit support, obtained 
                        similarly as above.
    data : ``list``, the molecular sequence data.
    pden : ``np.array``, the stationary probability vector for the evolution model.
    subModel : ``tuple``, parameters for the evolution model.    
    emp_tree_freq : ``dict``, optinal, a dictionary of the empirical probabilities 
                    of tree topologies, used for computing the KL divergence.
    
    References
    ----------
    .. [1] Cheng Zhang and Frederick A. Matsen IV. "Variational Bayesian Phylogenetic
           Inference",  In Proceedings of the 7th International Conference on Learning
           Representations (ICLR), 2019. (https://openreview.net/forum?id=SJVmjjR9FX)
    
    .. [2] Andriy Mnih and Danilo Rezende. Variational inference for monte carlo
           objectives. In Proceedings of the 33rd International Conference on Machine 
           Learning (ICML), 2016. (https://arxiv.org/abs/1602.06725)   
    
    .. [3] Jorg Bornschein and Yoshua Bengio. Reweighted wake-sleep. In Proceedings of
           the International Conference on Learning Representations (ICLR), 2015. 
           (https://arxiv.org/abs/1406.2751)
    
    """
    
    EPS = 1e-100
    def __init__(self, taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden, subModel, emp_tree_freq=None, 
                 scale=0.1, branch_distribution='Normal'):
        super(SBN_VBPI, self).__init__(taxa, rootsplit_supp_dict, subsplit_supp_dict)
        self.emp_tree_freq = emp_tree_freq
        if emp_tree_freq:
            self.trees , self.emp_freqs = zip(*emp_tree_freq.iteritems())
            self.emp_freqs = np.array(self.emp_freqs)           
            self.negDataEnt = np.sum(self.emp_freqs * np.log(np.maximum(self.emp_freqs, self.EPS)))
        # set up the phylogenetic model
        self.phylo = PHY(data, taxa, pden, subModel, scale=scale)
        self.log_p_tau = -np.sum(np.log(np.arange(3, 2*len(taxa)-3, 2)))
        
        
        self.loc_parser = ParamParser()
        self.shape_parser = ParamParser()
        for split in rootsplit_supp_dict:
            self.loc_parser.add_item(split)
            self.shape_parser.add_item(split)
        self.loc_parser.add_dict('rootsplits')
        self.shape_parser.add_dict('rootsplits')
        for parent in subsplit_supp_dict:
            if self.toBitArr.merge(parent).count() == self.ntaxa:
                for child in subsplit_supp_dict[parent]:
                    self.loc_parser.add_item(parent + child)
                    self.shape_parser.add_item(parent + child)
        
        self.brlen_distribution = getattr(distributions, branch_distribution)(2*self.ntaxa-3)
        self.num_brlen_params = self.loc_parser.num_params
        self.loc = np.zeros(self.num_brlen_params)
        self.shape = np.zeros(self.num_brlen_params)
        self.loc_parser.assign(self.loc, 'rootsplits', -2.0)
        self.shape_parser.assign(self.shape, 'rootsplits', 1.0)
        
    def kl_div(self):
        kl_div = 0.0
        for tree, wt in self.emp_tree_freq.iteritems():
            kl_div += wt * np.log(max(np.exp(self.tree_loglikelihood(tree)), self.EPS))
        kl_div = self.negDataEnt - kl_div
        return kl_div
        
            
    def get_loc(self, name):
        return self.loc_parser.get(self.loc, name)
    
    def get_shape(self, name):
        return self.shape_parser.get(self.shape, name)
        
    def get_indexes(self, names):
        return [self.loc_parser.get_indexes(name)[0] for name in names]
        
    @property
    def rootsplit_loc(self):
        return {split: self.get_loc(split) for split in self.rootsplit_supp_dict}
    
    @property
    def subsplit_loc(self):
        return {parent: {child: self.get_loc(parent+child) for child in self.subsplit_supp_dict[parent]} for parent in self.subsplit_supp_dict if self.toBitArr.merge(parent).count() == self.ntaxa}
    
    @property    
    def rootsplit_shape(self):
        return {split: self.get_shape(split) for split in self.rootsplit_supp_dict}
    
    @property
    def subsplit_shape(self):
        return {parent: {child: self.get_shape(parent+child) for child in self.subsplit_supp_dict[parent]} for parent in self.subsplit_supp_dict if self.toBitArr.merge(parent).count() == self.ntaxa}
        
    def save_params(self, filename, iter):
        with open(filename + '__CPDs.txt', 'a') as _CPDs_file:
            _CPDs_file.write('iter_' + str(iter) + '\t' + '\t'.join(str(num) for num in self._CPDs) + '\n')
        with open(filename + '_CPDs.txt', 'a') as CPDs_file:
            CPDs_file.write('iter_' + str(iter) + '\t' + '\t'.join(str(num) for num in self.CPDs) + '\n')
        with open(filename + '_loc.txt', 'a') as loc_file:
            loc_file.write('iter_' + str(iter) + '\t' + '\t'.join(str(num) for num in self.loc) + '\n')
        with open(filename + '_shape.txt', 'a') as shape_file:
            shape_file.write('iter_' + str(iter) + '\t' + '\t'.join(str(num) for num in self.shape) + '\n')
        
        
    def get_loc_shape(self, idxtosplitVec, subsplit_pair=None):
        if not subsplit_pair:
            loc, shape = zip(*[[self.get_loc(split), self.get_shape(split)] for split in idxtosplitVec])
        else:
            loc, shape = zip(*[[self.get_loc(split) + np.sum([self.get_loc(subsplit) for subsplit in subsplit_pair[split]]), 
                                self.get_shape(split) + np.sum([self.get_shape(subsplit) for subsplit in subsplit_pair[split]])] for split in idxtosplitVec])
        return np.array(loc), np.array(shape)
        
    def sample_branch(self, loc, shape, n_particles=1):
        return self.brlen_distribution.sample(loc, shape, n_particles)
            
    def collect_subsplit_pair(self, tree, nodetobitMap=None):
        subsplit_pair = defaultdict(list)
        node_to_split = {}
        if not nodetobitMap:
            nodetobitMap = self.node2bitMap(tree, bit_type='clade')
        
        for node in tree.traverse('postorder'):
            if not node.is_leaf() and not node.is_root():
                bipart_bitarr = min(nodetobitMap[child] for child in node.children)
                comb_parent_bipart_bitarr = ~nodetobitMap[node] + nodetobitMap[node]
                split = self.toBitArr.minor(nodetobitMap[node])
                node_to_split[node] = split
                subsplit_pair[split.to01()].append(comb_parent_bipart_bitarr.to01() + bipart_bitarr.to01())
        
        for node in tree.traverse('preorder'):
            if not node.is_root():
                if node.up.is_root():
                    bipart_bitarr = min(nodetobitMap[sister] for sister in node.get_sisters())
                else:
                    bipart_bitarr = min([nodetobitMap[node.get_sisters()[0]], ~nodetobitMap[node.up]])
                comb_parent_bipart_bitarr = nodetobitMap[node] + ~nodetobitMap[node]
                if not node.is_leaf():
                    split = node_to_split[node]
                else:
                    split = self.toBitArr.minor(nodetobitMap[node])
                subsplit_pair[split.to01()].append(comb_parent_bipart_bitarr.to01() + bipart_bitarr.to01())
        
        return subsplit_pair
        
    def branch_loglikelihood(self, branch, loc, shape, grad=False):
        return self.brlen_distribution.log_prob(branch, loc, shape, grad=grad)
        
    def brlen_reparam_grad(self, branch, loc, shape):
        return self.brlen_distribution.reparam_grad(branch, loc, shape)
        
    def branch_loglikelihood_param_grad(self, branch, loc, shape):
        return self.brlen_distribution.lp_param_grad(branch, loc, shape)
        
    def aggregate_loc_shape_grad(self, wts, samp_tree_list, gradient, samp_branch, loc, shape, idxtosplit_list, temp=1.0, subsplit_pair_list=None, batch_size=None):
        """
        Aggregate branch length gradients from multiple sampled tree topologies.
        
        Parameters
        ----------
        wts : ``np.array``, the weight vector for sampled tree topologies.
        samp_tree_list : ``list``, a list of sampled trees.
        gradient : ``np.ndarray``, a matrix of branch length gradient,
                    each row represents a tree topology.
        samp_branch : ``np.ndarray``, a matrix of sampled branches.
        loc, shape :  ``np.ndarray``, mean and std matrices for log branch lengths.
        idxtosplit_list : ``list``, a list of index to split list for 
                           sampled tree topologies.
        temp : ``float``, annealing temperature.
        subsplit_pair_list : ``list``, a list of dictionaries of primary
                              subsplit pairs for sampled trees.

        """
        
        loc_grad = np.zeros(self.num_brlen_params)
        shape_grad = np.zeros(self.num_brlen_params)
        if subsplit_pair_list is None:
            subsplit_pair_list = []
        for i, samp_tree in enumerate(samp_tree_list):
            indexes = self.get_indexes(idxtosplit_list[i])
            d_q_loc, d_q_shape = self.branch_loglikelihood_param_grad(samp_branch[i], loc[i], shape[i])
            d_log_probs_ratio= temp * gradient[i] + self.phylo.logprior(samp_branch[i], grad=True) - self.branch_loglikelihood(samp_branch[i], loc[i], shape[i], grad=True)
            d_reparam_loc, d_reparam_shape = self.brlen_reparam_grad(samp_branch[i], loc[i], shape[i])            
            d_loc = d_log_probs_ratio * d_reparam_loc - d_q_loc
            d_shape = d_log_probs_ratio * d_reparam_shape - d_q_shape

            loc_grad[indexes] = loc_grad[indexes] + wts[i] * d_loc
            shape_grad[indexes] = shape_grad[indexes] + wts[i] * d_shape
            if any(subsplit_pair_list):
                for j, split in enumerate(idxtosplit_list[i]):
                    for subsplit in subsplit_pair_list[i][split]:
                        index = self.loc_parser.get_indexes(subsplit)[0]
                        loc_grad[index] = loc_grad[index] + wts[i] * d_loc[j]
                        shape_grad[index] = shape_grad[index] + wts[i] * d_shape[j]
        
        return loc_grad, shape_grad
        
    def lower_bound_estimate(self, n_particles, rooted=False, sample_size=1000, use_subsplit_pair=False):
        lower_bound = np.empty(sample_size)
        for sample in range(sample_size):
            samp_tree_list = [self.sample_tree(rooted=rooted) for particle in range(n_particles)]
            nodetosplit_list = [self.node2bitMap(samp_tree, bit_type='split') for samp_tree in samp_tree_list]
            nodetoclade_list = [self.node2bitMap(samp_tree, bit_type='clade') for samp_tree in samp_tree_list]
            if use_subsplit_pair:
                subsplit_pair_list = [self.collect_subsplit_pair(samp_tree, nodetoclade_list[i]) for i, samp_tree in enumerate(samp_tree_list)]
            else:
                subsplit_pair_list = [None] * n_particles
            idxtosplit_list = [pinf.tree.namenum(samp_tree, self.taxa, nodetosplit_list[i]) for i, samp_tree in enumerate(samp_tree_list)]
            loc_list, shape_list = zip(*[self.get_loc_shape(idxtosplit, subsplit_pair_list[i]) for i, idxtosplit in enumerate(idxtosplit_list)])
            loc, shape = np.array(loc_list), np.array(shape_list)
            samp_branch = self.sample_branch(loc, shape)        
    
            loglikelihood = np.array([self.phylo.loglikelihood(samp_tree, samp_branch[i]) for i, samp_tree in enumerate(samp_tree_list)])
            comp_log_prob_ratios = np.array([self.phylo.logprior(samp_branch[i]) - self.tree_loglikelihood(samp_tree_list[i], nodetoclade_list[i]) for i in range(n_particles)]) - self.branch_loglikelihood(samp_branch, loc, shape) + self.log_p_tau
            log_prob_ratios = loglikelihood + comp_log_prob_ratios
            
            lower_bound[sample] = logmeanexp(log_prob_ratios)
            
        return lower_bound
        
    
    def importance_sampling_marginal_estimate(self, tree_list, sample_size=10, n_particles=1000, rooted=False, use_subsplit_pair=False):
        marginals = []
        for tree in tree_list:
            tree_copy = deepcopy(tree)
            nodetosplit = self.node2bitMap(tree_copy, bit_type='split')
            nodetoclade = self.node2bitMap(tree_copy, bit_type='clade')
            if use_subsplit_pair:
                subsplit_pair = self.collect_subsplit_pair(tree_copy, nodetoclade)
            else:
                subsplit_pair = None
            idxtosplit = pinf.tree.namenum(tree_copy, self.taxa, nodetosplit)
            loc, shape = self.get_loc_shape(idxtosplit, subsplit_pair)
            
            lower_bound = np.empty(sample_size)
            for sample in range(sample_size):
                samp_branch_array = self.sample_branch(loc, shape, n_particles=n_particles)
            
                loglikelihood = np.array([self.phylo.loglikelihood(tree_copy, samp_branch) for samp_branch in samp_branch_array])
                comp_log_prob_ratios = np.array([self.phylo.logprior(samp_branch) for samp_branch in samp_branch_array]) - self.branch_loglikelihood(samp_branch_array, loc, shape)
                log_prob_ratios = loglikelihood + comp_log_prob_ratios
                
                lower_bound[sample] = logmeanexp(log_prob_ratios)
            
            marginals.append(lower_bound)
        
        return np.asarray(marginals)
            
        
        
    def monte_carlo_estimate(self, n_particles, temp=1.0, rooted=False, use_subsplit_pair=False, batch_size=None):
        """
        Compute monte carlo estimates for variational inference (VIMCO and RWS).
        
        Parameters
        ----------
        n_particles : ``int``, number of samples for the training objective.
                       Should be greater than one for VIMCO.
        temp : ``float``, annealing temperature.
        use_subsplit_pair : ``boolean``, use primary subsplit pair if true.
         
        """
        
        samp_tree_list = [self.sample_tree(rooted=rooted) for particle in range(n_particles)]
        nodetosplit_list = [self.node2bitMap(samp_tree, bit_type='split') for samp_tree in samp_tree_list]
        nodetoclade_list = [self.node2bitMap(samp_tree, bit_type='clade') for samp_tree in samp_tree_list]
        if use_subsplit_pair:
            subsplit_pair_list = [self.collect_subsplit_pair(samp_tree, nodetoclade_list[i]) for i, samp_tree in enumerate(samp_tree_list)]
        else:
            subsplit_pair_list = [None] * n_particles
        idxtosplit_list = [pinf.tree.namenum(samp_tree, self.taxa, nodetosplit_list[i]) for i, samp_tree in enumerate(samp_tree_list)]
        loc_list, shape_list = zip(*[self.get_loc_shape(idxtosplit, subsplit_pair_list[i]) for i, idxtosplit in enumerate(idxtosplit_list)])
        loc, shape = np.array(loc_list), np.array(shape_list)
        samp_branch = self.sample_branch(loc, shape)
        
        loglikelihood, gradient = zip(*[self.phylo.loglikelihood(samp_tree, samp_branch[i], batch_size=batch_size, grad=True, value_and_grad=True) for i, samp_tree in enumerate(samp_tree_list)])
        loglikelihood, gradient = np.array(loglikelihood), np.concatenate(gradient).reshape(n_particles, -1)
        logq_tree, CPDs_grad = zip(*[self.tree_loglikelihood(samp_tree, nodetoclade, grad=True, value_and_grad=True) for samp_tree, nodetoclade in zip(*[samp_tree_list, nodetoclade_list])])
        logq_tree, CPDs_grad = np.array(logq_tree), np.concatenate(CPDs_grad).reshape(n_particles, -1)
        comp_log_prob_ratios = np.array([self.phylo.logprior(samp_branch[i]) - logq_tree[i] for i in range(n_particles)]) - self.branch_loglikelihood(samp_branch, loc, shape) + self.log_p_tau      
        log_prob_ratios = temp * loglikelihood + comp_log_prob_ratios
        _log_prob_ratios = loglikelihood + comp_log_prob_ratios
        
        return samp_tree_list, log_prob_ratios, _log_prob_ratios, loglikelihood, gradient, CPDs_grad, samp_branch, loc, shape, idxtosplit_list, subsplit_pair_list
        
    
    def rws(self, stepsz, maxiter=1000000, test_freq=1000, lb_test_freq=1000, lb_test_sampsz=1000, rooted=False, n_particles=20, anneal_freq=20000, anneal_rate=0.75, clip=100., batch_size=None,
            momentum=0.9, decay=0.0, sgd_solver='adam', init_temp=0.001, warm_start_interval=50000, use_subsplit_pair=False):
        test_kl_div = []
        test_lower_bound = []
        
        lbs, lls = [], []
        infer_opt = SGD_Server({'CPDs': self.num_CPDs, 'loc': self.num_brlen_params, 'shape': self.num_brlen_params}, momentum=momentum, decay=decay)
        if not isinstance(stepsz, dict):
            stepsz = {'CPDs': stepsz, 'loc': stepsz, 'shape': stepsz}
            
        run_time = -time.time()           
        for it in range(1, maxiter+1):
            temp = min(1., init_temp + it * 1.0/warm_start_interval)
            samp_tree_list, log_prob_ratios, _log_prob_ratios, loglikelihood, gradient, CPDs_grad, samp_branch, \
                loc, shape, idxtosplit_list, subsplit_pair_list = self.monte_carlo_estimate(n_particles, temp=temp, rooted=rooted, use_subsplit_pair=use_subsplit_pair, batch_size=batch_size)
            
            particle_wts = softmax(log_prob_ratios)
            lower_bound = logmeanexp(_log_prob_ratios)
            
            CPDs_grad = self.aggregate_CPDs_grad(update_wts, CPDs_grad, clip=clip)
            loc_grad, shape_grad = self.aggregate_loc_shape_grad(particle_wts, samp_tree_list, gradient, samp_branch, loc, shape, idxtosplit_list, temp=temp, subsplit_pair_list=subsplit_pair_list, batch_size=batch_size)
            update_dict = getattr(infer_opt, sgd_solver)(stepsz, {'CPDs': self._CPDs, 'loc': self.loc, 'shape': self.shape}, {'CPDs': CPDs_grad, 'loc': loc_grad, 'shape': shape_grad})
            self._CPDs = self._CPDs + update_dict['CPDs']
            self.CPDs = softmax_parser(self._CPDs, self.CPDs_parser, self.dict_names)
            self.loc = self.loc + update_dict['loc']
            self.shape = self.shape + update_dict['shape']
            
            lls.append(np.max(loglikelihood))
            lbs.append(lower_bound)
            
            if it % anneal_freq == 0:
                for var in stepsz:
                    stepsz[var] *= anneal_rate
            
            if it % test_freq == 0:
                run_time += time.time()
                test_kl_div.append(self.kl_div())                    
                print 'Iter {} ({:.1f}s): Lower Bound {:.4f} | Loglikelihood {:.4f} | KL {:.6f}'.format(it, run_time, np.mean(lbs), np.max(lls), test_kl_div[-1])
                if it % lb_test_freq == 0: 
                    run_time = -time.time()                  
                    test_lower_bound.append(np.mean(self.lower_bound_estimate(n_particles, sample_size=lb_test_sampsz, use_subsplit_pair=use_subsplit_pair)))
                    run_time += time.time()
                    print '>>> Iter {} ({:.1f}s): Test Lower Bound {:.4f}'.format(it, run_time, test_lower_bound[-1])
                    
                lbs, lls = [], []
                run_time = -time.time()
                    
        return test_kl_div, test_lower_bound             
            
           
    def vimco(self, stepsz, maxiter=1000000, test_freq=1000, lb_test_freq=1000, lb_test_sampsz=1000, rooted=False, n_particles=20, anneal_freq=20000, anneal_rate=0.75, clip=100., batch_size=None,
              momentum=0.9, decay=0.0, sgd_solver='adam', init_temp=0.001, warm_start_interval=50000, use_subsplit_pair=False, output_file=None, Process=''):
        test_kl_div = []
        test_lower_bound = []
        
        lbs, lls = [], []
        # var_dict= {'CPDs': self.num_CPDs, 'loc': self.num_brlen_params, 'shape': self.num_brlen_params}
        infer_opt = SGD_Server({'CPDs': self.num_CPDs, 'loc': self.num_brlen_params, 'shape': self.num_brlen_params}, momentum=momentum, decay=decay)
        if not isinstance(stepsz, dict):
            stepsz = {'CPDs': stepsz, 'loc': stepsz, 'shape': stepsz}
        
        run_time = -time.time()
        for it in range(1, maxiter+1):
            temp = min(1., init_temp + it * 1.0/warm_start_interval)
            samp_tree_list, log_prob_ratios, _log_prob_ratios, loglikelihood, gradient, CPDs_grad, samp_branch, \
                loc, shape, idxtosplit_list, subsplit_pair_list = self.monte_carlo_estimate(n_particles, temp=temp, rooted=rooted, use_subsplit_pair=use_subsplit_pair, batch_size=batch_size)
                
            particle_wts = softmax(log_prob_ratios)
            lower_bound = logmeanexp(log_prob_ratios)
            lower_bound_approx = logmeanexp(log_prob_ratios, exclude=True)
            _lower_bound = logmeanexp(_log_prob_ratios)
            
            if lower_bound == -np.inf:
                update_wts = -particle_wts
            else:
                update_wts = lower_bound - lower_bound_approx - particle_wts
            update_wts = np.clip(update_wts, -clip, clip)
            CPDs_grad = self.aggregate_CPDs_grad(update_wts, CPDs_grad, clip=clip)
            loc_grad, shape_grad = self.aggregate_loc_shape_grad(particle_wts, samp_tree_list, gradient, samp_branch, loc, shape, idxtosplit_list, temp=temp, subsplit_pair_list=subsplit_pair_list, batch_size=batch_size)
            update_dict = getattr(infer_opt, sgd_solver)(stepsz, {'CPDs': self._CPDs, 'loc': self.loc, 'shape': self.shape}, {'CPDs': CPDs_grad, 'loc': loc_grad, 'shape': shape_grad})
            self._CPDs = self._CPDs + update_dict['CPDs']
            self.CPDs = softmax_parser(self._CPDs, self.CPDs_parser, self.dict_names)
            self.loc = self.loc + update_dict['loc']
            self.shape = self.shape + update_dict['shape']
            
            lls.append(np.max(loglikelihood))
            lbs.append(_lower_bound)
            
            if it % anneal_freq == 0:
                # stepsz *= anneal_rate
                for var in stepsz:
                    stepsz[var] *= anneal_rate

            
            if it % test_freq == 0:
                run_time += time.time()
                if self.emp_tree_freq:
                    test_kl_div.append(self.kl_div())
                    # test_lower_bound.append(np.mean(lbs))
                    print Process + 'Iter {} ({:.1f}s): Lower Bound {:.4f} | Loglikelihood {:.4f} | KL {:.6f}'.format(it, run_time, np.mean(lbs), np.max(lls), test_kl_div[-1])
                else:
                    self.save_params(output_file, it)
                    print Process + 'Iter {} ({:.1f}s): Lower Bound {:.4f} | Loglikelihood {:.4f}'.format(it, run_time, np.mean(lbs), np.max(lls))
                if it % lb_test_freq == 0:
                    run_time = -time.time()
                    test_lower_bound.append(np.mean(self.lower_bound_estimate(n_particles, sample_size=lb_test_sampsz, use_subsplit_pair=use_subsplit_pair)))
                    run_time += time.time()
                    print Process + '<<< Iter {} ({:.1f}s): Test Lower Bound {:.4f}'.format(it, run_time, test_lower_bound[-1])
                    
                lbs, lls = [], []
                run_time = -time.time()
                    
        return test_kl_div, test_lower_bound         
        