"""
Evaluate accuracy of rexgen_direct model from https://github.com/connorcoley/rexgen_direct
based on https://github.com/connorcoley/rexgen_direct/blob/master/rexgen_direct/rank_diff_wln/directcandranker.py
"""

import argparse
from collections import Counter
import math, sys, random
from optparse import OptionParser
import threading
from multiprocessing import Queue
import rdkit
from rdkit import Chem
import os
import numpy as np 
import tensorflow as tf
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from rexgen_direct.core_wln_global.directcorefinder import DirectCoreFinder 
from rexgen_direct.scripts.eval_by_smiles import edit_mol
from rexgen_direct.rank_diff_wln.nn import linearND, linear
from rexgen_direct.rank_diff_wln.mol_graph_direct_useScores import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph, smiles2graph, bond_types
from rexgen_direct.rank_diff_wln.models import *

def load_reactions(data_path):
    rxn_tuple_list = []
    with open(data_path, 'r') as data:
        for line in data:
            reaction_str, edit_str = line.strip("\r\n ").split()
            reactant_str, product_str = reaction_str.split(">>")
            rxn_tuple_list.append((reactant_str, product_str))
    return rxn_tuple_list

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

class DirectCandRanker():
    def __init__(self, hidden_size, depth, core_size,
            max_ncand, topk):
        self.hidden_size = hidden_size 
        self.depth = depth 
        self.core_size = core_size 
        self.max_ncand = max_ncand 
        self.topk = topk 

    def load_model(self, model_path):
        hidden_size = self.hidden_size 
        depth = self.depth 
        core_size = self.core_size 
        max_ncand = self.max_ncand 
        topk = self.topk 

        self.graph = tf.Graph()
        with self.graph.as_default():
            input_atom = tf.placeholder(tf.float32, [None, None, adim])
            input_bond = tf.placeholder(tf.float32, [None, None, bdim])
            atom_graph = tf.placeholder(tf.int32, [None, None, max_nb, 2])
            bond_graph = tf.placeholder(tf.int32, [None, None, max_nb, 2])
            num_nbs = tf.placeholder(tf.int32, [None, None])
            core_bias = tf.placeholder(tf.float32, [None])
            self.src_holder = [input_atom, input_bond, atom_graph, bond_graph, num_nbs, core_bias]

            graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs) 
            with tf.variable_scope("mol_encoder"):
                fp_all_atoms = rcnn_wl_only(graph_inputs, hidden_size=hidden_size, depth=depth)

            reactant = fp_all_atoms[0:1,:]
            candidates = fp_all_atoms[1:,:]
            candidates = candidates - reactant
            candidates = tf.concat([reactant, candidates], 0)

            with tf.variable_scope("diff_encoder"):
                reaction_fp = wl_diff_net(graph_inputs, candidates, hidden_size=hidden_size, depth=1)

            reaction_fp = reaction_fp[1:]
            reaction_fp = tf.nn.relu(linear(reaction_fp, hidden_size, "rex_hidden"))

            score = tf.squeeze(linear(reaction_fp, 1, "score"), [1]) + core_bias # add in bias from CoreFinder
            scaled_score = tf.nn.softmax(score)

            tk = tf.minimum(topk, tf.shape(score)[0])
            _, pred_topk = tf.nn.top_k(score, tk)
            self.predict_vars = [score, scaled_score, pred_topk]

            self.session = tf.Session()
            saver = tf.train.Saver()
            saver.restore(self.session, model_path)
    
    def predict(self, react, top_cand_bonds, top_cand_scores=[], scores=True, top_n=100):
        '''react: atom mapped reactant smiles
        top_cand_bonds: list of strings "ai-aj-bo"'''

        cand_bonds = []
        if not top_cand_scores:
            top_cand_scores = [0.0 for b in top_cand_bonds]
        for i, b in enumerate(top_cand_bonds):
            x,y,t = b.split('-')
            x,y,t = int(float(x))-1,int(float(y))-1,float(t)

            cand_bonds.append((x,y,t,float(top_cand_scores[i])))

        while True:
            src_tuple,conf = smiles2graph(
                react, None, cand_bonds, None, core_size=self.core_size, 
                cutoff=self.max_ncand, testing=True
            )
            if len(conf) <= self.max_ncand:
                break
            ncore -= 1

        feed_map = {x:y for x,y in zip(self.src_holder, src_tuple)}
        cur_scores, cur_probs, candidates = self.session.run(self.predict_vars, feed_dict=feed_map)
        

        idxfunc = lambda a: a.GetAtomMapNum()
        bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                      Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        bond_types_as_double = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 1.5: 4}

        # Don't waste predictions on bond changes that aren't actually changes
        rmol = Chem.MolFromSmiles(react)
        rbonds = {}
        for bond in rmol.GetBonds():
            a1 = idxfunc(bond.GetBeginAtom())
            a2 = idxfunc(bond.GetEndAtom())
            t = bond_types.index(bond.GetBondType()) + 1
            a1,a2 = min(a1,a2),max(a1,a2)
            rbonds[(a1,a2)] = t

        cand_smiles = []; cand_scores = []; cand_probs = [];
        for idx in candidates:
            cbonds = []
            # Define edits from prediction
            for x,y,t,v in conf[idx]:
                x,y = x+1,y+1
                if ((x,y) not in rbonds and t > 0) or ((x,y) in rbonds and rbonds[(x,y)] != t):
                    cbonds.append((x, y, bond_types_as_double[t]))
            pred_smiles = edit_mol(rmol, cbonds)
            cand_smiles.append(pred_smiles)
            cand_scores.append(cur_scores[idx])
            cand_probs.append(cur_probs[idx])

        outcomes = []
        if scores:
            for i in range(min(len(cand_smiles), top_n)):
                outcomes.append({
                    'rank': i + 1,
                    'smiles': cand_smiles[i],
                    'score': cand_scores[i],
                    'prob': cand_probs[i],
                })
        else:
            for i in range(min(len(cand_smiles), top_n)):
                outcomes.append({
                    'rank': i + 1,
                    'smiles': cand_smiles[i],
                })

        return outcomes

def check_predictions(react, directcorefinder, directcandranker):
    product_mol = Chem.MolFromSmiles(react[1])
    # Erase atom mapping
    for atom in product_mol.GetAtoms():
        atom.SetAtomMapNum(0)
    # Get canonical form of product to string match against outcomes
    product_smiles = Chem.MolToSmiles(product_mol).split(".")

    (react, bond_preds, bond_scores, cur_att_score) = directcorefinder.predict(react[0])
    outcomes = directcandranker.predict(react, bond_preds, bond_scores)
    for outcome in outcomes:
        # Outcome includes reagents (but product may not) so check if list is subset
        # Use Counter to account for duplicate molecules
        if not Counter(product_smiles) - Counter(outcome["smiles"]):
            return True
    return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run rexgen_direct model")
    parser.add_argument('--cand-model-path', type=str, required=True)
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--max_ncand', type=int, default=1500)
    parser.add_argument('--cand-hidden-size', type=int, default=500)
    parser.add_argument('--cand-depth', type=int, default=3)
    parser.add_argument('--cand-core-size', type=int, default=16)
    args = parser.parse_args()

    # Load all reactions to evaluate
    reactions = load_reactions(args.data_path)

    # Load model to predict bond changes from reactant graph
    directcorefinder = DirectCoreFinder()
    directcorefinder.load_model()

    # Load candidate enumeration and ranking model
    directcandranker = DirectCandRanker(
        args.cand_hidden_size, 
        args.cand_depth, 
        args.cand_core_size,
        args.max_ncand, 
        args.top_k
    )
    directcandranker.load_model(args.cand_model_path)

    pbar = tqdm(total=len(reactions))
    correct = 0
    total = 0
    for react in reactions:
        try:
            correct += check_predictions(react, directcorefinder, directcandranker)
            total += 1
        except:
            print("Invalid reactants string: " + react[0])
            
        pbar.set_postfix(
            in_top_k=correct, 
            total_preds=total, 
            k=args.top_k, 
            top_k_acc=round(100 * correct/total, 2)
        )
        pbar.update()
