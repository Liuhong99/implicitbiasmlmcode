import nltk
import random
import re
import numpy as np
import copy
import argparse
import csv
import tqdm
import os
from collections import Counter
from nltk.corpus import stopwords
from nltk.grammar import Nonterminal
from nltk.grammar import ProbabilisticProduction
from nltk.parse.generate import generate
from nltk.parse import RecursiveDescentParser
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk import induce_pcfg
from nltk import Nonterminal
from nltk import PCFG
from nltk import CFG
from nltk import pos_tag, ne_chunk
from nltk import tree
from nltk import Tree
from typing import Iterator, List, Tuple, Union
Symbol = Union[str, Nonterminal]

parser = argparse.ArgumentParser()
parser.add_argument('--rule-path', type=str, default='rule_deep_mlm.txt',
                    help='path to the pcfg rules')
parser.add_argument('--save-path', type=str, default='./',
                    help='path to save the text')
parser.add_argument('--sentence-num', type=int, default=10000,
                    help='number of generated sentences')
parser.add_argument('--length-limit', type=int, default=32,
                    help='limit of sentence length')
parser.add_argument('--task-type', type=str, default="B",
                    help='downstream task type', choices=["A", "B", "C"])

args = parser.parse_args()

class Generator(nltk.grammar.PCFG):
    def generate(self, n: int) -> Iterator[str]:
        for _ in range(n):
            yield self._generate_derivation(self.start())

    def _generate_derivation(self, nonterminal: Nonterminal) -> str:
        sentence: List[str] = []
        tree = ''
        symbol: Symbol
        derivation: str
        proba = 1
        cc = self._reduce_once(nonterminal)
        tree += '('
        tree += nonterminal.__str__()
        tree += ' '
        proba = cc[1]
        for symbol in cc[0]:
            if isinstance(symbol, str):
                derivation = symbol
                tree += derivation
                
            else:
                derivation, probb, tree1 = self._generate_derivation(symbol)
                proba *= probb
                tree += tree1
                
            if derivation != "":
                sentence.append(derivation)
        tree += ')'
                
        return " ".join(sentence), proba, tree

    def _reduce_once(self, nonterminal: Nonterminal) -> Tuple[Symbol]:
        c, prob = self._choose_production_reducing(nonterminal)
        return c.rhs(), prob

    def _choose_production_reducing(
        self, nonterminal: Nonterminal
    ) -> ProbabilisticProduction:
        productions: List[ProbabilisticProduction] = self._lhs_index[nonterminal]
        probabilities: List[float] = [production.prob() for production in productions]
        pairs = []
        for p,q in zip(productions, probabilities):
            pairs.append((p,q))
        return random.choices(pairs, weights=probabilities)[0]

def inside(sentence, grammar):
    # compute the probability of the root node given the sentence
    def __producers(rhs, prob):
        results = []

        productions = grammar._rhs_index[rhs]
        probabilities = [production.prob() for production in productions]

        for p,q in zip(productions, probabilities):
            results.append((p.lhs(),q * prob))

        return results

    def __producers_non(rhs, prob):
        results = []
        
        if rhs[0] not in grammar._rhs_index.keys():
            return []

        productions = grammar._rhs_index[rhs[0]]
        probabilities = [production.prob() for production in productions]

        for p,q in zip(productions, probabilities):
            if len(p.rhs()) > 1 and p.rhs()[1] == rhs[1]:
                results.append((p.lhs(),q * prob))

        return results

    def __to_tree(table, pointer, sentence, j, i, k):
        if pointer[j][i]: 
            rhs = '('
            nj1 = pointer[j][i][k][0][0]
            ni1 = pointer[j][i][k][0][1]
            nk1 = pointer[j][i][k][0][2]
            rhs += (__to_tree(table, pointer, sentence, nj1, ni1, nk1))

            nj2 = pointer[j][i][k][1][0]
            ni2 = pointer[j][i][k][1][1]
            nk2 = pointer[j][i][k][1][2]
            rhs += (__to_tree(table, pointer, sentence, nj2, ni2, nk2))
            
            rhs += ')'

        else: 
            rhs = sentence[i-1]

        tree = '(' + table[j][i][k][0]._symbol
        tree += ' '
        tree += rhs
        tree += ')'

        return tree
    
    def __print_table(table):
        for row in table:
            print(row[1:])
            
    sentence = sentence.split()
    length = len(sentence)
    table = [None] * (length)
    for j in range(length):
        table[j] = [None] * (length+1)
        for i in range(length+1):
            table[j][i] = []

    pointer = [None] * (length)
    for j in range(length):
        pointer[j] = [None] * (length+1)
        for i in range(length+1):
            pointer[j][i] = []

    for k in range(1, length+1):
        table[k-1][k].extend(__producers(sentence[k-1], 1))

    for i in range (1, length+1):
        for j in range(i-2, -1, -1):
            current_lhs_prob = {}
            for k in range(j+1, i):
                for l in range(len(table[j][k])):
                    for m in range(len(table[k][i])):
                        prob = table[j][k][l][1] * table[k][i][m][1]
                        rhs = (table[j][k][l][0], table[k][i][m][0])
                        lhs = __producers_non(rhs, prob)
                        if lhs:
                            for (lfs, prob) in lhs:
                                if lfs not in current_lhs_prob.keys():
                                    current_lhs_prob[lfs] = prob
                                else:
                                    current_lhs_prob[lfs] += prob
            table[j][i].extend([(xx, current_lhs_prob[xx]) for xx in current_lhs_prob.keys()])
    
    
    root_prob = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0, "S7": 0, "S8": 0, "S9": 0}
    for aa, bb in table[0][length]:
        if aa.symbol() in root_prob.keys():
            root_prob[aa.symbol()] += bb
    return root_prob

def inside_partial(sentence, grammar, length_target):
    def __producers(rhs, prob):
        results = []

        productions = grammar._rhs_index[rhs]
        probabilities = [production.prob() for production in productions]

        for p,q in zip(productions, probabilities):
            results.append((p.lhs(),q * prob))

        return results

    def __producers_non(rhs, prob):
        results = []
        
        if rhs[0] not in grammar._rhs_index.keys():
            return []

        productions = grammar._rhs_index[rhs[0]]
        probabilities = [production.prob() for production in productions]

        for p,q in zip(productions, probabilities):
            if len(p.rhs()) > 1 and p.rhs()[1] == rhs[1]:
                results.append((p.lhs(),q * prob))

        return results

    def __to_tree(table, pointer, sentence, j, i, k):
        if pointer[j][i]:
            rhs = '('

            nj1 = pointer[j][i][k][0][0]
            ni1 = pointer[j][i][k][0][1]
            nk1 = pointer[j][i][k][0][2]
            rhs += (__to_tree(table, pointer, sentence, nj1, ni1, nk1))

            nj2 = pointer[j][i][k][1][0]
            ni2 = pointer[j][i][k][1][1]
            nk2 = pointer[j][i][k][1][2]
            rhs += (__to_tree(table, pointer, sentence, nj2, ni2, nk2))
            
            rhs += ')'

        else: 
            rhs = sentence[i-1]

        tree = '(' + table[j][i][k][0]._symbol
        tree += ' '
        tree += rhs
        tree += ')'

        return tree
    
    def __print_table(table):
        for row in table:
            print(row[1:])
            
    sentence = sentence.split()
    length = len(sentence)
    table = [None] * (length)
    for j in range(length):
        table[j] = [None] * (length+1)
        for i in range(length+1):
            table[j][i] = []

    pointer = [None] * (length)
    for j in range(length):
        pointer[j] = [None] * (length+1)
        for i in range(length+1):
            pointer[j][i] = []

    for k in range(1, length+1):
        table[k-1][k].extend(__producers(sentence[k-1], 1))

    for i in range (1, length+1):
        for j in range(i-2, -1, -1):
            current_lhs_prob = {}
            for k in range(j+1, i):
                for l in range(len(table[j][k])):
                    for m in range(len(table[k][i])):
                        prob = table[j][k][l][1] * table[k][i][m][1]
                        rhs = (table[j][k][l][0], table[k][i][m][0])
                        lhs = __producers_non(rhs, prob)
                        if lhs:
                            for (lfs, prob) in lhs:
                                if lfs not in current_lhs_prob.keys():
                                    current_lhs_prob[lfs] = prob
                                else:
                                    current_lhs_prob[lfs] += prob
            table[j][i].extend([(xx, current_lhs_prob[xx]) for xx in current_lhs_prob.keys()])
    
    root_prob = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "S5": 0, "S6": 0, "S7": 0, "S8": 0, "S9": 0, 
    "A1_0": 0, "A1_1": 0, "A1_2": 0, "A1_3": 0, "A1_4": 0, "A1_5": 0, "A1_6": 0, "A1_7": 0, "A1_8": 0, "A1_9": 0,
    "A2_0": 0, "A2_1": 0, "A2_2": 0, "A2_3": 0, "A2_4": 0, "A2_5": 0, "A2_6": 0, "A2_7": 0, "A2_8": 0, "A2_9": 0,
    "A3_0": 0, "A3_1": 0, "A3_2": 0, "A3_3": 0, "A3_4": 0, "A3_5": 0, "A3_6": 0, "A3_7": 0, "A3_8": 0, "A3_9": 0,
    "A4_0": 0, "A4_1": 0, "A4_2": 0, "A4_3": 0, "A4_4": 0, "A4_5": 0, "A4_6": 0, "A4_7": 0, "A4_8": 0, "A4_9": 0 }
    
    for aa, bb in table[0][length_target]:
        if aa.symbol() in root_prob.keys():
            root_prob[aa.symbol()] += bb
    return root_prob



def labeling(sent, generator, task_type):
    if task_type == "A":
        root_prob = inside(sent, generator)
        return max(root_prob, key=root_prob.get)
    elif task_type == "B":
        root_prob = inside_partial(sent, generator, 4)
        return max(root_prob, key=root_prob.get)
    else:
        root_prob = inside_partial(sent, generator, 2)
        return max(root_prob, key=root_prob.get)
    return 

with open(args.rule_path) as f:
    x = f.readlines()

xp = []
for xx in x:
    n = len(xx.split())
    if n==3:
        if "S" == xx.split()[0][0]:
            xp.append(xx.split()[0] + ' -> ' + xx.split()[1] + ' [' + str((float(xx.split()[2].strip('\n')))) + ']\n')
        else:
            xp.append(xx.split()[0] + ' -> ' + '\'' + xx.split()[1] + '\'  [' + str((float(xx.split()[2].strip('\n')))) + ']\n')
    else:
            xp.append(xx.split()[0] + ' -> ' + xx.split()[1] + '  ' + xx.split()[2]  + '  [' + str((float(xx.split()[3].strip('\n')))) + ']\n')

xq = {}
for xx in xp:
    if xx.split('[')[0] not in xq:
        xq[xx.split('[')[0]] = float(xx.split('[')[-1].split(']')[0])
    else:
        xq[xx.split('[')[0]] += float(xx.split('[')[-1].split(']')[0])
xqq = []
for xxx in xq.keys():
    xqq.append(xxx + '[' + str(xq[xxx]) + ']\n')

generator = Generator.fromstring(''.join(xqq)) 

vocab = set()
for prod in generator.productions():
    for rhs in prod.rhs():
        if isinstance(rhs, str):
            vocab.add(rhs)
            
vocab = list(vocab)
vocab.sort()

n=args.sentence_num
sentences = []
while len(sentences) < n:
    for sentence in generator.generate(1):
        if len(sentence[0].split(' ')) < args.length_limit and len(sentence[0].split(' ')) > 4:
            sentences.append((sentence[0]))

labels = []
for sent in tqdm.tqdm(sentences):
    labels.append(labeling(sent, generator, args.task_type))
    
if args.task_type == "A":
    C2i={'S1':0 , 'S2': 1, 'S3': 2, 'S4': 3, 'S5': 4, 'S6': 5, 'S7': 6, 'S8': 7, 'S9': 8}
else:
    nont = set()
    for prod in generator.productions():
        nont.add(prod.lhs()) 
    nont = list(nont)
    nont.sort()
    nont_dict = {}
    for i, nn in enumerate(nont):
        nont_dict[nn] = i
    C2i={}
    for opp in nont_dict.keys():
        C2i[opp.symbol()] = nont_dict[opp]

with open(os.path.join(args.save_path, 'pcfg_' + args.task_type + '.csv'), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['text'] + ['label'])
    for ii in range(args.sentence_num):
        writer.writerow([sentences[ii], C2i[labels[ii]]])