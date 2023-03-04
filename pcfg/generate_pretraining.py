import nltk
import random
import numpy as np
import copy
import argparse
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
parser.add_argument('--sentence-num', type=int, default=1000000,
                    help='number of generated sentences')
parser.add_argument('--shard-num', type=int, default=200,
                    help='number of shards')
parser.add_argument('--length-limit', type=int, default=32,
                    help='limit of sentence length')

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

n=args.sentence_num
print('Generating training shards')
for i in tqdm.tqdm(range(args.shard_num)):
    sentences = []
    while len(sentences) < n:
        for sentence in generator.generate(1):
            if len(sentence[0].split(' ')) < args.length_limit:
                sentences.append((sentence[0]))
    with open(os.path.join(args.save_path, 'training' + str(i) + '.txt'), 'w') as f:
        for sent in sentences:
            f.write(sent + '\n')
print('Generating test shards')
for i in range(10):
    n=10000
    sentences = []
    while len(sentences) < n:
        for sentence in generator.generate(1):
            if len(sentence[0].split(' ')) < args.length_limit:
                sentences.append((sentence[0]))
    with open(os.path.join(args.save_path, 'test' + str(i) + '.txt'), 'w') as f:
        for sent in sentences:
            f.write(sent + '\n')