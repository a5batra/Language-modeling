#!/bin/python

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import collections
from collections import defaultdict
from math import log
import sys

# Python 3 backwards compatibility tricks
if sys.version_info.major > 2:

    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))

    def unicode(*args, **kwargs):
        return str(*args, **kwargs)

class LangModel:
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        for s in corpus:
            self.fit_sentence(s)
        self.norm()

    def perplexity(self, corpus):
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        num_words = 0.0
        sum_logprob = 0.0
        for s in corpus:
            num_words += len(s) + 1 # for EOS
            sum_logprob += self.logprob_sentence(s)
        return -(1.0/num_words)*(sum_logprob)

    def logprob_sentence(self, sentence):
        p = 0.0
        for i in xrange(len(sentence)):
            p += self.cond_logprob(sentence[i], sentence[:i])
        p += self.cond_logprob('END_OF_SENTENCE', sentence)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.model = dict()
        self.lbackoff = log(backoff, 2)

    def inc_word(self, w):
        if w in self.model:
            self.model[w] += 1.0
        else:
            self.model[w] = 1.0

    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for word in self.model:
            tot += self.model[word]
        ltot = log(tot, 2)
        for word in self.model:
            self.model[word] = log(self.model[word], 2) - ltot

    def cond_logprob(self, word, previous):
        if word in self.model:
            return self.model[word]
        else:
            return self.lbackoff

    def vocab(self):
        return self.model.keys()

class Bigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.unigram = dict()
        self.bigram = dict()
        self.lbackoff = log(backoff, 2)

    def inc_bigram(self, bi):
        if bi[1] in self.unigram:
            self.unigram[bi[1]] += 1.0
        else:
            self.unigram[bi[1]] = 1.0
        if (bi[0], bi[1]) in self.bigram:
            self.bigram[bi] += 1.0
        else:
            self.bigram[bi] = 1.0
    
    def fit_sentence(self, sentence):
        start = ['START_OF_SENTENCE']
        end = ['END_OF_SENTENCE']
        sentence = start + sentence + end
        for i in range(1, len(sentence)):
            self.inc_bigram((sentence[i-1], sentence[i]))
        self.inc_bigram((sentence[-1], 'END_OF_SENTENCE'))

    def norm(self):
        """Normalize and convert to log2-probs."""
        tot = 0.0
        for bi in self.bigram:
            tot += self.bigram[bi]
        ltot = log(tot, 2)
        for bi in self.bigram:
            self.bigram[bi] = log(self.bigram[bi], 2) - ltot
            
    def cond_logprob(self, word, previous):
        bi = (previous, word)
        cond_logprob = 0.0
        if bi in self.bigram:
            cond_logprob += self.bigram[bi] / self.unigram[previous]
        else:
            cond_logprob += self.lbackoff
        return cond_logprob
     
    def vocab(self):
        return self.unigram.keys()


class Trigram(LangModel):
    def __init__(self, backoff = 0.000001):
        self.lbackoff = log(backoff, 2)
        self.unigrams = dict()
        self.bigrams = dict()
        self.trigrams = dict()
        self.bigrams_history = defaultdict(int)
        self.trigrams_history = defaultdict(int)
        self.vocabulary = set(['END_OF_SENTENCE'])
#         self.d1 = d1
#         self.d2 = d2
        
    def inc_trigram_bigram_unigram(self, tri):
        if tri in self.trigrams:
            self.trigrams[tri] += 1.0
        else:
            self.trigrams[tri] = 1.0
        
        bi = (tri[1], tri[2])
        if bi in self.bigrams:
            self.bigrams[bi] += 1.0
        else:
            self.bigrams[bi] = 1.0
        uni = tri[2]
        if uni in self.unigrams:
            self.unigrams[uni] += 1.0
        else:
            self.unigrams[uni] = 1.0
#         Updating the histories for trigram and bigram
        if (tri[0], tri[1]) in self.trigrams_history:
            self.trigrams_history[(tri[0], tri[1])] += 1.0
        else:
            self.trigrams_history[(tri[0], tri[1])] = 1.0
        if bi[0] in self.bigrams_history:
            self.bigrams_history[bi[0]] += 1.0
        else:
            self.bigrams_history[bi[0]] = 1.0
    
    def fit_sentence(self, sentence):
        start = ['START_OF_SENTENCE', 'START_OF_SENTENCE']
        end = ['END_OF_SENTENCE']
        sentence = start + sentence + end
        for i in range(2, len(sentence)):
            self.inc_trigram_bigram_unigram((sentence[i-2], sentence[i-1], sentence[i]))
            word = sentence[i]
            self.vocabulary.add(word)
        # Including the EOS in our list of possible trigrams
        if len(sentence) > 3:
            self.inc_trigram_bigram_unigram((sentence[-2], sentence[-1], 'END_OF_SENTENCE'))
        
    def norm(self):
        """Normalize and convert to log2-probs."""
        # For trigrams
        tot = 0.0
        # Calculating the total number of trigrams
        for tri in self.trigrams:
            tot += self.trigrams[tri]
        ltot = log(tot, 2) 
        for tri in self.trigrams:
            self.trigrams[tri] = log(self.trigrams[tri], 2) - ltot
        # For bigrams
        tot = 0.0
        for bi in self.bigrams:
            tot += self.bigrams[bi]
        ltot = log(tot, 2)
        for bi in self.bigrams:
            self.bigrams[bi] = log(self.bigrams[bi], 2) - ltot
#         # For unigrams
        tot = 0.0
        for uni in self.unigrams:
            tot += self.unigrams[uni]
        ltot = log(tot, 2)
        for uni in self.unigrams:
            self.unigrams[uni] = log(self.unigrams[uni], 2) - ltot
#         # For trigram history (context)
        tot = 0.0
        for context_words in self.trigrams_history:
            tot += self.trigrams_history[context_words]
        ltot = log(tot, 2)
        for context_words in self.trigrams_history:
            self.trigrams_history[context_words] = log(self.trigrams_history[context_words], 2) - ltot
#         # For bigram history (context)
        tot = 0.0
        for context_word in self.bigrams_history:
            tot += self.bigrams_history[context_word]
        ltot = log(tot, 2)
        for context_word in self.bigrams_history:
            self.bigrams_history[context_word] = log(self.bigrams_history[context_word], 2) - ltot  
        
    def cond_logprob(self, word, previous):
        """Calculates the log2 conditional probability of the word
            given previous words"""
        if len(previous) == 0:
            prev_word = 'START_OF_SENTENCE'
            prev_to_prev_word = 'START_OF_SENTENCE'
        elif len(previous) == 1:
            prev_word = previous[-1]
            prev_to_prev_word = 'START_OF_SENTENCE'
        else:
            prev_word = previous[-1]
            prev_to_prev_word = previous[-2]
        
        tri = (prev_to_prev_word, prev_word, word)
        bi = (prev_to_prev_word, prev_word)
        uni = word
        vocab_len = len(self.vocabulary)
        cond_prob = -8
     
#         if tri in self.trigrams:
#             cond_prob += self.d2 * (self.trigrams[tri]) / (self.trigrams_history[(tri[0], tri[1])])
# #             cond_prob += self.d2 * (self.trigrams[tri]) / (self.bigrams[(tri[0], tri[1])])
#         if bi in self.bigrams:
#             cond_prob += self.d1 * (self.bigrams[bi]) / (self.bigrams_history[bi[0]])
#         #         elif bigram in self.bigrams:
#         #             cond_prob += 1.0 / (self.trigrams_history[(trigram[0], trigram[1])] + vocab_len)
# #         if unigram in self.unigrams:
# #             cond_prob += self.d1 * self.unigrams[unigram]
#         else:
#             cond_prob += self.lbackoff
    
        if tri in self.trigrams:
            cond_prob += (self.trigrams[tri] + 0.001) / (self.trigrams_history[(tri[0], tri[1])] + 0.001 * vocab_len)
        elif bi in self.bigrams:
            cond_prob += (0.001) / (self.trigrams_history[(tri[0], tri[1])] + 0.001 * vocab_len)
        else:
            cond_prob += 1.0 / vocab_len
        return cond_prob
    
    def vocab(self):
        vocab = list(self.vocabulary)
        return vocab
        
        
        