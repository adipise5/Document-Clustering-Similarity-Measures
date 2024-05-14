from __future__ import division
import nltk
import random
import re, pprint, os, numpy,sys

from nltk import cluster
from nltk.cluster import KMeansClusterer
from nltk.cluster import cosine_distance, euclidean_distance
from nltk.cluster.util import jaccard_distance
from collections import defaultdict
from sklearn.metrics.cluster import entropy
#!python
# -*- coding: utf-8 -*-

import codecs
from HWNet import isHinWord

'''
I have used the information gathered using Hindi Wordnet to detect
prefixes and suffixes. This information is used to create two lists
each for prefixes and suffixes:
***fix_list: This contains the top ***fixes that have a confidence of 0.5 or more
unsafe_***fix_list: This contains top ***fixes with a confidence from 0 till 0.5
'''

def remove_infix(word, infix):
    if infix in word:
        i = word.index(infix)
        j = i+len(infix)

        return word[:i] + word[j:]
    else:
        return word


prefix_list = ['मुस्']
unsafe_prefix_list = ['प्र','ब','वि','सं','स','सु','क','दि','परि',
                    'महा','जन','कार्य','राम','अ','आ','सि',
                    'उप','अनु','अधि','सम्']

suffix_list = ['ों','ने','ओं','कर','ते','ें','ो','गा','एं','गे','एँ',
                    'ंगे']
unsafe_suffix_list = ['े','ं','ी','ा','ता','ियों','यों','ती','या',
                    'गी','ाने','जी','न','त','ई','ये']

# Unused at the moment. Will be the (default) output format of the stemmer.
# This is mainly so more information can be passed to the user if he/she so desires.
# There shall be an option during the initialization of the Stemmer class to
#  choose the output format.
class Stem(object):
    word = ''
    confidence = 0
    origin = ''
    def __init__(self):
        self.word = ''
        self.confidence = 0
    def __init__(self,word):
        self.word = word
        self.confidence = 1
    def __init__(self,word,confidence):
        self.word = word
        self.confidence = confidence
    def __repr__(self):
        print self.word
    def setConfidence(self, confidence):
        self.confidence = confidence
    def getConfidence(self):
        return self.confidence


# The main class. This is the one that shall do all the hard work.
class Stemmer(object):
    return_string = False
    def __init__(self):
        pass
    def __init__(self,**kwargs):
        for key, value in kwargs:
            if key == 'return_string':
                self.return_string = bool(value)

    def stem(self, word):
        #word = self._clean_word(word)

        # Rule Based
        # Verbs:
        if word.endswith('ना'): word = word[:-2]

        # Nouns:
        pass

        # Adjectives:
        pass

        # Adverbs:
        pass

        # Other:
        pass

        if isHinWord(word) or isHinWord(word+'ना'):
            return word

        word = self._remove_suffixes(word)
        if isHinWord(word) or isHinWord(word+'ना'):
            return word

        word = self._remove_prefixes(word)
        if isHinWord(word) or isHinWord(word+'ना'):
            return word

        word = self._edit_distance(word)

        return word

    def load_corpus(self, text):
        pass

    # Doesn't work, for some reason.
    def _clean_word(self,word):
        clean=''
        for lett in word:
            if ord(lett)>128:
                clean+=lett
        return clean

    def _remove_prefixes(self,word):
        # Looks at all prefixes and sees if removing one of them leads to
        # a confirmed valid word.
        for elem in unsafe_prefix_list+prefix_list:
            if word.startswith(elem):
                word1 = word[len(elem):]
                if isHinWord(word1):
                    return word1
        
        # Removes all possible prefixes from list.
        for elem in prefix_list:
            if word.startswith(elem):
                word = word[len(elem):]
                if isHinWord(word):
                    return word
        return word

    def _remove_suffixes(self,word):
        # Looks at all suffixes and sees if removing one of them leads to
        # a confirmed valid word.
        for elem in unsafe_suffix_list+suffix_list:
            if word.endswith(elem):
                word1 = word[:-len(elem)]
                if isHinWord(word1):
                    return word1

        # Removes all possible suffixes from list.
        for elem in suffix_list:
            if word.endswith(elem):
                word = word[:-len(elem)]
                if isHinWord(word):
                    return word
        return word

    def _edit_distance(self, word):
        return word

# Code to read in a directory of text files, create nltk.Text objects out of them,
# load an nltk.TextCollection object and create a BOW with TF*IDF values.

# First set the variable path to the directory path.  Use
# forward slashes (/), even on Windows.  Make sure you
# leave a trailing / at the end of this variable.

# Path for mac OS X will look something like this:
path = ""
# Empty list to hold text documents.
texts = []
stemmer = Stemmer()
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    
    return stemmed


# Iterate through the  directory and build the collection of texts for NLTK.
dict1 = {}
dict1 = defaultdict(lambda:0,dict1)
for subdir, dirs, files in os.walk(path):
    for file in sorted(files):
        url = subdir + os.path.sep + file
        f = open(url);
        raw = f.read()
        f.close()
        tokens = nltk.word_tokenize(raw) 
        tokens = stem_tokens(tokens, stemmer)
        
        text = nltk.Text(tokens)
        for x in tokens:
            dict1[x]+=1
        texts.append(text)

print "Prepared ", len(texts), " documents..."
print "They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]"

#Load the list of texts into a TextCollection object.
collection = nltk.TextCollection(texts)
print "Created a collection of", len(collection), "terms."

#get a list of unique terms
unique_terms = list(set(collection))
def cnt(x):
    return dict1[x]
unique_terms.sort(key=cnt,reverse=True)
print "Unique terms found: ", len(unique_terms)
newlist = []
for x in collection:
    if x in unique_terms[:3000]:
        newlist.append(x)

newcollection = nltk.TextCollection(newlist)

# Function to create a TF*IDF vector for one document.  For each of
# our unique words, we have a feature which is the td*idf for that word
# in the current document
def TFIDF(document):
    word_tfidf = []
    for word in unique_terms[:3000]:
        word_tfidf.append(newcollection.tf_idf(word,document))
    return word_tfidf

### And here we actually call the function and create our array of vectors.
vectors = [numpy.array(TFIDF(f)) for f in texts if len(f) != 0]
print "Vectors created."
# print "First 10 words are", unique_terms[:10]
# print "First 10 stats for first document are", vectors[0][0:10]


# clusterer = KMeansClusterer(3,euclidean_distance)
# clusters = clusterer.cluster(vectors,assign_clusters=True, trace=False)
# # means=clusterer.means()
# print entropy(clusters)
# print path

import sys
from nltk.cluster.util import manhattan_distance, chebyschev_distance,pearson_distance,jaccard_distance
from nltk.cluster import cosine_distance, euclidean_distance

mets = ['cosine','euclidean','manhattan','chebyschev','pearson','jaccard']

import os
f=open('outh2.txt','w')
for met in mets:


    if(met=='cosine'):
        metric = cosine_distance
    elif(met=='euclidean'):
        metric = euclidean_distance
    elif(met=='manhattan'):
        metric = manhattan_distance
    elif(met=='chebyschev'):
        metric = chebyschev_distance
    elif(met=='pearson'):
        metric = pearson_distance
    elif(met=='jaccard'):
        metric = jaccard_distance

    a=0.0
    b=0.0
    for x in range(30):
        clusterer = KMeansClusterer(3,metric,avoid_empty_clusters=True)
        clusters = clusterer.cluster(vectors,assign_clusters=True, trace=False)
        # means=clusterer.means()
        a+=entropy(clusters)

        labels=[]

        cnt = 0
        lcnt=[]
        for root,dirs,files in os.walk(path):
            if(len(files) > 0):
                cnt+=1
                lcnt.append(0)
                for i in range(len(files)):
                    labels.append(cnt-1)
        ans=0
        for i in range(0,3):
            for x in range(0,cnt):
                lcnt[x]=0
            for j in range(0,len(labels)):
                if(clusters[j] == i):
                    lcnt[labels[j]]+=1
            ans += max(lcnt)
        b += (float(ans)/len(labels))
    a=a/5.0
    b=b/5.0
    f.write(path+'\n'+ met+ '\n' +'Purity:')
    f.write(str(b))
    f.write('\n' +'Entropy:')
    f.write(str(a))
    f.write('\n' + '--------------------'+ '\n')
    print "Entropy:",a
    print "Purity:",b
f.close()
