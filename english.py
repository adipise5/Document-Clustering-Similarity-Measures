from __future__ import division
import nltk
import random
import re, pprint, os, numpy
import sys
from nltk import cluster
from nltk.cluster import KMeansClusterer
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
stemmer = PorterStemmer()
from sklearn.metrics.cluster import entropy
# Code to read in a directory of text files, create nltk.Text objects out of them,
# load an nltk.TextCollection object and create a BOW with TF*IDF values.

# First set the variable path to the directory path.  Use
# forward slashes (/), even on Windows.  Make sure you
# leave a trailing / at the end of this variable.

path = sys.argv[1]

# Empty list to hold text documents.
texts = []
stopwords = set(nltk.corpus.stopwords.words('english'))
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        if item not in stopwords:
            stemmed.append(stemmer.stem(item))    
    return stemmed

# Iterate through the  directory and build the collection of texts for NLTK.
dict1 = {}
dict1 = defaultdict(lambda:0,dict1)
for subdir, dirs, files in os.walk(path):
    for file in files:
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

import sys
from nltk.cluster.util import manhattan_distance, chebyschev_distance,pearson_distance,jaccard_distance
from nltk.cluster import cosine_distance, euclidean_distance

mets = ['cosine','euclidean','manhattan','chebyschev','pearson','jaccard']

import os
f=open(sys.argv[2],'w')
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
   
    a=a/30.0
    b=b/30.0
    f.write(path+'\n'+ met+ '\n' +'Purity:'+str(b)+ '\n' +'Entropy:'+str(a)+ '\n' '--------------------'+ '\n')
    print "Entropy:",a
    print "Purity:",b
f.close()