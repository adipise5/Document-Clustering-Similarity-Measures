import pickle

word2Synset = pickle.load(open("WordSynsetDict.pk"))

def isHinWord(wordE):
	try:
		word = wordE.decode('utf-8', 'ignore')
	except UnicodeEncodeError:
		word=wordE
	if word2Synset.has_key(word):
	    return True
	else:
	    return False

def getSynSet(word):
	if isHinWord(word):
		return word2Synset[word]
	else:
		return []

