# Ryan Laurents
# 10/10/2020
# University of Texas at Arlington
# CSE 4334 - Data Mining
# Programming Assignment 1

import os
import collections
import math
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

print("Working...")
# Preprocessing
corpusroot = './presidential_debates'
allWords = []
wordsPerDoc = {}
uniquesPerDoc = {}
tfPerDoc = {}
df = {}
tfidfByDoc = {}
docList = []

stemmer = PorterStemmer()
stopword = stopwords.words('english')
docCount = 0

for filename in os.listdir(corpusroot):
    # Read the files and convert text to lowercase
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close()
    doc = doc.lower()

    docList.append(filename)
    words = []
    uniques = []
    tf = {}

    # Tokenize using a regex
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(doc)
    for w in tokens:
        allWords.append(w)

    # Check words against a list of stopwords
    for w in tokens:
        if w not in stopword:
            # Stem the words as we append to a list of good tokens
            words.append(stemmer.stem(w))

            if stemmer.stem(w) not in uniques:
                uniques.append(stemmer.stem(w))

    # Store processed words/uniques into dict sorted by filename
    wordsPerDoc[filename] = words
    uniquesPerDoc[filename] = uniques

    # Check uniques for current file for document frequency
    for term in uniques:
        if term in df:
            df[term] += 1
        else:
            df[term] = 1

    for word in words:
        if word in tf:
            tf[word] += 1
        else:
            tf[word] = 1

    for key in tf:
        tf[key] = (1 + (math.log(tf[key], 10)))

    tfPerDoc[filename] = tf
    tfidfByDoc[filename] = tf

    docCount += 1

# Calculate inverse document frequency from df dict
idf = {}
for key in df:
    idf[key] = math.log((docCount/df[key]),10)



#################################################################
##################### F U N C T I O N S #########################
#################################################################

def getidf(token):
    if token not in idf:
        return -1
    else:
        return idf[token]

def gettf(filename, token):
    if token not in tfPerDoc[filename]:
        return -1
    else:
        return tfPerDoc[filename][token]


def getweight(filename, token):
    idf = getidf(token)
    if filename not in os.listdir(corpusroot):
        return 0
    elif idf == -1 or idf == 0:
        return 0

    sumSqs = 0
    for term in tfPerDoc[filename]:
        temp = (gettf(filename, term) * getidf(term))
        sumSqs += temp ** 2

    magnitude = math.sqrt(sumSqs)

    tf = gettf(filename, token)

    return((tf * idf) / magnitude)


def query(qstring):
    tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
    tokens = tokenizer.tokenize(qstring)

    stemmed = []
    for word in tokens:
        if word not in stopword:
            stemmed.append(stemmer.stem(word))

    postings = {}
    for term in stemmed:
        temp = {}
        for doc in tfPerDoc:
            if term in tfPerDoc[doc]:
                #print("Document: %s  Term: %s  TF: %f" % (doc, term, tfPerDoc[doc][term]))
                temp[doc] = tfPerDoc[doc][term]
        sort = sorted(temp.items(), key = lambda x: x[1], reverse = True)
        count = 0
        reducedToTen = {}
        for key, value in sort:
            if count == 10:
                break
            reducedToTen[key] = value
            count += 1

        print("Top 10 posting list [Not normalized] for %s: " % (term))
        print(reducedToTen)
        postings[term] = reducedToTen

    docList = []
    for term in postings:
        for doc in postings[term]:
            docList.append(doc)

    docCheck = collections.Counter(docList)
    for document, count in docCheck.most_common(1):
        if count == len(stemmed):





    return ("Query function not fully operational yet. :(", 0)



########## PASTE TEST COMMANDS BELOW ###########
#print("%.12f" % getidf("health"))
#print("%.12f" % getidf("agenda"))
#print("%.12f" % getidf("vector"))
#print("%.12f" % getidf("reason"))
#print("%.12f" % getidf("hispan"))
#print("%.12f" % getidf("hispanic"))
#print("%.12f" % getweight("2012-10-03.txt","health"))
#print("%.12f" % getweight("1960-10-21.txt","reason"))
#print("%.12f" % getweight("1976-10-22.txt","agenda"))
#print("%.12f" % getweight("2012-10-16.txt","hispan"))
#print("%.12f" % getweight("2012-10-16.txt","hispanic"))
print("(%s, %.12f)" % query("particular constitutional amendment"))
