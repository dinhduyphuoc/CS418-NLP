# Student ID: 19125115
# Student name: Dinh Duy Phuoc

import string
import math
import sys
import csv
import os
import re
from nltk import download
from nltk.corpus import stopwords, wordnet
from nltk.corpus.reader.wordnet import NOUN

# Download nltk data
download("wordnet")
download("omw-1.4")
download("stopwords")

# Fix some edge cases such as ["US", "u"] -> ["US", "us"], ["DS", "d"] -> ["DS", "ds"]
def lemmatize(word, pos=NOUN):
    lemmas = wordnet._morphy(word, pos)
    return max(lemmas, key=len) if lemmas else word


def preprocess(text):
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Remove punctuations
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove white spaces
    text = text.strip()
    # Tokenization
    text = text.split()
    # Remove stop words
    text = [word for word in text if word not in stopwords.words("english")]
    # Lemmatization
    text = [lemmatize(word) for word in text]

    return text


# Iterate through all documents in corpus and calculate IDF of each word
# IDF = log(total number of docs / number of docs with term t in it)
def calculate_idf(corpus):
    idf = {}
    for doc in corpus:
        for word in set(doc):
            if word not in idf:
                idf[word] = 0
            idf[word] += 1
    for word in idf:
        idf[word] = math.log(len(corpus) / idf[word])

    return idf


# TF = numbers of term in doc / total number of words in doc
def calculate_tf(word, word_in_doc):
    return word_in_doc.count(word) / len(word_in_doc)


# Iterate through all documents in corpus and calculate TF-IDF of each word
# TF-IDF = TF * IDF
def process(corpus, top_keywords=5):
    idf = calculate_idf(corpus)
    keywords = []
    for doc in corpus:
        tf_idf = {}
        for word in doc:
            if word not in tf_idf:
                tf_idf[word] = 0
            tf_idf[word] = calculate_tf(word, doc) * idf[word]

        keywords.append(sorted(tf_idf, key=tf_idf.get, reverse=True)[:top_keywords])

    return keywords


if __name__ == "__main__":

    # Read the directory path and output name from command line arguments
    directory = sys.argv[1]
    output_name = sys.argv[2]

    print("Working on directory: " + directory)

    # Get the corpus from the directory
    corpus = []
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r") as file:
            corpus.append(preprocess(file.read()))

    # get the keywords from the corpus
    keywords = process(corpus)

    # Write the results to csv file
    with open(output_name, "w") as file:
        print("Writing to file: " + output_name)
        writer = csv.writer(file)
        writer.writerows(sorted(zip(os.listdir(directory), keywords)))

    print("Done!")
