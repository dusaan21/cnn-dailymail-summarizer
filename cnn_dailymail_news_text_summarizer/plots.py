import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import numpy as np
from .dataset import remove_punctuation
from collections import Counter, defaultdict 

def plot_num_characters(data, column):
    plt.hist(data[column].str.len(), bins=50, edgecolor='white')
    plt.xlabel(f'Number of Characters in {column}')
    plt.ylabel(f'Number of {column}s')
    plt.title(f'Distribution of Characters per {column}')
    plt.show()

def plot_num_words(data, column):
    plt.hist(data[column].str.split().map(lambda x: len(x)), bins=50, edgecolor='white')
    plt.xlabel(f'Number of Words in {column}')
    plt.ylabel(f'Number of {column}s')
    plt.title(f'Distribution of Words per {column}')
    plt.show()

def plot_num_sentences(data, column):
    plt.hist(data[column].apply(lambda x: len(nltk.sent_tokenize(x))), bins=50, edgecolor='white')
    plt.xlabel(f'Number of Sentences in {column}')
    plt.ylabel(f'Number of {column}s')
    plt.title(f'Distribution of Sentences per {column}')
    plt.show()

def plot_mean_word_length(data, column='article'):
    data['mean_word_length'] = data[column].map(lambda x : np.mean([len(word) for word in x.split()]))
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, y='mean_word_length')
    plt.ylabel("Mean Word Length")
    plt.title(f'Boxplot of Mean Word Length per {column}')
    plt.show()


def create_corpus(data, column='article'):
    corpus = []
    words = data[column].str.lower().apply(remove_punctuation).str.split()
    words = words.values.tolist()
    corpus = [word for i in words for word in i]
    return corpus

def plot_most_frequent_stopwords(data, stop_words, top_n=40, column='article'):
    corpus = create_corpus(data, column)
    dic = defaultdict(int)
    for word in corpus:
        if word in stop_words:
            dic[word] += 1
    top = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    x, y = zip(*top)
    plt.figure(figsize=(14, 7))
    plt.bar(x[:top_n], y[:top_n], color='blue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top {top_n} Most Frequent Stopwords')
    plt.xticks(rotation=45)
    plt.show()

def plot_most_frequent_words(data, stop_words, top_n=40, column='article'):
    counter=Counter(create_corpus(data, column))
    most=counter.most_common()

    x, y= [], []
    for word,count in most:
        if (word not in stop_words):
            x.append(word)
            y.append(count)
    plt.figure(figsize=(14, 7))
    plt.bar(x[:top_n], y[:top_n], color='blue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(f'Top {top_n} Most Frequent Non-Stopwords')
    plt.xticks(rotation=45)
    plt.show()

def get_top_ngram(data, stop_words, n=2, top_n=20):
    data['article'] = data['article'].str.lower().apply(remove_punctuation)
    cv = CountVectorizer(ngram_range=(n, n), stop_words=list(stop_words))
    ngrams = cv.fit_transform(data['article'])
    count_values = ngrams.toarray().sum(axis=0)
    ngram_freq = pd.DataFrame(sorted([(count_values[i], k) for k, i in cv.vocabulary_.items()], reverse = True))
    ngram_freq.columns = ['frequency', 'ngram']
    sns.barplot(x=ngram_freq['frequency'][:top_n], y=ngram_freq['ngram'][:top_n])
    if n == 2:
        plt.title(f'Top {top_n} Most Frequent Bigrams')
    elif n == 3:
        plt.title(f'Top {top_n} Most Frequent Trigrams')
    else:
        plt.title(f'Top {top_n} Most Frequent Ngrams')
    plt.show()
