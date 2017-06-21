#!/usr/bin/python
# -*- coding: utf-8 -*-

# This script was developed by Samuel Caetano
# Preprocessing script. 
# This script works on preprocessing the dataset available, cleaning document
# content, removing unwanted tokens (such as twitter mentions, numbers, URLs,
# HTMLs and hashtags), tokenizing, removing stop-words. In the final stage it
# goes stemming and weighting them

# External imports
import nltk as NL 
import math
import time
import pickle
import re
import sys
import pandas as pd
import numpy as np
import logging
import json
import os

# Internal imports
sys.path.insert(0, '../lib')
import DatabaseMethods as dbm

# For more term weighting go to:
# http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/
def Tf(term, tweet): 
    # Pure term frequency
    return float(tweet.count(term))

# Clears th tweet, removes 'trash'
def ClearTweet(arg): 
    arg = re.sub('ç', 'c', arg, flags = re.M)
    arg = re.sub('á|à|â|ã','a', arg, flags = re.M)
    arg = re.sub('é|è|ê','e', arg, flags = re.M)
    arg = re.sub('ì|í|î|ĩ','i', arg, flags = re.M)
    arg = re.sub('ò|ó|ô|õ','o', arg, flags = re.M)
    arg = re.sub('ú|ù|û|ũ','u', arg, flags = re.M)
    arg = re.sub('(<[^>]+>)', '', arg, flags = re.M)
    arg = re.sub('@[\w]+', '', arg, flags = re.M)
    arg = re.sub('www.[\w]+.[\w]+(.[\w][\w])', '', arg, flags = re.M)
    arg = re.sub('\#+[\w_]+[\w\'_\-]*[\w_]+', '', arg, flags = re.M)
    arg = re.sub('_', ' ', arg, flags = re.M)
    arg = re.sub('\d+', '', arg, flags = re.M)
    arg = re.sub('http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', '', arg, flags = re.M)
    arg = re.sub('[:=;][oO\-]?[D\)\]\(\]/\\OpP]', '', arg, flags = re.M)
    arg = re.sub('(-|\\.|\\$|\\*|\\+|\\?|\\!|\\{|\\}|\\[|\\]|\\(|\\)|\\:)*', '', arg, flags = re.M)
    return arg

    
    # Removes HTML: ('<[^>]+>)
    # Removes mentions: ('@[\w]+)
    # Removes hashtags: ('\#+[\w_]+[\w\'_\-]*[\w_]+)
    # Removes numbers: ('\d+','', arg, flags = re.M)
    # Removes URLs: (http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|
    # [!*\(\),]|(?:%[0-9a-f][0-9a-f]))+
    # Ignores smiles: ([:=;][oO\-]?[D\)\]\(\]/\\OpP])
    #arg = re.sub('(<[^>]+>)|(@[\w]+)|(\#+[\w_]+[\w\'_\-]*[\w_]+)|(\d+)|'+
    #'(http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+)|'+
    #'([:=;][oO\-]?[D\)\]\(\]/\\OpP])|(h[t]+)', '', arg, flags = re.M)
    #return arg

# Tokenize words
def tokenize(arg): 
    return tokens_re.findall(arg)

# Stemize words
def stem(arg): 
    lowercase = arg.lower()
    
    portuguese_stem = NL.stem.snowball.SnowballStemmer("portuguese").stem
    english_stem = NL.stem.snowball.SnowballStemmer("english").stem
    
    stem_pt = portuguese_stem(lowercase)
    
    if(stem_pt == arg):
        stem_en = english_stem(lowercase)
        return stem_en
    else:
        return stem_pt
    
# Regular expression set to ignores following patterns
regex_str = [r'^[<[^>]+>]', # ignores HTML tags
            r'^[@[^.]+]', # ignores mentions
            r"^(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # ignores hash-tags
            r"""^http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|
            (?:%[0-9a-f][0-9a-f]))+""", # ignores URLs
            r'^(?:(?:\d+,?)+(?:\.?\d+)?)', # ignores numbers
            r"(?:[a-z][a-z'\-_]+[a-z])" ] # agrupes palavras

# Compiles regex
tokens_re = re.compile(r'('+'|'.join(regex_str)+')',
                       re.VERBOSE | re.IGNORECASE)

# Applies regex to stop-words list
stop_words = [ClearTweet(str(word.encode(encoding='UTF-8', errors='strict')))
              for word in NL.corpus.stopwords.words('portuguese')]+['pra']

# Start
time_begin = time.asctime(time.localtime(time.time()))

# Collection with all insurance-companies and it's followers
# >U< is for Universe, which is a dict which keys are insurance-companies
# and the values are the followers
U = dbm.GetUniverse()

# Creates log file
logging.basicConfig(filename = 'preprocessing_results.log', \
    level = logging.DEBUG)

for insurance_id in U.keys():
    print "Getting followers feed from %s" % dbm.GetAccountLabel(insurance_id)    
    
    # >followers_feed< is a dict data type where keys are followers
    # and values are tweet ids from these followers
    follower_feed = dbm.GetFollowerTweets(U, insurance_id)
    
    # Removes followers with no tweets inserted
    # Note: There are difference from tweets in Twitter and tweets in database;
    # a user can have tweets in Twitter but none in database, because after
    # removing whatever's necessary this tweet can have no content to be 
    # inserted into the database
    follower_feed = dict((k, v) for k, v in follower_feed.iteritems() if v)
    
    m = json.dumps({'message': 'Working', \
        'place_at': 'Created followers feed'}), \
            time.asctime(time.localtime(time.time()))
    logging.info(m)

    # Dict used to create the .csv file
    # Keys are tweet ids and values are the whole 
    # stopwordless-stemmed-tokenized tweets from every single follower from the
    # current insurance-company
    collection_of_tweets = dict()
    
    # >root_derivations< is a dict which keys are root tokens (stem) and
    # values are derivations from that stem. i.e, {r, [w1, w2, ..., wK]}
    # Where r ε {stem(w) | w belongs to >basket_of_terms<}
    # and w ε >possible_roots<. i.e. {'work', ['worked', 'working']}
    R = dict()
    basket = []
    
    for i, follower in enumerate(follower_feed.keys()):
        # >t1< is a dict where keys are tweet ids and values are 
        # tokenized tweet contents
        t1 = dict()
        
        # >t2< is a dict where keys are tweet ids and values are
        # tokenized tweet contents without stop-words
        t2 = dict()
        
        # >t3< is a dict where keys are tweet ids and values are
        # stemmed-tokenized tweet contents
        t3 = dict()
        
        
        # 1. Tokenize tweets
        print "Follower %s" % dbm.GetAccountLabel(follower)
        print "\t\tTokenizing..."
        try:
            # Tokenized tweets set
            for tweet in follower_feed[follower].keys():
                t1[tweet] = tokenize(ClearTweet(\
                    str(follower_feed[follower][tweet].encode(\
                    encoding='UTF-8', errors='strict'))))
                
            m = json.dumps({'message': 'Working', \
                'place_at': 'Tokenized tweets'}), \
                    time.asctime(time.localtime(time.time()))
            logging.info(m)
        except(KeyError):
            m = json.dumps({'message': 'KeyError', \
                'place_at': 'Tokenized tweets'}), \
                    time.asctime(time.localtime(time.time()))
            logging.warning(m)
            break
        
        # 2. Removes stop-words
        try:
            print "\t\tRemoving stop-words ..."
            for tweetId, tweet in t1.iteritems():
                t2[tweetId] = list(set(tweet) - set(stop_words))
            
            
            m = json.dumps({'message': 'Working', \
                'place_at': 'Removed stop-words'}), \
                    time.asctime(time.localtime(time.time()))
            logging.info(m)
            
        except(KeyError):
            m = json.dumps({'message': 'KeyError', \
                'place_at': 'Removed stop-words'}), \
                    time.asctime(time.localtime(time.time()))
            logging.warning(m)
            break
        
        
        # Remove from tokenized those tweets which were not tokenized
        t2 = {_[0]: t2[_[0]] for _ in t2.iteritems() if _[1] != []}
        
        
        # 2.1. Fills set with tweets
        # >possible_roots< is a possible root list where values are tokens from 
        # the stop-wordless-tokenized tweets
        possible_roots = [word for k, tweet in t2.iteritems() for word in tweet]
        
        # 2.2 Stemize tweets
        try:
            print "\t\tStemming..."
            for tweetId, tweet in t2.iteritems():
                t3[tweetId] = [stem(word) for word in tweet]
                basket.append(stem(word))
            
            m = json.dumps({'message': 'Working', \
                'place_at': 'Stemmed'}), \
                    time.asctime(time.localtime(time.time()))
            logging.info(m)
        except(KeyError):
            m = json.dumps({'message': 'KeyError', \
                'place_at': 'Stemmed'}), \
                    time.asctime(time.localtime(time.time()))
            logging.warning(m)
            break
        
        
        # 3. Removes from the possible root list those
        # tokens which weren't stemmized
        for tweet in t3.values():
            possible_roots = list(set(possible_roots) - set(tweet))
        possible_roots = list(set(possible_roots))
        
        m = json.dumps({'message': 'Working', \
            'place_at': \
                'Removed not stemmized terms in possible roots'}), \
                time.asctime(time.localtime(time.time()))
        logging.info(m)       
        
        # 3.1 Converts tokens to their stems
        s = stem
        possible_roots = map(lambda x: s(x), possible_roots)
        
        # 3.2 Starts dict inserting stems in keys
        #for _ in possible_roots:
        #    R.update({_: [],})
        R.update({_: [] for _ in possible_roots})
        
        basket_of_terms = list(set(
            [_ for tweet in t2.values() for _ in tweet]))
        
        
        # 4. Fills pair set {r: [w1, w2, w3, ..., wK]} in R
        # where r ε {stem(w) | w belongs to R} and wK ε possible_roots
        try:
            num_of_terms = len(basket_of_terms)
            print '\t\tFilling root x possible derivations dictionary'
            print '\t\tNumber of terms: %d' % num_of_terms
            
            for word in basket_of_terms:
                wordstemmed = stem(word)
                if wordstemmed in R.keys():
                    R[stem(word)] += [word,]
                    
            print "\t%d of %d"\
                % (i+1, len(follower_feed.keys()))
            
            m = json.dumps({'message': 'Working', \
                'place_at': 'Created R pair set'}), \
                    time.asctime(time.localtime(time.time()))
            logging.info(m)
            
        except(KeyError):
            m = json.dumps({'message': 'KeyError', \
                'place_at': 'Created R pair set'}), \
                    time.asctime(time.localtime(time.time()))
            logging.warning(m)
            break
        
        features_len = len(R.keys())
            
        for k, v in t3.iteritems():
            if v:
                collection_of_tweets[str(k)] = v
                    
    # 5. Genarates the output file in document_idXtoken format
    # where document_id ε collection and token ε t2  
    m = ('Generating dataset file', \
                time.asctime(time.localtime(time.time())))
    logging.info(m)
    
    # Counter counts the frequency each term appeared in all tweets
    from collections import Counter
    basket = Counter(basket)
    
    # Sorts the terms by its frequencies
    basket = sorted(basket.iteritems(), key = lambda t: t[1], reverse=True)

    # Select only the 2000 more frequent
    basket = {_[0]:_[1] for _ in basket[:2000]}
    
    data = {token:[] for token in basket.keys()}
    
    for term in data.keys():
        for tweet in collection_of_tweets.values():
            data[term].append(Tf(term, tweet))

    m = ('Generating dataframe', \
                time.asctime(time.localtime(time.time())))
    logging.info(m)                 
    
    data_frame = pd.DataFrame.from_dict(data,\
        orient = 'index')
        
    label = dbm.GetAccountLabel(insurance_id)
    
    if not os.path.isdir('../analytics/%s' % label):
        os.mkdir('../analytics/%s' % label)
        
    label = dbm.GetAccountLabel(insurance_id)
    data_frame.to_csv('../analytics/%s/%s.csv' % (label, label),\
        index_label = ['',] + collection_of_tweets.keys())
    
    print '\t%s followers succesfully preprocessed!\n' % \
        dbm.GetAccountLabel(insurance_id)

print '\nPreprocessing done!'
print 'Collections inserted: %d' % len(U)
print 'Started at ', time_begin
print "Finished at ", time.asctime(time.localtime(time.time()))
m = ('Finished', \
                time.asctime(time.localtime(time.time())))
logging.info(m)
