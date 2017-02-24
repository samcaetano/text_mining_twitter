#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    AUTOR: Samuel Caetano
    DESCRICAO: 
"""

# External imports
import nltk as NL 
import math
import time
import pickle
import re
import sys
import pandas as pd
import numpy as np
import timeit

# Internal imports
sys.path.insert(0, '/home/r2d2/projeto_ext_vero/Engine/lib')

import DatabaseMethods as dbm

# Bloco para tf_idf : algoritmo base no site ->
#...http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/
#
def Tf(term, tweet): 
    # Term frequency : o numero de vezes que o term apareceu no
    #...documento (tweet), dividido pelo numero total de termos no documento
    #return 1+math.log10(float(tweet.count(term))/float(len(tweet)))
    return float(tweet.count(term)) # puro
    #return float(tweet.count(term))/float(len(tweet)) # normalizado

#2. n_contains : retorna o numero de documentos que contem o termo
def NContains(word, tweets_tokenized):
    return sum(1 for tweet in tweets_tokenized if word in tweet)

#3. idf : 'inverse document frequency' : mede quao comum um termo eh no conjunto
#com todos os documentos (tweets)
def Idf(word, tweets_collection):
    # (numero total de documentos)/(numero total de documentos que contem o termo)
    return math.log10(float(len(tweets_collection))/float(NContains(word, t2)))

def TfIdf(word, tweet, tweets_tokenized, tweets_collection): # tf
    return Tf(word, tweet)*Idf(word, tweets_collection)
# end

def ClearTweet(arg): # realiza uma limpeza nos tweets, retirando os lixos
    arg = re.sub('ç', 'c', arg, flags = re.M)
    arg = re.sub('á|à|â|ã','a', arg, flags = re.M)
    arg = re.sub('é|è|ê','e', arg, flags = re.M)
    arg = re.sub('ì|í|î|ĩ','i', arg, flags = re.M)
    arg = re.sub('ò|ó|ô|õ','o', arg, flags = re.M)
    arg = re.sub('-', '', arg, flags = re.M)
    arg = re.sub('(<[^>]+>)|(@[\w]+)|(\#+[\w_]+[\w\'_\-]*[\w_]+)|(\d+)|(http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+)|([:=;][oO\-]?[D\)\]\(\]/\\OpP])|(h[t]+)', '', arg, flags = re.M)
    #('<[^>]+>) remove HTMLs;
    #('@[\w]+) remove mentions;
    #('\#+[\w_]+[\w\'_\-]*[\w_]+) remove hashtags;
    #('\d+','', arg, flags = re.M) remove numeros;
    #(http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+
    #...remove URLs;
    #([:=;][oO\-]?[D\)\]\(\]/\\OpP]) ignora smiles
    return arg

# Tokeniza por palavras
def tokenize(arg): 
    return tokens_re.findall(arg)

# Stemizacao por palavra (Stemmizar a partir da tokenizacao(?))
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
    

regex_str = [r'^[<[^>]+>]', # ignora HTML tags
            r'^[@[^.]+]', # ignora mentions
            r"^(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # ignora hash-tags
            r"""^http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|
            (?:%[0-9a-f][0-9a-f]))+""", # ignora URLs
            r'^(?:(?:\d+,?)+(?:\.?\d+)?)', # ignora numbers
            r"(?:[a-z][a-z'\-_]+[a-z])" ] # agrupa palavras

tokens_re = re.compile(r'('+'|'.join(regex_str)+')',
                       re.VERBOSE | re.IGNORECASE) # compila expressao regular
# Limpa a lista de stop words
stop_words = [ClearTweet(str(word.encode(encoding='UTF-8', errors='strict')))
              for word in NL.corpus.stopwords.words('portuguese')]

time_begin = time.asctime(time.localtime(time.time())) # begin

#

# Colecao com todas as seguradoras e seus seguidores
# Eh um dictionary onde as chaves sao as seguradoras 
# e os valores sao os seguidores dessa seguradora
U = dbm.GetUniverse()

for seguradoraId in U.keys():
    print "seguradoraId: %s" % dbm.GetAccountLabel(seguradoraId)
    
    # followersFeed eh um dict onde as chaves sao os seguidores
    #...e os valores sao os ids dos tweets desses seguidores
    followersFeed = dbm.GetFollowerTweets(U, seguradoraId)
    
    # Remove seguidores nao tem tweets inseridos
    followersFeed = dict((k, v) for k, v in followersFeed.iteritems() if v)
    
    errfd = open("ERR.txt", "w")

    preprocessedTweetsBySeg = dict()
    R = dict() # R : conjunto de raizes e suas derivacoes
    
    for i, follower in enumerate(followersFeed.keys()):
        
        #t1 = [] # t1 : tweet tokenizados
        t1 = dict()
        t2 = dict()
        t3 = dict() # t3 : tweet preprocessado (sem stopword e tokenizado)
        
        # 1. tokeniza tweets
        print "\tSeguidor %s" % dbm.GetAccountLabel(follower)
        print "\t\tTokenizando..."
        try:
            #  conjunto de tweets tokenizados
            for tweet in followersFeed[follower].keys():
                #print tweet, followersFeed[follower][tweet]
                t1[tweet] = tokenize(ClearTweet(\
                    str(followersFeed[follower][tweet].encode(\
                    encoding='UTF-8', errors='strict'))))
        except(KeyError):
            errfd.write("[Erro na tokenizacao]\n");
            break
        
        # 2. Retira stopwords
        # t2 : conjunto de tweets tokenizados sem stopwords
        try:
            print "\t\tRemovendo stopwords ..."
            for tweetId, tweet in t1.iteritems():
                t2[tweetId] = list(set(tweet) - set(stop_words))
                ##t2[tweetId] = [word for word in set(tweet)-set(stop_words)]
        except(KeyError):
            errfd.write("[Erro na remocao de stopwords]\n")
            break
        
        # 2.1. Preenche conjunto com os tweets
        # Rp (Raizes possiveis) : conjunto com todas...
        #...possiveis palavras que foram stemizadas
        Rp = [word for k, tweet in t2.iteritems() for word in tweet]
        
        # 2.2 Stemiza tweets
        try:
            print "\t\tStemizando..."
            for tweetId, tweet in t2.iteritems():
                t3[tweetId] = [stem(word) for word in tweet]
        except(KeyError):
            errfd.write("[Erro na stemizacao]\n")
            break
        
        
        # 3. Retira do conjunto de todas palavras possivelmente...
        #...stemizadas aquelas que nao foram stemizadas
        for k, tweet in t3.iteritems():
            Rp = list(set(Rp) - set(tweet))
        Rp = list(set(Rp))
        
        # 3.1 Converte os termos em seus respectivos stems
        s = stem
        Rp = map(lambda x: s(x), Rp)
        
        # 3.2 Inicializa o dict colocando os stems nas chaves
        for _ in Rp:
            R.update({_: [],})
            
        # 4. Preenche conjunto de pares {r: [w1, w2, ..., wK]} em R,
        #...onde r ε {stem(w) | w pertence a R} e wK ε Rp
        basketOfTerms = list(set([_ for tweet in t2.values() for _ in tweet]))
               
        
        try:
            maxiter = len(basketOfTerms)
            print "\t\tPreenchendo hash de raiz x possiveis palavras..."
            print '\t\tNumero de iteracoes: %d' % maxiter
            #s = stem
            """d = time.clock()
            for word in Rp:
                R[s(word)] = map(lambda x: x, filter(lambda x: s(x) == s(word), basketOfTerms))
            e = time.clock()
            print 'old approuch took ', e-d"""
            
            for word in basketOfTerms:
                wordstemmed = stem(word)
                if wordstemmed in R.keys():
                    R[stem(word)] += [word,]
                    
            print "\t%d de %d"\
                % (i+1, len(followersFeed.keys()))
        except(KeyError):
            errfd.write("[Erro ao preencher hash de raizes]")
            break
        
        #BEGIN - INSERCAO NO BANCO
        dbm.InsertPreprocessedTweets(t3)
        dbm.InsertRoot(followersFeed[follower], R, t2, seguradoraId)
        #END
        
        for k, v in t3.iteritems():
            if v:
                preprocessedTweetsBySeg[str(k)] = v
                    
    # 5. Gerar arquivo no formato documentoIdXpalavra,
    #...onde palavra ε t2 e documentoID ε coleção
    #print "Preprocessed tweets by seg: ", preprocessedTweetsBySeg

    data = dict()
    
    for tweet in preprocessedTweetsBySeg.values():
        if tweet:
            for word in R.keys():
                if word in data.keys():
                    data[word] += [Tf(word, tweet),]
                else:
                    data.update({word: [Tf(word, tweet),]})
                    
    data_frame = pd.DataFrame.from_dict(data,\
        orient = 'index')
    
    import os
    
    label = dbm.GetAccountLabel(seguradoraId)
    
    if not os.path.isdir('../analytics/%s' % label):
        os.mkdir('../analytics/%s' % label)
        
    label = dbm.GetAccountLabel(seguradoraId)
    data_frame.to_csv('../analytics/%s/%s.csv' % (label, label),\
        index_label = ['',]+preprocessedTweetsBySeg.keys())

    """
    try:
        print "\t\tGerando arquivo de saida..."
        temp = []
        scores = ""
        for term in R.keys():
            temp.append(term)
            scores += str(term)+";"
            #print term
        scores += "\n"
        for tweetId in preprocessedTweetsBySeg.keys():
            #print tweetId, preprocessedTweetsBySeg[tweetId]
            if len(preprocessedTweetsBySeg[tweetId]) != 0:
                scores += str(tweetId)+";"
                for word in temp:
                    if (word in preprocessedTweetsBySeg[tweetId]):
                        scores += str(Tf(word, preprocessedTweetsBySeg[tweetId]))+";"
                    else:
                        scores += "0;"
                scores += "\n"
        #
        arq = open('../scores/'+str(seguradoraId)+'.txt', 'w')
        arq.write(scores)
        arq.close()
      
    except(KeyError):
        errfd.write("[Erro ao gerar arquivo de saida]")
        break
    """
    
    print "\tSeguradora "+str(seguradoraId)+" cadastrada!\n"
    errfd.close()

# insere tweets preprocessados no DB
#ins ert_preprocessed_tweets(tweets_preprocessed, ids_collection)

print "\nPREPROCESSAMENTO FINALIZADO"
print "Coleções preprocessadas : "+str(len(U))
print "Iniciado as: "+time_begin
print "Finalizado as: "+time.asctime(time.localtime(time.time()))+"\n"
#Done

