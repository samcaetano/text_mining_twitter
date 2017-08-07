#!/usr/bin/python
# -*- coding: utf-8 -*-

# This script was developed by Samuel Caetano
# This script works on generationg posprocessing analysis.
# Important links: [https://joernhees.de/blog/2015/08/26/, 
# scipy-hierarchical-clustering-and-dendrogram-tutorial/] and 
# [http://brandonrose.org/clustering]

# External imports
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.cm as cm
from spherecluster import SphericalKMeans as SKMeans
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.metrics import silhouette_samples, silhouette_score
from collections import Counter
import json
import logging
import time
import matplotlib
matplotlib.rcParams.update({'font.size': 24})

# Internal imports
sys.path.insert(0, '../lib')
import DatabaseMethods as dbm


def RetriveContent(documents_id):
    docs_content = []
    for document_id in documents_id:
        try:
            tweet_preprocessed = dbm.GetTweetById(document_id[0])
            docs_content += [tweet_preprocessed.split(' '),]
        except(AttributeError):
            next
            
    return docs_content

def FancyDendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Dendrograma Clusterização Hierarquica (truncado)')
        plt.xlabel('sample index or (cluster size)', fontsize=16)
        plt.ylabel('distance', fontsize=16)
        for i, d, c in zip(ddata['icoord'], \
                            ddata['dcoord'], \
                            ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5), \
                             textcoords='offset points', \
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

def LoadFromDataFrame(seg):
    label = dbm.GetAccountLabel(seg)
    
    df = pd.read_csv('../analytics/%s/%s_sliced.csv' % (label, label))
    df = df.dropna(axis = 'columns', how = 'all')
    #df = df.drop('Unnamed: 0', axis = 1)
    
    M = pd.DataFrame.as_matrix(df)

    terms = [term for term in M.T[0]]
    #d0 =  [df['Unnamed: 0.1'][r] for r in df.index]
    
    #d1 = [[c] for c in df.columns[1:]]
    doc_ids = [[c] for c in df.columns[1:]]
   
    scores = [row for row in M.T[2:]]
    
    del M
    del df
    # The content returned is a list of three lists, the first list is a list
    # of terms, the second is a document id list and the third is a score list
    # (where which score list corresponds to the scores of terms in each 
    # document)
    return [terms, doc_ids, scores]
    #return [d0, d1, d2]

def Draw2DClusters(arg, seguradora): 
    # >arg< is the dissimilarity matrix
    mds = MDS(n_components = 2,\
              dissimilarity = 'precomputed',\
              random_state = 1)
    pos = mds.fit_transform(arg)

    
    plt.scatter(pos[:, 0],\
                pos[:, 1],\
                c = clusters_colors)
    
    plt.title('Documentos e seus clusters')

    label = dbm.GetAccountLabel(seguradora)
    plt.savefig('../analytics/%s/%s_partitional_cluster.png' % (label, label))
    plt.close()
    
def DrawDendrogram(arg, labels, seguradora):
    # Calculate full dendrogram
    label = dbm.GetAccountLabel(seguradora)
    
    plt.title('Dendrograma de %s', label)
    plt.ylabel('Documentos', fontsize=16)
    plt.xlabel('Dissimilaridade', fontsize=16)
    dendrogram(
        arg,
        leaf_rotation = 0.,  # rotates the x axis labels
        leaf_font_size = 8.,  # font size for the x axis labels
        labels = [' '.join(e) for e in labels],
        orientation = 'left'
    )
    
    plt.savefig('../analytics/%s/%s_hierarchical_cluster.png' % (label, label))    
    plt.close()

def Silhouette(X, seguradora):
    insurance_label = dbm.GetAccountLabel(seguradora)
    maxx = len(X)
    
    if maxx > 11:
        maxx = 11
        
    range_of_clusters = list(range(2, maxx))
    clusters_silhouette = dict()

    for n_clusters in range_of_clusters:
        # Initialize the clusterer with n_clusters value 
        #...and a random generator
        # seed of 10 for reproducibility.
        clusterer = SKMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation 
        #...of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        
        clusters_silhouette.update({n_clusters:silhouette_avg})

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

    plt.title('Silhueta media de %s' % insurance_label)
    plt.xlabel('Numero de clusters', fontsize=16)
    plt.ylabel("Silhueta media", fontsize=16)
    plt.plot(clusters_silhouette.keys(), clusters_silhouette.values())
    plt.savefig("../analytics/%s/%s_silhuette.png" \
        % (insurance_label, insurance_label))
    plt.close()
    
    
    silhouettes = [v for v in clusters_silhouette.values()]
    
    
    for k, v in clusters_silhouette.iteritems():
        if max(silhouettes) == v:
            return k
        
def clusterSorter(arg, data):
    docs_organized_per_cluster = dict()
    
    for index, lst in enumerate(arg):
        if lst[0] in docs_organized_per_cluster.keys():  
            docs_organized_per_cluster[lst[0]] +=  [data[1][index][0],]
        else:
            docs_organized_per_cluster.update({lst[0]:[data[1][index][0],]})
    
    return docs_organized_per_cluster

def GetCommomTermsInCluster(clusters, docs):
    A = dict()
    
    for cluster in clusters:
        terms_per_cluster = Counter()
        
        for doc_id in clusters[cluster]:
            terms_per_cluster.update(docs[doc_id])
        
        A.update({cluster:terms_per_cluster})
    
    return A

def RetrieveFollowers(argLst):
    return list(set([dbm.GetFollowerByTweetId(_) for _ in argLst]))

# This gives the information about what is in the cluster.
# Which words are inside the cluster, what it can mean and who uses it
def intraCluster(cluster, seg, collection, cluster_info,\
                 cluster_index, documents_id):

    tweets = []
    
    # Realizes the sorting of terms by quantity of occurences inside a cluster
    items_sorted =  sorted([_ for _ in cluster.iteritems()],\
                                key = lambda freq: freq[1],\
                                reverse = True)
    
    tweetsInCluster = []
    
    # Run through each cluster term which is sorted decrescently
    for item in items_sorted:
        try:            
            # List with all tweets which contains the term
            tweets = [_ for _ in dbm.GetTweetIdByTerm(item[0], seg)
                      if str(_) in documents_id]
            
            # List with the derivations of stemmed term
            variations = dbm.GetDerivatives(item[0])
            
            # The w term appears x times in this cluster
            print '''\t\t\t[%s] aparece %d vez(es) nesse cluster.
                \t\tEsse termo pode ser: %s''' % (item[0], item[1], variations)
            
            if seg in collection.keys():
                collection[seg] += [item[0],]
            else:
                collection[seg] = [item[0],]
            
        except(Exception):
            next
            
        tweetsInCluster += tweets

    followers = dbm.GetFollowerByAccount(list(set(tweetsInCluster)), seg)
    followers = set(followers) - set([seg])
    
    
    print '''\t\tExistem %d seguidores de %s que usam os termos nesse cluster.'''\
        % (len(followers), dbm.GetAccountLabel(seg))
        
    print '\t\tEsses seguidores sao os seguintes ', list(followers)
    
    get_der = dbm.GetDerivatives(items_sorted[0][0])
    
    possibilities = items_sorted[0][0] if get_der == None else get_der
        
    valid = True
    for v in cluster_info.values():
        if possibilities in v:
            valid = False

    if valid:
        cluster_info[cluster_index] += [len(followers), possibilities]
    else:
        cluster_info.pop(cluster_index)

def extraCluster(clusters, terms_clusterized, seg, collection, documents_id):
    # Creates structure in format:
    # {cluster_index: [frequency of terms in database, people in this cluster]}
    cluster_info = dict()   

    documents_id = [_[0] for _ in documents_id]
    
    label = dbm.GetAccountLabel(seg)
    
    cluster_weightered = []
    for k, v in clusters.iteritems():
        cluster_weightered += [(k, len(v)),]
    
    # Lists the 25 biggest clusters (with more tweets inside)
    for _ in sorted(cluster_weightered, key = lambda weight: weight[1],\
        reverse = True)[:25]:
        
        print '\tO cluster %d tem %d tweet(s)' % (_[0], _[1])
        print '''\t\tEsses tweet(s) tem %d termo(s), que sao os seguintes:'''\
            % (len(terms_clusterized[_[0]].keys()))

        cluster_info.update({_[0]:[_[1],]})
        
        intraCluster(terms_clusterized[_[0]], seg, collection, 
                    cluster_info, _[0], documents_id)
        
        print 'cluster info ', cluster_info
                
    PrintBubbleChart(seg, cluster_info)
    ###
    try:
        import math
        PrintTagCloud(seg,
                    [v[2] for v in cluster_info.values()],
                    [math.log10(v[0]*v[1]) for v in cluster_info.values()])
    except(TypeError):
        print 'algum erro em ', cluster_info.values()
    
def PrintBubbleChart(seg, cluster_info):
    from pylab import *
    from scipy import *
    import math
    
    x = []
    y = []
    color = []
    area = []
    
    for key in cluster_info.keys():
        # Number of people inside the cluster
        x.append(cluster_info[key][1])
        
        # Number of tweets in cluster occurs in database
        y.append(cluster_info[key][0])
        
        # Proporcional area to the number of people inside the cluster
        k = cluster_info[key][1]
        area.append(math.pi * (k)**2)
        
        # Color corresponds to the number of occurences of terms in database
        color.append(cluster_info[key][1])
        
        text(cluster_info[key][1], cluster_info[key][0], cluster_info[key][2],
             size = 7, horizontalalignment = 'center')
        
    sct = scatter(x, y, c = color, s = area, linewidths = 2, edgecolor = 'w')
    sct.set_alpha(1)
    
    label = dbm.GetAccountLabel(seg)
    
    xmax = max([x[1] for x in cluster_info.values()])
    ymax = max([y[0] for y in cluster_info.values()])
    
    axis([0, xmax+(1/float(xmax)),
          0, ymax+(1/float(ymax))])
    
    title('%s' % label)
    xlabel('Numero de follower dentro do cluster', fontsize=16)
    
    # This is the sum of occurrences in the database from all 
    # terms inside the cluster
    ylabel('''Numero de tweets dentro do cluster''', fontsize=16)
    
    savefig('../analytics/%s/%s_bubblechart.png' % (label, label))
    close()
    
def PrintTagCloud(seg, terms, td_scores):
    from os import path
    from wordcloud import WordCloud

    d = path.dirname(__file__)
    
    # Tuple like follows (term, frequency)
    frequencies = []
    # Gera a tag cloud da lista de tuplas das palavras e suas frequencias
    for i in range(len(terms)):
        frequencies += [(terms[i], td_scores[i]),]
    wordcloud = WordCloud().generate_from_frequencies(frequencies)

    import matplotlib.pyplot as plt
    plt.imshow(wordcloud)
    plt.axis('off')

    # lower max_font_size
    label = dbm.GetAccountLabel(seg)
    wordcloud = WordCloud(max_font_size=40).generate_from_frequencies(frequencies)
    plt.title(label)
    
    plt.savefig('../analytics/%s/%s_tagcloud.png' % (label, label))
    plt.close()
    
def PrintVennDiagram(A, B, C):
    # Visualization of the Venn diagram of the 3 biggest insurance-companies    
    from matplotlib_venn import venn3
    
    # Given the insurance-company id. search it followers in database
    setA = dbm.GetFollowerBySeg(A)
    setB = dbm.GetFollowerBySeg(B)
    setC = dbm.GetFollowerBySeg(C)
    
    labelA = dbm.GetAccountLabel(A)
    labelB = dbm.GetAccountLabel(B)
    labelC = dbm.GetAccountLabel(C)
    
    set1 = set(setA)
    set2 = set(setB)
    set3 = set(setC)

    venn3([set1, set2, set3], (labelA, labelB, labelC))
    plt.title('Diagrama de Venn das 3 maiores seguradoras em numero de follower')
    plt.savefig('../analytics/venn_diagram.png')
    plt.close()

# Creates log file
logging.basicConfig(filename = 'posprocessing_outputs.log',\
                    level = logging.DEBUG)

seguradoras = dbm.GetAllSeguradoras()
collection = dict()

followers_count = [(i, len(dbm.GetFollowerBySeg(_)))\
                    for i, _ in enumerate(seguradoras)]
foo = sorted(followers_count, key = lambda x:x[1], reverse = True)

try:
    PrintVennDiagram(seguradoras[foo[0][0]], seguradoras[foo[1][0]],\
                    seguradoras[foo[2][0]])
except(IndexError):
    print 'Contas insuficientes para gerar diagrama de Venn'


for seguradora in seguradoras:
    try:
        label = dbm.GetAccountLabel(seguradora)
        
        #if os.path.isfile('../analytics/%s/%s_bubblechart.png' % (label, label)):
        #    print '%s ja processada' % label
        #    continue
        
        print 'processando %s' % label
        
        
        data = LoadFromDataFrame(seguradora)
        
        cr = RetriveContent(data[1])
        
        # >documents< is a dictionary where key is tweet id and value is
        # tweet content
        documents = dict()
        for _, k in enumerate(data[1]):
            try:
                documents.update({k[0]:cr[_]})
            except(IndexError):
                next
            
        docs_content = np.array([_ for _ in cr])
        

        # Original data matrix
        M = np.array([lst for lst in data[2]])
        #M = M.astype(np.float)
        #
        # Reports to logfile
        m = json.dumps({'message': 'Working',\
            'place_at': 'Original data matrix created'}),\
            time.asctime(time.localtime(time.time()))
        logging.info(m)

        # Similarity and dissimilarity matrix
        similarity = cosine_similarity(M)
        dissimilarity = 1 - similarity
        
        # Reports to logfile
        m = json.dumps({'message': 'Working', \
            'place_at': 'Similarity and dissimilarity matrix created'}), \
            time.asctime(time.localtime(time.time()))
        logging.info(m)

    
        # Calculates silhouette
        k = Silhouette(M, seguradora)
        # Reports to logfile
        m = json.dumps({'message': 'Working', \
            'place_at': 'Silhouette calculated'}), \
                time.asctime(time.localtime(time.time()))
        logging.info(m)
        continue
        # Spherical clustering
        skm = SKMeans(n_clusters = k, random_state = 0) 
        skm.fit(M)
        clusters_colors = skm.labels_
        
        # Reports to logfile
        m = json.dumps({'message': 'Working', \
            'place_at': 'Spherical cluster calculated'}), \
                time.asctime(time.localtime(time.time()))
        logging.info(m)

        # Hierarchical clustering
        from scipy.spatial.distance import pdist
        dist = pdist(M, 'cosine')
        dist =  dist[~np.isnan(dist)]
        
        Z = linkage(dist, method = 'average')
        
        # Reports to logfile
        m = json.dumps({'message': 'Working', \
            'place_at': 'Hiearchical cluster calculated'}), \
                time.asctime(time.localtime(time.time()))
        logging.info(m)

        # Dendrogram plot
        # Calculate cut in dendrogram
        cutted = cut_tree(Z, height = 0.75)
        
        # Creates the clusters, which is a dictionary where key is the cluster 
        # index and value is a list of documents that belong to the cluster
        clusters = clusterSorter(cutted, data)
        
        print 'Processed account: %s' % dbm.GetAccountLabel(seguradora)
        print '\tO k ideal, por clusterizacao particional, eh %d' % k
        print '\tO k ideal, por clusterizacao hierarquica, eh %d' \
            % len(clusters.keys())
        
        commom_terms_per_cluster = GetCommomTermsInCluster(clusters, documents)        
        
        # Draws spherical cluster
        Draw2DClusters(dissimilarity, seguradora)

        # Draws dendrogram
        DrawDendrogram(Z, docs_content, seguradora)
        
        extraCluster(clusters,
                     commom_terms_per_cluster,
                     seguradora,
                     collection,
                     data[1])
        
        
        collection[seguradora] = commom_terms_per_cluster
        
        # Reports to logfile
        m = json.dumps({'message': 'Working', \
            'place_at': 'Dendrograms calculated and ploted'}), \
                time.asctime(time.localtime(time.time()))
        logging.info(m)
        
        del M
        del Z
        
    except(IOError, ValueError), e:
        print e
        next

print 'Analise entre seguradoras'
print '\t%d seguidores totais' % dbm.GetFollowerCount()

for k in collection.keys():
    print "\t%d (%f) seguidores de %s"\
        % (len(dbm.GetFollowerBySeg(k)) - 1,\
            (float(\
                len(dbm.GetFollowerBySeg(k)) - 1)/float(\
                    dbm.GetFollowerCount())),dbm.GetAccountLabel(k))

import itertools
combinations = itertools.combinations(collection.keys(), 2)
print '\tCombinacoes '

for combination in combinations:
    follower_from_comb = dbm.GetFollowerFromCombination(combination[0],
                                                        combination[1])
    total_number = len(follower_from_comb)
    
    labelA = dbm.GetAccountLabel(combination[0])
    labelB = dbm.GetAccountLabel(combination[1])
    
    print '\t%d (%f) seguidores de %s e %s' \
        % (total_number,\
           float(float(total_number) / float(dbm.GetFollowerCount())),\
       labelA, labelB)
