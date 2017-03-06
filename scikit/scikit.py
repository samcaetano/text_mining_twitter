#!/usr/bin/python
# -*- coding: utf-8 -*-

# This script was developed by Samuel Caetano
# This script works on generating posprocessing analysis.
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

# Internal imports
sys.path.insert(0, '../lib')
import DatabaseMethods as dbm


def RetriveContent(arg):
    content_splited = []
    for document_index, document_id in enumerate(arg[1]):
        content = []
        for position, score in enumerate(arg[2][document_index]):
            if(float(score) > 0.):
                content += [arg[0][position],]        
        content_splited += [content,]
    # The content returned is a list of three lists, the first list is a list
    # of terms, the second is a document id list and the third is a score list
    # (where which score list corresponds to the scores of terms in each 
    # document)
    return content_splited

def FancyDendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
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
    
    df = pd.read_csv('../analytics/%s/%s_sliced.csv' \
        % (label, label))
    df = df.dropna(axis = 'columns',\
            how = 'all')
    df = df.drop('Unnamed: 0', \
        axis = 1)

    d0 =  [df['Unnamed: 0.1'][r] \
        for r in df.index]    
    
    d1 = [[c] \
        for c in df.columns[2:]]
    
    outter_tmp = []
    tmp = []
    
    for c in df.columns[2:]:
        tmp = []
        for r in df.index:
            tmp += [str(df[c][r]),]
        outter_tmp += [tmp,]
    d2 = outter_tmp
    
    return [d0, d1, d2]

def Draw2DClusters(arg, seguradora): 
    # >arg< is the dissimilarity matrix
    mds = MDS(n_components = 2, \
        dissimilarity='precomputed', \
            random_state=1)
    pos = mds.fit_transform(arg)

    
    plt.scatter(pos[:, 0], \
        pos[:, 1], \
            c = clusters)
    plt.title('Documents and its clusters')

    label = dbm.GetAccountLabel(seguradora)
    plt.savefig('../analytics/%s/%s_partitional_cluster.png' \
        % (label, label))
    
    plt.close()
    
def DrawDendrogram(arg, labels, seguradora):
    # Calculate full dendrogram
    plt.title('Hierarchical clustering dendrogram')
    plt.ylabel('Documents')
    plt.xlabel('Dissimilarity')
    dendrogram(
        arg,
        leaf_rotation=0.,  # rotates the x axis labels
        leaf_font_size=8.,  # font size for the x axis labels
        labels = [' '.join(e) for e in labels],
        orientation = 'left'
    )
    label = dbm.GetAccountLabel(seguradora)
    plt.savefig('../analytics/%s/%s_hierarchical_cluster.png' \
        % (label, label))
    plt.close()
   
# Not needed
"""def DendrogramCut(Z, cut):
    FancyDendrogram(
        Z,
        truncate_mode='lastp',
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True,
        annotate_above=10,
        max_d=cut,  # plot a horizontal cut-off line
    )
    plt.savefig("hierarchical_cluster.png")
    plt.close()"""

def Silhouette(X, seguradora):
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

    plt.xlabel('Number of clusters')
    plt.ylabel("Silhuette average")
    plt.plot(clusters_silhouette.keys(), clusters_silhouette.values())
    
    insurance_label = dbm.GetAccountLabel(seguradora)
    plt.savefig("../analytics/%s/%s_silhuette.png" \
        % (insurance_label, insurance_label))
    plt.close()
    
    
    silhouettes = [v for v in clusters_silhouette.values()]
    
    
    for k, v in clusters_silhouette.iteritems():
        if max(silhouettes) == v:
            return k
        
def DocumentIndexSorter(arg, data):
    docs_organized_per_cluster = dict()
    
    for index, lst in enumerate(arg):
        if lst[0] in docs_organized_per_cluster.keys():  
            docs_organized_per_cluster[lst[0]] +=  [data[1][index][0],]
        else:
            docs_organized_per_cluster.update({lst[0]:[data[1][index][0],]})
    
    return docs_organized_per_cluster

def GetCommomTermsInCluster(docs_organized_per_cluster, docs):
    A = dict()
    
    for cluster in docs_organized_per_cluster:
        terms_per_cluster = Counter()
        for index, docIndex in enumerate(docs_organized_per_cluster[cluster]):
            terms_per_cluster.update(docs[docIndex])
        A.update({cluster:terms_per_cluster})
    return A

def RetrieveFollowers(argLst):
    return list(set([dbm.GetFollowerByTweetId(_) for _ in argLst]))

def PrintInCluster(cluster, seg, totalFreq, collection,\
    cluster_info, cluster_index):

    tweets = []
    
    # Realizes the sorting of terms per quantity of occurences inside a cluster
    items_sorted =  sorted([_ for _ in cluster.iteritems()], 
                    key = lambda freq: freq[1],
                    reverse = True)
    
    tweetsInCluster = []
    
    # Percorre cada termo do cluster...
    # ...que esta ordenado do maior para o menor.
    # Run through each cluster term which is sorted decrescently
    for item in items_sorted:
        try:
            # Proportion between quantity of occurences of term in general set
            # over the quantity of occurences inside the cluster
            weight = (float(item[1]) / float(totalFreq));
            
            # List with all tweets which contains the term
            tweets = dbm.GetTweetIdByTerm(item[0], seg)
            
            # List with the derivations of stemmed term
            variations = dbm.GetDerivatives(item[0])
            
            # The w term appears x times in this cluster but appears in 
            # y documents from the insurance-company which is not necessarily
            # inside this cluster
            print '''\t\t\t%s appears %d times inside this cluster.
                \tThis term can be: %s''' % (item[0], item[1], variations)
            print '\t\t\t\tThis word has a weight of %f ' % weight
            
            if seg in collection.keys():
                collection[seg] += [item[0],]
            else:
                collection[seg] = [item[0],]
            
        except(Exception):
            next
            
        tweetsInCluster += tweets

    lst = dbm.GetFollowerByAccount(list(set(tweetsInCluster)), seg)
    print '\t\tThis cluster has %d followers from %s account' \
        % (len(set(lst)-set([seg])), dbm.GetAccountLabel(seg))
    print '\t\tThe follower(s) is/are ', list(set(lst)-set([seg]))
    
    cluster_info[cluster_index] += [len(set(lst)-set([seg])), \
        dbm.GetDerivatives(items_sorted[0][0])]
    
    
    
    
def PrintBigCluster(generalCluster, seg, collection):
    adder = []
    # Creates structure in format:
    # {cluster_index: [frequency of terms in database, people in this cluster]}
    cluster_info = dict()
    
    for clusterIndex, cluster in generalCluster.iteritems():
        add = []
        for term in cluster.keys():
            add += dbm.GetTweetIdByTerm(term, seg)
        adder += [(clusterIndex, len(add)),]
    
    # Print clusters
    for _ in sorted(adder,\
        key = lambda occurence:occurence[1],\
        reverse = True):
        if _[1] > 0:
            # x :: numero de seguidores dentro do cluster 
            # y :: numero de aparicoes das palavras do cluster...
            #...que estao no banco de dados
            # x is the number of followers inside the cluster
            # y is the number of total occurences of the terms in the cluster
            
            print '\tThe cluster %d has term which appears in' \
                % _[0],
            print '%d cluster-independent tweets' \
                % _[1]
            print '\t\tThis cluster has the following terms'
            cluster_info.update({_[0]:[_[1],]})
            
            PrintInCluster(generalCluster[_[0]], seg, _[1], collection, cluster_info, _[0])        
    PrintBubbleChart(seg, cluster_info)
    
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
        
        # Number of times that the terms in cluster occurs in database
        y.append(cluster_info[key][0])
        
        # Proporcional area to the number of people inside the cluster
        k = cluster_info[key][1]
        if k >= 50:
            k = 50 + cluster_info[key][1]*0.01
        elif k <= 5:
            k = 5 + cluster_info[key][1]*0.01
        area.append(math.pi * (k)**2)
        
        # Color corresponds to the number of occurences of terms in database
        color.append(cluster_info[key][1])
        
        text(cluster_info[key][1],\
                cluster_info[key][0],\
                cluster_info[key][2],\
                size = 7, horizontalalignment = 'center')
    sct = scatter(x, \
                    y, \
                    c = color, \
                    s = area, \
                    linewidths = 2, \
                    edgecolor = 'w')
    sct.set_alpha(1)
    
    label = dbm.GetAccountLabel(seg)
    xmax = max([x[1] for x in cluster_info.values()])
    ymax = max([y[0] for y in cluster_info.values()])
    axis([0, xmax+(1/float(xmax)),\
        0, ymax+(1/float(ymax))])
    title('%s' % label)
    xlabel('Number of people inside the cluster')
    
    # This is the sum of occurrences in the database from all 
    # terms inside the cluster
    ylabel('''Number of term occurrences in database''')
    
    savefig('../analytics/%s/%s_bubblechart.png' % (label, label))
    close()
    
def PrintTagCloud(seg):
    from os import path
    from wordcloud import WordCloud

    d = path.dirname(__file__)
    
    # Tuple like follows (term, frequency)
    terms_in_database = dbm.GetAllRoots(seg)
    
    frequencies = []
    for term in terms_in_database:
        tag = str(term[0]).replace("'", "")
        freq = len(dbm.GetTagOccurency(tag))
        if  freq >= 20 and freq <= 25:
            frequencies += [(dbm.GetDerivatives(tag), freq),]
    
    # Gera a tag cloud da lista de tuplas das palavras e suas frequencias
    wordcloud = WordCloud().generate_from_frequencies(frequencies)

    import matplotlib.pyplot as plt
    plt.imshow(wordcloud)
    plt.axis('off')

    # lower max_font_size
    label = dbm.GetAccountLabel(seg)
    wordcloud = WordCloud(max_font_size=40).generate_from_frequencies(frequencies)
    plt.title(label)
    #plt.imshow(wordcloud)
    #plt.axis("off")
    
    plt.savefig('../analytics/%s/%s_tagcloud.png' \
        % (label, label))
    plt.close()
    
def PrintVennDiagram(A, B, C):
    # Visualization of the Venn diagram of the 3 biggest insurance-companies    
    from matplotlib_venn import venn3
    
    # Given the insurance-company id. search it followers in database
    setA, setB, setC = dbm.GetFollowerBySeg(A), \
        dbm.GetFollowerBySeg(B), \
            dbm.GetFollowerBySeg(C)
    
    labelA, labelB, labelC = dbm.GetAccountLabel(A), \
        dbm.GetAccountLabel(B),\
            dbm.GetAccountLabel(C)
    
    set1, set2, set3 = set(setA), \
        set(setB), \
            set(setC)

    venn3([set1, set2, set3], (labelA, labelB, labelC))
    plt.title('Venn diagram of the 3 biggest insurance-companies')
    plt.savefig('../analytics/venn_diagram.png')
    plt.close()
    
    
    
# Creates log file
logging.basicConfig(filename = 'posprocessing_outputs.log', \
    level = logging.DEBUG)

seguradoras = dbm.GetAllSeguradoras()
collection = dict()
followers_count = [(i, len(dbm.GetFollowerBySeg(_))) for i, _ in enumerate(seguradoras)]
foo = sorted(followers_count,\
        key = lambda x:x[1],\
        reverse = True)

PrintVennDiagram(seguradoras[foo[0][0]], seguradoras[foo[1][0]], seguradoras[foo[2][0]])


for seguradora in seguradoras:
    try: 
        label = dbm.GetAccountLabel(seguradora)
        print label
        
        data = LoadFromDataFrame(seguradora)
        
        cr = RetriveContent(data)
        
        documents = dict()
        for _, k in enumerate(data[1]):
            documents.update({k[0]:cr[_]})
            
        docs_content = np.array([_ for _ in cr])
        

        # Original data matrix
        M = np.array([lst for lst in data[2]])
        M = M.astype(np.float)
        
        # Reports to logfile
        m = json.dumps({'message': 'Working', \
            'place_at': 'Original data matrix created'}), \
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
        
        # Spherical clustering
        skm = SKMeans(n_clusters = k, random_state = 0) 
        skm.fit(M)
        clusters = skm.labels_
        
        # Reports to logfile
        m = json.dumps({'message': 'Working', \
            'place_at': 'Spherical cluster calculated'}), \
                time.asctime(time.localtime(time.time()))
        logging.info(m)

        # Hierarchical clustering
        from scipy.spatial.distance import pdist
        Z = linkage(pdist(M, 'cosine'), method = 'average')
        
        # Reports to logfile
        m = json.dumps({'message': 'Working', \
            'place_at': 'Hiearchical cluster calculated'}), \
                time.asctime(time.localtime(time.time()))
        logging.info(m)

        # Dendrogram plot
        # Calculate cut in dendrogram
        cutted = cut_tree(Z, height=0.75)
        
        docsIndex_sorted_to_clusters = DocumentIndexSorter(cutted, data)
        
        print 'Processed account: %s' \
            % dbm.GetAccountLabel(seguradora)
        
        print '\tThe ideal k, by partitional clustering, is %d' % k
        print '\tThe ideal k, by hierarchical clustering, is %d' \
            % len(docsIndex_sorted_to_clusters.keys())

        commom_terms_per_cluster = \
            GetCommomTermsInCluster(docsIndex_sorted_to_clusters, documents)
        
        
        PrintBigCluster(commom_terms_per_cluster, seguradora, collection)
        
        # Draws tag-cloud
        PrintTagCloud(seguradora)
        
        collection[seguradora] = commom_terms_per_cluster
                    
        # Draws spherical cluster
        Draw2DClusters(dissimilarity, seguradora)
    
        # Draws dendrogram
        DrawDendrogram(Z, docs_content, seguradora)
        
        # Reports to logfile
        m = json.dumps({'message': 'Working', \
            'place_at': 'Dendrograms calculated and ploted'}), \
                time.asctime(time.localtime(time.time()))
        logging.info(m)
        
    except(IOError, ValueError), e:
        print e
        next

print 'Analysis among insurance-companies'
print '\t%d total users' % dbm.GetFollowerCount()

for k in collection.keys():
    print "\t%d (%f) followers of %s"\
        % (len(dbm.GetFollowerBySeg(k))-1,\
            (float(\
                len(dbm.GetFollowerBySeg(k))-1)/float(\
                    dbm.GetFollowerCount())),dbm.GetAccountLabel(k))
import itertools
combinations = itertools.combinations(collection.keys(), 2)
print '\tCombinations '
for combination in combinations:
    total_number = len(dbm.GetFollowerFromCombination(combination[0],\
        combination[1]))
    labelA = dbm.GetAccountLabel(combination[0])
    labelB = dbm.GetAccountLabel(combination[1])
    print '\t%d (%f) followers of %s and %s' \
        % (total_number,\
            float(float(total_number)/float(dbm.GetFollowerCount())),\
                labelA,labelB)
