#!/usr/bin/python

# This script was developed by Samuel Caetano
# Dimentionality reduction script. 
# This script works on reducing the dimensionality of the dataset. To generate 
# the cuts it calculates the inferior and superior quartiles, and then ignores
# all terms which frequency is below the superior quartile (quartile 3). Draws
# some charts for further analysis

# External imports
import pandas as pd
import numpy as np
import sys

# Internal imports
sys.path.insert(0, '../lib')
import DatabaseMethods as dbm


def PlotDataFromFile(insurance_id):
    
    import matplotlib.pyplot as plt
    
    insurance_label = dbm.GetAccountLabel(insurance_id)
    
    # Reads the dataframe
    df = pd.read_csv('%s/%s.csv' % (insurance_label, insurance_label))
    
    # Drops all NaNs columns
    df = df.dropna(axis = 'columns',\
        how = 'all')

    # Calculates the occurence frequencies of the term in the documents
    terms = []
    terms_freq = []
    for row in df.index:
        
        occurences = 0
        
        for column in df.columns[1:]:
            occurences += df[column][row]
        
        terms += [df['Unnamed: 0'][row],]
        terms_freq += [occurences,]
        
    # Pairs term and it's frequencies
    zipped = zip(terms, terms_freq)
    
    # Sorts decrescently terms with it's frequencies
    zipped_sorted = sorted(zipped, key = lambda t: t[1],\
        reverse = True)
    
    # Data array
    data = np.array([_[1] for _ in zipped_sorted])
    data_labels = [_[0] for _ in zipped_sorted]
    
    # Finds the quartiles and median
    q1, medians, q3 = np.percentile(data, [25, 50, 75])
    
    print q1, medians, q3
    
    # Draws the bars chart. Term per occurence
    plt.figure(1)
    plt.bar(np.arange(len(data_labels)),\
            data,\
            align = 'center',\
            alpha = 0.5)
    
    plt.xlabel('Terms')
    plt.ylabel('Occurences')
    plt.title('Occurrence of terms in documents of %s'\
        % insurance_label)
    plt.savefig('%s/%s_ocorrenceTerms.png'\
        % (insurance_label, insurance_label))    
    plt.close()
    
    # Desenha o boxplot
    # Draws the boxplot
    plt.figure(2)
    plt.title('Boxplot of %s'\
        % insurance_label)
    bp = plt.boxplot(data)
    plt.savefig('%s/%s_boxplot.png'\
        % (insurance_label, insurance_label))
    plt.close()
    
    # Draws the violin
    plt.figure(3)
    plt.title('Density and occurrence of terms in docs from %s'\
        % insurance_label)
    plt.xlabel('Density')
    plt.ylabel('Occurrences')
    plt.violinplot(data,\
        showmeans = True,\
        showmedians = True,\
            showextrema = True)
    plt.savefig('%s/%s_violinplot.png'\
        % (insurance_label, insurance_label))
    
    CutFile(q1, q3, zipped, df, insurance_label)
    
    plt.close()
    
def CutFile(quartile1, quartile3, data, df, label):
    
    print 'Cutting %s file' % label
    
    sequence_to_remove = []
    
    for i, _ in enumerate(data):
        if _[1] < quartile3:
            sequence_to_remove += [i,]
    
    df = df.drop(df.index[sequence_to_remove])
    
    sequence_to_remove = []
    for c in df.columns:
        if c != 'Unnamed: 0':
            acc = 0.
            for r in df.index:
                acc += df[c][r]
            if acc == 0.:
                print c, ' removed'
                #sequence_to_remove += [c,]
                df = df.drop(c, axis = 1)
    
    df.to_csv('%s/%s_sliced.csv'\
        % (label, label))
    
general_insurances = dbm.GetAllSeguradoras()

for insurance_id in general_insurances:
    try:
        PlotDataFromFile(insurance_id)
    except(IOError):
        next
    except(KeyError), e:
        with open("ERR.txt", "w") as arq:
            arq.write("\n[KeyError slicing the file]\n")
        print "\t\tError reducing dimensionality"
        print e
    print "\t\tSlice done"
    
