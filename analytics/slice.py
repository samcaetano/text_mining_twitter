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
import sys, os

# Internal imports
sys.path.insert(0, '../lib')
import DatabaseMethods as dbm


def PlotDataFromFile(insurance_id):
    
    import matplotlib.pyplot as plt
    
    insurance_label = dbm.GetAccountLabel(insurance_id)
    
    # Reads the dataframe
    df = pd.read_csv('%s/%s.csv' % (insurance_label, insurance_label))
    
    # Drops all NaNs columns
    df = df.dropna(axis = 'columns', how = 'all')

    # Calculates the occurence frequencies of the term in the documents
    M = pd.DataFrame.as_matrix(df)
    
    terms = []
    terms_freq = []
    
    for row in M:
        terms.append(row[0])
        terms_freq.append(sum(list(row[1:])))
        
    del M
    
    # Pairs term and it's frequencies
    zipped = zip(terms, terms_freq)
    
    # Sorts decrescently terms with it's frequencies
    zipped_sorted = sorted(zipped, key = lambda t: t[1],
                           reverse = True)
    
    # Data array
    data = np.array([_[1] for _ in zipped_sorted])
    data_labels = [_[0] for _ in zipped_sorted]
    
    # Finds the quartiles and median
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    
    print '1st Quartile', q1
    print 'Median ',  median
    print '3rd Quartile', q3
    
    # Draws the bars chart. Term per occurence
    plt.figure(1)
    plt.bar(np.arange(len(data_labels)),
            data,
            align = 'center',
            alpha = 0.5)
    
    plt.xlabel('Terms')
    plt.ylabel('Occurences')
    plt.title('Occurrence of terms in documents of %s'
              % insurance_label)
    plt.savefig('%s/%s_ocorrenceTerms.png'\
        % (insurance_label, insurance_label))    
    plt.close()
    
    # Desenha o boxplot
    # Draws the boxplot
    plt.figure(2)
    plt.title('Boxplot of %s'
              % insurance_label)
    bp = plt.boxplot(data)
    plt.savefig('%s/%s_boxplot.png'\
        % (insurance_label, insurance_label))
    plt.close()
    
    # Draws the violin
    plt.figure(3)
    plt.title('Density and occurrence of terms in docs from %s'
              % insurance_label)
    plt.xlabel('Density')
    plt.ylabel('Occurrences')
    plt.violinplot(data, 
                   showmeans = False,
                   showmedians = True)
    plt.savefig('%s/%s_violinplot.png'
                % (insurance_label, insurance_label))
    
    CutFile(q3, zipped, df, insurance_label)
    
    plt.close()
    del df
    
def CutFile(q3, data, df, label):
    print 'Cutting %s file' % label
    
    # Itera entre os valores que sao menores do que o quartil 3
    sequence_to_remove = [index for index, v in enumerate(data) if v[1] < q3]
    
    # Removes terms who has freqs below the quartile3
    df = df.drop(df.index[sequence_to_remove])
    
    M = pd.DataFrame.as_matrix(df)
    
    column_to_remove = []
    for i, column in enumerate(M.T):
        try:
            if sum(list(column)) == 0:
                column_to_remove.append(i)
        except(TypeError):
            next

    del M
    
    column_names = [df.columns[index] for index in column_to_remove]
    for i, name in enumerate(column_names):
        try:
            df = df.drop(str(name), axis = 1)
            print name, ' deleted @ ', i
        except(IndexError):
            print 'erro em ', name
    
    df.to_csv('%s/%s_sliced.csv' % (label, label))  
    del df
    
    # TESTE v
    df = pd.read_csv('%s/%s_sliced.csv' % (label, label))
    M = pd.DataFrame.as_matrix(df)
    print M
    for i, column in enumerate(M.T):
        try:
            if sum(list(column)) == 0:
                print 'erro'
        except(TypeError):
            next
            
    del M
    del df
    #TESTE ^
    
    
general_insurances = dbm.GetAllSeguradoras()

for insurance_id in general_insurances:
    #insurance_id = 1202130601
    try:
        label = dbm.GetAccountLabel(insurance_id)
        if os.path.isfile('%s/%s_sliced.csv' % (label, label)):
            print '%s already sliced' % label
            next
        else:
            print 'Sliced %s started' % label
            PlotDataFromFile(insurance_id)
            print "Slice done"
    except(IOError):
        next
    except(KeyError), e:
        with open("message.err", "w") as arq:
            arq.write("\n[KeyError slicing the file]\n")
            arq.close()
        print "\t\tError reducing dimensionality"
        print e
    break
