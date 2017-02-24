# insurance-analysis
An analysis on the clustering of insurance's follower's accounts from Twitter.
This analysis was made possible by an engine designed by me and separated into four steps;
  1. Mining data from Twitter: which means, retrievinf from Twitter's social platform the current followers from a given insurance account.
  2. Preprocessing: which was meant to be an script that would do some modeling on the original amount of data, which was obtained by the previous step. This script goes on tokenizing, stemming and creating a data structure of term root and possible derivations of that root from the documents from a given insurance's followers set, i.e. play => plays | playied.
  3. Dimensionality reduction on the original data mined previously in step #1; such script calculates the inferior, superior quartiles and the mean of the original set of terms among documents and then ignores terms which occurs less than the superior quatile (quartile 3), slicing the original dataset and copying its sliced version to a more compact and more relevant dataset.
  4. Statiscal analysis script that generates all sort of visualization charts, from bar charts to tag clouds. This last script enables the whole analysis.
Such engine was designed to have one folder to aim all analysis, this is 'analysis' folder which holds all insurance's charts.
