Text mining applied to a study case in Twitter
  This project proposes to make use of data from Twitter to discover which words are most frequently reproduced by the followers of the brazilian insurance company’s profiles on Twitter. This project applies Data and Text Mining techniques to, respectively, solve the problem of collecting and treating data.
  The implemented code above may contain some errors, considering the date which I finished this project and updates in the implementation of Python-2 libraries, such as Numpy, Sk-Learn, Twitter platform and related. But the big picture presented in the process pipeline is still working.
The structure above is as follows:
   # analytics/ 
      contains dimensionality reduction and plotting scripts. Read script for details.
   # lib/       
      contains persistence scripts to deal with the database. Read script for details.
   # preprocessing/
      contains text mining features. Read script for details.
   # retrieve/
      contains data retrieval scripts from Twitter platform.
   # scikit/
       contains the pipelining of the scripts presented in above folders. It is responsible for the pre-processing and pos-processing of data. It deals with data visualization also.
This was a naive github commit and my first experience with GitHub, so files commit may confuse you. To deal with that, take a closer look in the filename, which the right file is the one with the greater number appending the file name. I stopped this project in february/2017.
