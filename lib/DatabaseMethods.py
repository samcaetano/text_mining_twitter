#!/usr/bin/python
# This file holds all database related methods

# External imports
import sqlite3 as sql
import nltk as NL
import twitter
import sys

import time


sys.path.insert(0, '/home/r2d2/projeto_ext_vero/Engine/oauth')
import OAuth

con = sql.connect("../sql/TwitterA.db")

#
def getFollower(arg):
    return OAuth.twitter_api.users.show(user_id=arg)

def getTimeline(arg, count, repeat):
    print "\tPegando a timeline de %s" % arg
    # realiza a primeira busca
    timeline =\
        OAuth.twitter_api.statuses.user_timeline(\
            screen_name=arg, count=count)
    
    # entra somente se a busca for repetida
    for _ in range(repeat):
        kwargs = dict()
        try:
            #atualiza o parametro de busca
            kwargs.update({'screen_name':arg})
            
            # atualiza o max_id
            kwargs.update({'max_id':timeline[count-1]['id']})
        except (IndexError), e:
            break
        try:
            timeline =\
                timeline + OAuth.twitter_api.statuses.user_timeline(**kwargs)
        except (twitter.api.TwitterHTTPError),e:
            break
    return timeline
#

def InsertFollowers(account):
    print "\tInserindo seguidores da %d" % account['id']    

    # recebe os seguidores da seguradora corrente...
    #...passada por parametro. Retorno incompleto
    followers = OAuth.twitter_api.followers.list(user_id = account['id'])
    for x in followers['users']:
        follower = getFollower(x['id'])
        try:
            # recebe o conjunto de tweets do usuario corrente
            follower_tweets =\
                getTimeline(follower['screen_name'], 200, 0) 
        except (twitter.api.TwitterHTTPError, NameError), e:
            next
        Insert(follower, follower_tweets, 0, account['id'])
    print

def Insert(account, tweets, isSeguradora, account_seg):
    
    print "\tInserindo %d no banco de dados" % account['id']
    
    con.execute("""INSERT OR IGNORE INTO Usuario
        (idUsuario, screen_name, name, created_at, isSeguradora, location)
        VALUES (?, ?, ?, ?, ?, ?)""",\
        (account['id'], account['screen_name'],\
            account['name'], account['created_at'],\
                isSeguradora, account['location']))
    
    for i in tweets:
        
        con.execute("""INSERT OR IGNORE INTO Tweet
            (idTweet, Usuario_idUsuario,
            is_retweeted, tweet_text,
            favorite_count, retweeted_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",\
                (i['id'], account['id'],\
                    i['retweeted'], i['text'],\
                        i['favorite_count'], i['retweet_count'],\
                            i['created_at']))

        for x in i['entities']['hashtags']:

            if hasattr(x, 'text'):
                
                con.execute("""INSERT OR IGNORE INTO Hashtag 
                    (hashtag_content) VALUES (?)""", (x['text'],))
            
                cursor = con.cursor()
            
                cursor.execute("""SELECT (idHashtag) FROM Hashtag
                    WHERE (hashtag_content = ?)""", (x['text'],))
            
                for row in cursor.fetchall():
                    idHashtag = cursor
                    con.execute("""INSERT OR IGNORE INTO Hashtag_Tweet
                        (Hashtag_idHashtag, Tweet_idTweet)
                        VALUES (?,?)""", (idHashtag, i['id']))

    if isSeguradora == 1:
        #preenche tabelas Seguradora e Seguradora_Usuario
        con.execute("""INSERT OR IGNORE INTO Seguradora 
            (idSeguradora, followers_count, statuses_count) 
            VALUES (?,?,?)""",\
                (account['id'], account['followers_count'],\
                    account['statuses_count']))
        con.execute("""INSERT OR IGNORE INTO Seguradora_Usuario
            (Seguradora_idSeguradora, Usuario_idUsuario) VALUES (?,?)""",\
                (account['id'], account['id']))
    else:
        con.execute("""INSERT OR IGNORE INTO Seguradora_Usuario 
            (Seguradora_idSeguradora, Usuario_idUsuario) VALUES (?,?)""",\
                (account_seg, account['id']))
    con.commit()
    print "\t\tInsercao feita!"

def InsertRoot(data, R, t2, segId): # 
    # data eh followers[follower]
    print '\t\tInserindo raizes no bd...'
    
    for k, v in R.iteritems():
        #
        aux = str(v).replace("""'""", "")
        aux2 = aux.replace('[', "")
        new_v = aux2.replace(']', "")
        
        #print "Root_Derivative ", k, v, segId
        con.execute("""INSERT OR IGNORE INTO Root_Derivative
                (root, derivative, Seguradora_idSeguradora)
                VALUES (?, ?, ?)""", (str(k), str(new_v), segId))
        #
        
        tweetId = 0
        for key, value in t2.iteritems():
            #print "v, value", v, value
            if set(value).intersection(v):
                tweetId = key
                #print '\t\t FOUND', tweetId
                
            if tweetId != 0:
                con.execute("""INSERT OR IGNORE INTO Root_Score
                    (Root_idRoot, Tweet_idTweet, Seguradora_idSeguradora)
                    VALUES (?,?,?)""",\
                        (k, tweetId, segId))
                #print "Root_Score ", k, tweetId, segId
        
    con.commit()
    #i = 0
    #for root in R:
    #    for e in R[root]: # 'e' eh uma lista com as derivacoes da raiz 'root'
    #        #print "Root_derivative > ", (root, str([i for i in e]), segId)
    #        con.execute("""INSERT OR IGNORE INTO Root_Derivative
    #            (root, derivative, Seguradora_idSeguradora)
    #            VALUES (?, ?, ?)""", (root, str([_ for _ in e]), segId))
    #    for i, t in enumerate(t2):
    #        try:
    #            for w in t:
    #                if(NL.stem.snowball.SnowballStemmer("portuguese").stem(\
    #                    NL.stem.snowball.SnowballStemmer("english").stem(\
    #                        w.lower()).lower()) == root):
    #                    con.execute("""INSERT OR IGNORE INTO Root_Score
    #                        (Root_idRoot, Tweet_idTweet, Seguradora_idSeguradora)
    #                        VALUES (?,?,?)""", (root, [index for index in data.keys()][i], segId))
    #        except(IndexError, e):
    #            next
    #con.commit()
    
def InsertPreprocessedTweets(arg):
    print '\t\tInserindo no banco de dados...'
    
    keys = arg.keys()
    values = arg.values()
    
    for _ in zip(keys, values):
        con.execute("""UPDATE OR IGNORE Tweet SET tweet_preprocessed=?
        WHERE idTweet LIKE ?""", (' '.join(_[1]), _[0]))
    #for k, v in arg.iteritems():
    #    con.execute("""UPDATE OR IGNORE Tweet SET tweet_preprocessed=?
    #    WHERE idTweet LIKE ?""", (' '.join(v), k))
    
    
    con.commit()
    print '\t\tTerminado'
    
def GetUniverse():
    cursor = con.cursor()
    
    print "\t\tLendo banco de dados..."
    universe = dict() # universe : conjunto universo de todas as seguradoras
    
    cursor.execute("""SELECT * FROM Seguradora_Usuario""")
    for row in cursor.fetchall():
        if row[0] != row[1]:
            if row[0] in universe:
                if row[1] not in universe:
                    universe[row[0]].append(row[1])
            else:
                if row[1] not in universe:
                    universe[row[0]] = [row[1]]
    return universe

def GetFollowerTweets(U, seguradoraId):
    cursor = con.cursor()

    followers_collection = dict()
    # Percorre cada seguidor da seguradora
    for follower in U[seguradoraId]:
        tweets_per_follower = dict()
        
        cursor.execute("""SELECT idTweet, tweet_text FROM Tweet
        WHERE (Usuario_idUsuario = ?)""",(follower,))
        
        for row in cursor.fetchall():
            tweets_per_follower.update({row[0]:row[1]})
            
        # Agrupa todos os tweets do seguidor corrente da seguradora
        followers_collection.update({follower:tweets_per_follower})
        
    return followers_collection

def GetAccountLabel(arg):
    cursor = con.cursor()
    cursor.execute("""SELECT screen_name FROM Usuario 
        WHERE (idUsuario = ?)""", (arg,))
    for row in cursor.fetchall():
        return row[0]
    
def GetDerivatives(arg):
    cursor = con.cursor()
    cursor.execute("""SELECT derivative FROM Root_Derivative
        WHERE root = ?""", (arg,))
    for row in cursor.fetchall():
        return row[0]
    
def GetAllRoots(segId):
    cursor = con.cursor()
    cursor.execute("""SELECT root FROM Root_Derivative
        WHERE Seguradora_idSeguradora = ?""", (segId,))
    return cursor.fetchall()

def GetTagOccurency(tag):
    cursor = con.cursor()
    cursor.execute("""SELECT Root_idRoot FROM Root_Score
        WHERE Root_idRoot = ?""", (tag,))
    return cursor.fetchall()
    
def GetTweetIdByTerm(arg, seg):
    tweets = []
    cursor = con.cursor()
    cursor.execute("""SELECT Tweet_idTweet FROM Root_Score
        WHERE Root_idRoot = ? and Seguradora_idSeguradora = ?""",\
            (arg, seg))
    for row in cursor.fetchall():
        tweets += [row[0],]
    return tweets

def GetFollowerByTweetId(arg):
    cursor = con.cursor()
    cursor.execute("""SELECT Usuario_idUsuario FROM Tweet 
        WHERE idTweet = ?""", (arg,))
    for row in cursor.fetchall():
        return row[0]

def GetFollowerByAccount(arg, arg1):
    userByTweetId = list(set([GetFollowerByTweetId(_) for _ in arg]))
    userBySeguradora = []
    cursor = con.cursor()
    cursor.execute("""SELECT Usuario_idUsuario FROM Seguradora_Usuario
        WHERE Seguradora_idSeguradora = ?""", (arg1,))
    for row in cursor.fetchall():
        userBySeguradora += [row[0],]
    return list(set(userBySeguradora) & set(userByTweetId))

def GetFollowerCount():
    cursor = con.cursor()
    cursor.execute("""SELECT idUsuario FROM Usuario WHERE isSeguradora = 0""")
    return len(cursor.fetchall())

def GetFollowerBySeg(seg):
    cursor = con.cursor()
    cursor.execute("""SELECT Usuario_idUsuario FROM Seguradora_Usuario
        WHERE Seguradora_idSeguradora = ?""", (seg,))
    return cursor.fetchall()

def GetAllSeguradoras():
    cursor = con.cursor()
    cursor.execute("""SELECT idSeguradora FROM Seguradora
        WHERE followers_count > 1""")
    return [_[0] for _ in cursor.fetchall()]

def GetTermBySeg(seg):
    cursor = con.cursor()
    cursor.execute("""SELECT Root_idRoot FROM Root_Score
        WHERE Seguradora_idSeguradora = ?""", (seg,))
    return cursor.fetchall()

def GetTweetById(tweetId):
    cursor = con.cursor()
    cursor.execute("""SELECT tweet_preprocessed FROM Tweet
        WHERE idTweet = ?""", (tweetId,))
    return cursor.fetchall()

def GetLocations(seg):
    # Lista de seguidores da 
    seg_followers = []
    
    cursor = con.cursor()
    cursor.execute("""SELECT Usuario_idUsuario FROM Seguradora_Usuario
        WHERE Seguradora_idSeguradora = ?""", (seg,))
    
    for user in cursor.fetchall():
        
        cursor.execute("""SELECT screen_name, place_name FROM Usuario
            WHERE place_name != '' AND idUsuario = ?""", (user[0],))
        seg_followers += cursor.fetchall()
    return seg_followers

def GetFollowerFromCombination(a, b):
    cursor = con.cursor()
    cursor.execute("""SELECT Usuario_idUsuario from Seguradora_Usuario 
        WHERE Seguradora_idSeguradora = ? 
        INTERSECT 
        SELECT Usuario_idUsuario FROM Seguradora_Usuario 
        WHERE Seguradora_idSeguradora = ?""", (a, b))
    return cursor.fetchall()
