#!/usr/bin/python

# This script was developed by Samuel Caetano
# Data miner script. 
# Using Twitter API, and OAuth 2.0 authentication, retrieves all available
# followers from a given account and inserts it properly into the database

# External imports
import sqlite3 as sql
import twitter
import json
import time
import sys
import os
import pickle
import logging

# Internal imports
sys.path.insert(0, \
    '/home/r2d2/projeto_ext_vero/Engine/oauth')
import OAuth

sys.path.insert(0, \
    '../lib')
import DatabaseMethods as dbm

con = sql.connect('/home/r2d2/projeto_ext_vero/Engine/sql/TwitterA.db')

# Search the followers from a given user
def get_followers_current_user(arg, count):
    # Receives a set with the current insurance's company followers
    followers_ids = OAuth.twitter_api.followers.ids(\
        user_id = arg, count = count)
    
    # Saves the ids from the current insurance's company followers
    followers = followers_ids['ids']
    kwargs = dict()
    
    # Go through the search
    while followers_ids['next_cursor'] != 0:
        # >next_cursor< indicates the next followers' page id
        try:
            kwargs.update({'user_id':arg})
            kwargs.update({'count':count})
            kwargs.update({'cursor':followers_ids['next_cursor']})
        except IndexError as e:
            m = """Unexpected index error in 
            get_followers_current_user() \t""", \
                time.asctime(time.localtime(time.time()))
            print m
            logging.warning(m)
            break
        try:
            # Do the search through the available >cursors<
            followers_ids = OAuth.twitter_api.followers.ids(**kwargs)
            followers = followers + followers_ids['ids']
        except twitter.api.TwitterHTTPError as e:
            if str(e).find('401'):
                m = json.dumps({'code': 401, \
                    'message': 'Unauthorized', 'account': arg}), \
                        time.asctime(time.localtime(time.time()))
                print m
                logging.info(m)
            elif str(e).find('429'):
                m = json.dumps({'code': 429, \
                    'message': 'Rate limit exceeded'}), \
                        time.asctime(time.localtime(time.time()))
                print m
                logging.info(m)
            break
    return followers

# >repeat< determines how many times to loop this search
def get_tweets_current_user(q, count, repeat):
    # Do the search
    try:
        return OAuth.twitter_api.statuses.user_timeline(\
            user_id = q, count = count, trim_user = True)
    except twitter.api.TwitterHTTPError as e:
        if str(e).find('401'):
            m = json.dumps({'code': 401, \
                'message': 'Unauthorized', 'account': q}), \
                    time.asctime(time.localtime(time.time()))
            print m
            logging.info(m)
        elif str(e).find('429'):
            m = json.dumps({'code': 429, \
                'message': 'Rate limit exceeded'}), \
                    time.asctime(time.localtime(time.time()))
            print m
            logging.info(m)

# Retrieves the account of the user given as parameter
def get_user(arg):
    try:
        user = OAuth.twitter_api.users.lookup(user_id=arg)
        return user[0]
    except twitter.api.TwitterHTTPError as e:
        if str(e).find('401'):
            m = json.dumps({'code': 401, \
                'message': 'Unauthorized', 'account': arg}), \
                    time.asctime(time.localtime(time.time()))
            print m
            logging.info(m)
        elif str(e).find('429'):
            m = json.dumps({'code': 429, \
                'message': 'Rate limit exceeded'}), \
                    time.asctime(time.localtime(time.time()))
            print m
            logging.info(m)

# Searches for the account of the user passed by parameter
def search_user(arg):
    try:
        return OAuth.twitter_api.users.search(q=arg)
    except twitter.api.TwitterHTTPError as e:
        if str(e).find('401'):
            m = json.dumps({'code': 401, \
                'message': 'Unauthorized', 'account': arg}), \
                    time.asctime(time.localtime(time.time()))
            print m
            logging.info(m)
        elif str(e).find('429'):
            m = json.dumps({'code': 429, \
                'message': 'Rate limit exceeded'}), \
                    time.asctime(time.localtime(time.time()))
            print m
            logging.info(m)

# Creates files with the followers' data, tweets and account
# (respectively) from a given insurance company
def seguradora_to_file(arg):
    str1 = str(arg)+"Followers.txt"
    str2 = str(arg)+"Tweets.txt"
    str3 = str(arg)+".txt"
    
    obj = open(str1, "a")
    pickle.dump(get_followers_current_user(arg, 5000), obj)
    obj.close()
    
    obj = open(str2,"a")
    pickle.dump(get_tweets_current_user(arg, 200, 0), obj)
    obj.close()
    
    obj = open(str3,"a")
    pickle.dump(get_user(arg), obj)
    obj.close()

# Retrieve the followers from a given insurance company from file
def get_followers_from_file(id_name):
    print 'get_seguradora_from_file()'
    str1 = str(id_name)+"Followers.txt"
    obj = open(str1,"r")
    f = pickle.load(obj)
    return f

# Retrieve insurance's company tweets from file
def get_tweets_from_file(id_name):
    print 'get_tweets_from_file()'
    str1 = str(id_name)+"Tweets.txt"
    obj = open(str1,"r")
    f = pickle.load(obj)
    return f

# Retrieve insurance's company account data from file
def get_seguradora_from_file(id_name):
    print 'get_seguradora_from_file()'
    str1 = str(id_name)+".txt"
    obj = open(str1,"r")
    f = pickle.load(obj)
    return f

def delete_files(arg):
    print 'delete_files()'
    str1 = str(arg)+"Followers.txt"
    str2 = str(arg)+"Tweets.txt"
    str3 = str(arg)+".txt"
    
    os.remove(str1)
    os.remove(str2)
    os.remove(str3)

# Inserts into the database the infos about the current insurance company
def insertSeguradoraFromFile(name):
    print 'insertSeguradoraFromFile()'
    tweets = get_tweets_from_file(name)
    seguradora = get_seguradora_from_file(name)
    insert(seguradora, tweets, 1, 0)

# Inserts the info from each follower of a given insurance company
def insertFollowerFromFile(id_seguradora, followers, followers_count,\
    followers_protected, followers_noTweets, followers_in, control):
    
    print 'insertFollowerFromFile()'
    
    # Reads from database the whole insurance's companies followers
    # previously inserted
    already_in = [_[0] for _ in dbm.GetFollowerBySeg(id_seguradora)]

    # Removes from followers to be inserted
    # those, who belongs to this insurance company, already inserted
    followers = [_ for _ in followers if _ not in already_in]

    # Marks the position of the present follower
    # Indicates each index of a list of followers
    k = 0
    
    # If true, then there are followers to be inserted yet
    # If false, there are none
    follower_left = True
    
    current_user = dict()
    
    for item in followers:
        if item == -1:
            # >control< show if all the followers were correctly inserted
            control += 1
    
    if control >= len(followers):
        follower_left = False
        
    # Recursion stop criterion
    if follower_left:
        for follower in followers:
            # If >follower == -1<, then follower already inserted
            # Goes to >else<
            if follower != -1:
                if available(follower):
                    time.sleep(1)
                    try:
                        user_timeline = \
                            get_tweets_current_user(follower, 200, 0)
                            
                        if len(user_timeline) != 0:
                            current_user = get_user(follower)
                        
                            insert(current_user, user_timeline, 0,\
                                id_seguradora)
                            
                            followers_in = followers_in + 1
                            
                            m = json.dumps({'code': 200, \
                                'message': 'Sucess!', \
                                'account': follower}), \
                                    time.asctime(time.localtime(time.time()))
                                
                            print m
                            logging.info(m)
                            
                        else:
                            followers_noTweets = followers_noTweets + 1
                            followers_count -= 1
                            control += 1
                            m = json.dumps({'code': 304, \
                                'message': 'No tweets available', \
                                'account': follower}), \
                                    time.asctime(time.localtime(time.time()))
                            print m
                            logging.info(m)
                        
                        # If follower correctly inserted then it's value 
                        # in it's position in the list is replaced by -1
                        followers[k] = -1
                    except Exception as e:
                        # Check if returned content is NoneType
                        # which means this follower is protected
                        if str(e).find('NoneType'):
                            followers_protected += 1                            
                            followers[k] = -1
                            followers_count -= 1
                            control += 1
                        next
                else:
                    # If current follower from the current insurance company
                    # had already been inserted, then it only inserts the 
                    # user-insurance relantionship. 
                    # And ignores the whole follower info
                    InsertInsuranceFollower(follower, id_seguradora)
                    followers_in = followers_in + 1
                    followers[k] = -1
                    next
            else:
                print k,' jumped'
            k += 1
        # Uses recurvise call of this function, in a way that
        # >followers< is passed with those followers 
        # who still need to be inserted
        insertFollowerFromFile(id_seguradora, followers,\
            followers_count, followers_protected,\
                followers_noTweets, followers_in, control)
    else:
        m = json.dumps({'code':200, \
            'message': 'Finished!', \
            'users': {
                'available': followers_count,
                'inserted': followers_in,
                'no-tweet': followers_noTweets,
                'protected': followers_protected}})
        
        # Outputs to log file
        logging.info(m)
        
        raise Exception(m)

# Verifies if current follower is already in the database;
# If it is in, return false. otherwise, returns true
def available(arg):
    cursor2 = con.cursor()
    cursor2.execute("""SELECT idUsuario FROM Usuario
        WHERE (idUsuario = ?)""",(arg,))
    for row in cursor2.fetchall():
        for _ in row:
            if _ == arg:
                return False
    return True

# Verifies if current tweet is already in database
# If it is in, returns false. otherwise returns true
def availableTweet(arg):
    cursor = con.cursor()
    cursor.execute("""SELECT idTweet FROM Tweet
        WHERE (idTweet = ?)""", (arg,))
    for row in cursor.fetchall():
        for _ in row:
            if _ == arg:
                return False
    return True

# Same verification as >available()< and >availableTweet()<
def availableHash(arg):
    cursor = con.cursor()
    cursor.execute("""SELECT hashtag_content FROM Hashtag
        WHERE (hashtag_content = ?)""", (arg,))
    for row in cursor.fetchall():
        for _ in row:
            if _ == arg:
                return False
    return True
    
def insert(account, tweets, isSeguradora, account_seg):
    con.execute("""INSERT OR IGNORE INTO Usuario (idUsuario, screen_name,
        name, created_at, isSeguradora, place_name)
        VALUES (?, ?, ?, ?, ?, ?)""",\
            (account['id'], account['screen_name'],\
                account['name'], account['created_at'],\
                    isSeguradora, account['location']))
        
    for i in tweets:
        if availableTweet(i['text'].lower()) == True:
            con.execute("""INSERT OR IGNORE INTO Tweet
                (idTweet, Usuario_idUsuario, is_retweeted, tweet_text,
                favorite_count, retweeted_count)
                VALUES (?,?,?,?,?,?)""",\
                    (i['id'], account['id'],\
                        i['retweeted'], i['text'].lower(),\
                            i['favorite_count'], i['retweet_count']))
                
            for x in i['entities']['hashtags']:
                if availableHash(x['text'].lower()) == True:
                    con.execute("""INSERT OR IGNORE INTO Hashtag
                        (hashtag_content) VALUES (?)""", (x['text'].lower(),))
                    
                cursor = con.cursor()
                
                cursor.execute("""SELECT (idHashtag) FROM Hashtag
                    WHERE (hashtag_content = ?)""", (x['text'].lower(),))
                
                for row in cursor.fetchall():
                    for idHashtag in row:
                        con.execute("""INSERT OR IGNORE INTO
                            Hashtag_Tweet (Hashtag_idHashtag, Tweet_idTweet)
                            VALUES (?,?)""", (idHashtag, i['id']))
    if isSeguradora == 1:
        # Fills Seguradora and Seguradora_Usuario tables
        con.execute("""INSERT OR IGNORE INTO Seguradora
            (idSeguradora, followers_count, statuses_count) VALUES (?,?,?)""",\
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

def InsertInsuranceFollower(account, account_seg):
    cursor = con.cursor()
    cursor.execute('''SELECT (Seguradora_idSeguradora) FROM Seguradora_Usuario
        WHERE Usuario_idUsuario = ?''', (account,))
    for row in cursor.fetchall():
        for id in row:
            if(id != account_seg):
                con.execute("""INSERT OR IGNORE INTO Seguradora_Usuario
                (Seguradora_idSeguradora, Usuario_idUsuario) VALUES (?,?)""",\
                (account_seg, account))
                con.commit()

def main():
    try:
        name = raw_input('Account name: ')
        search_account = search_user(name)
        print 'Resuts:'
        for i, v in enumerate(search_account):
            print '%d | %s' % (i+1, v['name'])
            
        option = input('Choose an account (1-10):')
        user = search_account[option-1]
        seguradora_to_file(user['id'])
        
        # Outputs to log file
        logging.info('Getting %s information' % dbm.GetAccountLabel(user['id']))
        return user
    except twitter.api.TwitterHTTPError as e:
        if str(e).find('401'):
            m = json.dumps({'code': 401, \
                'message': 'Unauthorized', 'account': user['id']})
            print m
            
            # Outputs to log file
            logging.info(m)
            
        elif str(e).find('429'):
            m = json.dumps({'code': 429, \
                'message': 'Rate limit exceeded'})
            print m
            
            # Outputs to log file
            logging.info(m)
        print 'Waiting API time'
        time.sleep(900)
        return user

logging.basicConfig(filename = 'results.log', \
    level = logging.DEBUG)

# Receives the data from the insurance-company
usr = main() 

# Gets current time
time_begin = time.asctime(time.localtime(time.time()))


# Inserts the insurance-company in the database
insertSeguradoraFromFile(usr['id']) 
print 'In execution...'
try:
    # Gets the quantity of followers from the 
    # current insurance-company
    followers_count = usr['followers_count']
    
    followers = get_followers_from_file(usr['id'])
    
except(twitter.api.TwitterHTTPError), e:
    print 'Error ', e
    logging.warning(e)

insertFollowerFromFile(usr['id'], followers, followers_count, 0, 0, 0, 0)

print 'Started at: ', time_begin
print 'Done at: ', time.asctime(time.localtime(time.time()))

# Delete used files
delete_files(usr['id'])

# Closes database connection
con.close()
