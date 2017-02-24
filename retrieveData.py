#!/usr/bin/python
# External imports
import sqlite3 as sql
import twitter
import json
import time
import sys
import os
import pickle

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
    # Receives a set with the current insurance's followers 
    followers_ids = OAuth.twitter_api.followers.ids(\
        user_id = arg, count = count)
    
    # Saves the ids from the current insurance's followers
    followers = followers_ids['ids']
    kwargs = dict()
    
    # Go through the search
    while followers_ids['next_cursor'] != 0:
    
        # >next_cursor< indicates the next followers' page id
        try:
            kwargs.update({'user_id':arg})
            kwargs.update({'count':count})
            kwargs.update({'cursor':followers_ids['next_cursor']})
        except (IndexError), e:
            print '\tget_followers_current_user() :: ', e
            break
        try:
            # Do the search through the available >cursors<
            followers_ids = OAuth.twitter_api.followers.ids(**kwargs)
            followers = followers + followers_ids['ids']
        except (twitter.api.TwitterHTTPError), e:
            print '\tget_followers_current_user() :: ', e
            break
    return followers

# >repeat< determines how many times to loop this search
def get_tweets_current_user(q, count, repeat):
    # Do the search
    try:
        return OAuth.twitter_api.statuses.user_timeline(\
            user_id = q, count = count, trim_user = True)
    except(Exception), e:
        print '\tget_tweets_current_user() :: ', e

# Retrieves the account of the user given as parameter
def get_user(arg):
    try:
        user = OAuth.twitter_api.users.lookup(user_id=arg)
        return user[0]
    except(Exception), e:
        print '\tget_user() :: ', e

# Procura pela conta de um usuario passado por parametro
def search_user(arg):
    try:
        return OAuth.twitter_api.users.search(q=arg)
    except(Exception), e:
        print '\tsearch_user() :: ', e

# Creates files with the followers' data, tweets and account
# (respectively) from a given insurance
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

# Retrieve the followers from a given insurance from file
def get_followers_from_file(id_name):
    print 'get_seguradora_from_file()'
    str1 = str(id_name)+"Followers.txt"
    obj = open(str1,"r")
    f = pickle.load(obj)
    return f

# Pega os tweets da seguradora a partir do arquivo
def get_tweets_from_file(id_name):
    print 'get_tweets_from_file()'
    str1 = str(id_name)+"Tweets.txt"
    obj = open(str1,"r")
    f = pickle.load(obj)
    return f

# Pega os dados da conta de uma seguradora a partir de um arquivo
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
    
# Insere no banco de dados as infos referentes a seguradora corrente
def insertSeguradoraFromFile(name):
    print 'insertSeguradoraFromFile()'
    tweets = get_tweets_from_file(name)
    seguradora = get_seguradora_from_file(name)
    insert(seguradora, tweets, 1, 0)

# Insere as infos de cada seguidor de uma determinada seguradora
def insertFollowerFromFile(id_seguradora, followers, followers_count,\
    followers_protected, followers_noTweets, followers_in, control):
    
    print 'insertFollowerFromFile()'
    
    # Le do banco de dados todos os seguidores da seguradora...
    #...corrente previamente cadastrados
    already_in = [_[0] for _ in dbm.GetFollowerBySeg(id_seguradora)]

    # Elimina dos seguidores a serem inseridos...
    #...os seguidores, desta seguradora, ja inseridos
    followers = [_ for _ in followers if _ not in already_in]
    
    # Marca a posicao do atual seguidor.
    # Indica cada indice da list de seguidores
    k = 0
    
    # Se verdadeiro, entao ainda restam seguidores para cadastrar.
    # Se falso, nao restam seguidores
    resta_seguidor = True
    
    current_user = dict()
    
    for item in followers:
        if item == -1:
            # Control para saber se todos os seguidores foram inseridos
            control = control + 1
    
    if control >= len(followers):
        resta_seguidor = False
        
    if resta_seguidor:
        for follower in followers:
            # Se follower for -1, entao quer dizer
            # que esse follower ja foi
            # cadastrado com sucesso. Nao entra
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
                        else:
                            followers_noTweets = followers_noTweets + 1
                            followers_count -= 1
                            control += 1
                            print "%d ignorado por nao possuir tweets.\
                            Tweets disponiveis %d"\
                                % (follower, len(user_timeline))
                        
                        # Se o seguidor for cadastrado com sucesso...
                        #...entao sua posicao na 'list'...
                        #...eh substituida por um -1
                        followers[k] = -1
                    except (Exception), e:
                        # Checa se o retorno eh do NoneType...
                        #...que significa que o usuario esta protegido
                        if str(e).find('NoneType'):
                            
                            followers_protected += 1                            
                            followers[k] = -1
                            followers_count -= 1
                            control += 1
                            print 'Nao autorizado ',
                            
                        print "%d nao inserido. Exception [%s]"\
                            % (follower, e)
                        next
                else:
                    # Se o seguidor corrente da seguradora corrente ja estiver...
                    #...cadastrado no banco de dados, entao so se insere...
                    #...no banco a relacao usuario-seguradora. E nao se...
                    #...insere todas as infos deste seguidor
                    print follower
                    if follower:
                        insert_SegUser(follower, id_seguradora)
                        followers_in = followers_in + 1
                        followers[k] = -1
                        next
            else:
                print k,'pulou'
            k = k + 1
        # Chama a funcao recursivamente, de modo que a cada chamada
        # a list followers eh passada com os seguidores que ainda
        # restam para serem cadastrados
        insertFollowerFromFile(id_seguradora, followers,\
            followers_count, followers_protected,\
                followers_noTweets, followers_in, control)
    else:
        raise Exception('Usuarios disponiveis: ',followers_count,\
            ' | Usuarios inseridos: ',followers_in,\
                ' | Usuarios sem tweets: ',followers_noTweets,\
                    ' | Usuarios protegidos: ', followers_protected)
                   
# Verifica se o seguidor corrente ja esta no banco.
# Se esta, available() retorna False. Se nao esta, retorna True
def available(arg):
    cursor2 = con.cursor()
    cursor2.execute("""SELECT idUsuario FROM Usuario
        WHERE (idUsuario = ?)""",(arg,))
    for row in cursor2.fetchall():
        for id in row:
            if id == arg:
                return False
    return True

# Verifica se o tweet corrente ja esta no banco de dados.
# Se sim, retorna False, caso contrario retorna True
def availableTweet(arg):
    cursor = con.cursor()
    cursor.execute("""SELECT idTweet FROM Tweet
        WHERE (idTweet = ?)""", (arg,))
    for rw in cursor.fetchall():
        for id in rw:
            if id == arg:
                return False
    return True

# Mesma verificacao de available() e availableTweet()
def availableHash(arg):
    cursor = con.cursor()
    cursor.execute("""SELECT hashtag_content FROM Hashtag
        WHERE (hashtag_content = ?)""", (arg,))
    for rw in cursor.fetchall():
        for _ in rw:
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
        # Preenche tabelas Seguradora e Seguradora_Usuario
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

def insert_SegUser(account, account_seg):
    cursor = con.cursor()
    cursor.execute("""SELECT (Seguradora_idSeguradora) FROM Seguradora_Usuario
        WHERE Usuario_idUsuario = ?""", (account,))
    for row in cursor.fetchall():
        for id in row[0]:
            if(id != account_seg):
                con.execute("""INSERT OR IGNORE INTO Seguradora_Usuario
                    (Seguradora_idSeguradora, Usuario_idUsuario) VALUES (?,?)""",\
                        (account_seg, account))
                con.commit()

def main():
    try:
        is_Seg = 1 # 0, entao usuario; 1, entao seguradora
        name = raw_input('Nome da conta: ')
        search_account = search_user(name)
        print 'Resuts:'
        for i, v in enumerate(search_account):
            print '%d | %s' % (i+1, v['name'])
            
        option = input('Escolha a conta (1-10):')
        user = search_account[option-1]
        seguradora_to_file(user['id'])
        return user
    except(twitter.api.TwitterHTTPError), e:
        print 'Error in main() :: ', e
        print 'Waiting API time'
        time.sleep(900)

usr = main() # Recebe os dados da seguradora assim que termina o main

time_begin = time.asctime(time.localtime(time.time()))
insertSeguradoraFromFile(usr['id']) # Insere a seguradora no banco de dados
print 'Em execucao....'
try:
    # Recebe a quantidade de seguidores da seguradora
    followers_count = usr['followers_count']
    
    followers = get_followers_from_file(usr['id'])
    
except(twitter.api.TwitterHTTPError), e:
    print 'Erro', e

insertFollowerFromFile(usr['id'], followers, followers_count, 0, 0, 0, 0)

print 'Iniciado as: ',time_begin
print 'Finalizado as: ',time.asctime(time.localtime(time.time()))

delete_files(usr['id'])
con.close()
