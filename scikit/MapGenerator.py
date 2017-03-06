# Internal imports
import sys
sys.path.insert(0, '../lib')
import DatabaseMethods as dbm

# External imports
import json
import folium
import geocoder

seguradoras = dbm.GetAllSeguradoras()

for seg in seguradoras:
    label = dbm.GetAccountLabel(seg)
    
    print 'Account: %s' % label

    latitudes = []
    longitudes = []
    user_names = []
    
    # Lista com tuplas de (usuario, localizacao)
    locations = dbm.GetLocations(seg)
    
    # Aplica expressao regular para limpar a localizacao
    for i, _ in enumerate(locations):
        print '%d of %d' \
            % (i+1, len(locations))
        try:
            g = geocoder.google(_[1])
            
            print g
            
            latitudes += [g.json['lat'],]
            longitudes += [g.json['lng'],]
            user_names += ["@"+_[0],]
        except(Exception) as e:
            print e
            next
        
    
    # Pega um mapa basico
    tweets_map = folium.Map(location = [30.0, 0.0], zoom_start = 2)
    
    for index, user in enumerate(user_names):
        folium.CircleMarker(\
            location = [latitudes[index], longitudes[index]],\
            popup = user_names[index],\
                radius = 100).add_to(tweets_map)
        
    # Cria e mostra o mapa
    tweets_map.save('../analytics/%s/%s_followers_map.html' \
        % (label, label))
    tweets_map
