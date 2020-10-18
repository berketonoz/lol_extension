from riotwatcher import LolWatcher, ApiError
from itertools import permutations
from os import path
import numpy as np
import csv
import json
import requests
import pandas as pd
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


token = "?api_key=" #Insert your API key here
host = "https://tr1.api.riotgames.com/" #Learn and change to your desired server location

class LOL_Extension:
    def __init__(self,username,host=None,token=None):
        self.host = host
        self.token = token
        self.username = username

    def GetEncryptedAccountIds(self):
        return requests.get(self.host+"/lol/summoner/v4/summoners/by-name/"+self.username+self.token).json()["accountId"]

    def GetMatches(self,data):
        return requests.get(self.host+"/lol/match/v4/matchlists/by-account/"+data+self.token).json()["matches"]

    def GetMatchInfosandProcess(self,data): #data = { match_ids: champion_ids}
        result = []
        labels = []
        dataa = {match["gameId"] : match["champion"] for match in data}
        counter = 0
        for match_id in dataa: #for each match
            counter+=1
            if counter % 10 == 0: #Uncomment this clause for test pruposes
                return result,labels
            print("Counter: ", counter)
            resp = requests.get(self.host+"/lol/match/v4/matches/"+str(match_id)+self.token).json()
            #print("Response: ", resp)
            if "status" in resp: #this means we have requested too much
                return result,labels
            champ_played = dataa[match_id]
            player = { match_id : {"championId": champ_played, "Win": participant["stats"]["win"], "teamId": participant["teamId"]} for participant in resp["participants"] if participant["championId"] == champ_played }
            enemy_id = player[match_id]["teamId"]%200 + 100 #get the enemy team id
            ally_champs = [participant["championId"] for participant in resp["participants"] if participant["teamId"] == player[match_id]["teamId"] and participant["championId"] != player[match_id]["championId"]]
            enemy_champs = [participant["championId"] for participant in resp["participants"] if participant["teamId"] == enemy_id ]
            all_for_one = all(champ == enemy_champs[0] for champ in enemy_champs) #if enemies are identical all
            if all_for_one: enemy_champs = [] #if all for one then empty enemy list
            if ally_champs != [] and enemy_champs != []: #append as data if not all for one
                result.append({"GameId": match_id, "ChampionId": champ_played, "Allies": ally_champs, "Enemies": enemy_champs}) #"Win": player[match_id]["Win"]
                labels.append(1) if player[match_id]["Win"] else labels.append(0)
        return result,labels

    def GetChampions(self):
        response = requests.get("http://ddragon.leagueoflegends.com/cdn/10.18.1/data/en_US/champion.json").json()
        champ_dict = { int(response["data"][champ]["key"]) : champ for champ in response["data"].keys() }
        return champ_dict

    def ExtractAndWrite(self,infos,target):
        file_h = None
        match_ids = []
        played_champ = []
        allies = []
        enemies = []
        results = []
        if path.exists("./%s.csv"%self.username):
            rd = csv.reader(open("./%s.csv"%self.username,"r"))
            index = 0
            for row in rd:
                if index != 0:
                    match_ids.append(row[0])
                    played_champ.append(row[1])
                    allies.append([row[2],row[3],row[4],row[5]])
                    enemies.append([row[6],row[7],row[8],row[9],row[10]])
                index += 1
            wr = csv.writer(open("./%s.csv"%self.username, "w+"))
            wr.writerow(["MatchId","a1","a2","a3","a4","a5","e1","e2","e3","e4","e5"])
            for match in infos:
                match_ids.append(match["GameId"])
                played_champ.append(match["ChampionId"])
                allies.append(match["Allies"])
                enemies.append(match["Enemies"])
                r = [ match["GameId"], match["ChampionId"], match["Allies"][0], match["Allies"][1], match["Allies"][2], match["Allies"][3], match["Enemies"][0], match["Enemies"][1], match["Enemies"][2], match["Enemies"][3], match["Enemies"][4], ]
                wr.writerow(r)
    
        return {
            "Matches": match_ids,
            "Characters": played_champ,
            "Allies": allies,
            "Enemies": enemies,
            "Target": target
        }

lol = LOL_Extension("Elroid",host,token)

champions = lol.GetChampions()

encrypted_ids = lol.GetEncryptedAccountIds()
matches = lol.GetMatches(encrypted_ids)
infos,target = lol.GetMatchInfosandProcess(matches)
data = lol.ExtractAndWrite(infos,target)
print("Data: ", data)

df = pd.DataFrame(data)

X = np.asarray(df[['Characters', 'Allies', 'Enemies']])
Y = np.asarray(df['Target'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle= True)
"""
print(infos)
print("Played Hero: ", played_champ)
print("Allies: ", allies)
print("Enemies: ", enemies)
"""

"""def ProcessData(data,target):
    result_data = []
    result_target = []
    index = 0
    for d in data:
        perms = permutations(d,len(d))
        for perm in perms:
            result_data.append(list(perm))
            result_target.append(target[index])
        index += 1
    return result_data,result_target
d,t = ProcessData(data,target)
"""
"""train_x,test_x,train_y,test_y = train_test_split(data,target,test_size=0.2)
print(train_x)
print(train_y)
print(test_x)
print(test_y)

for i in range(1,10):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(train_x, train_y)
    y_pred = classifier.predict(test_x)
    print("Accuracy using %i: %0.2f%%" %(i,accuracy_score(test_y, y_pred)*100))
"""
"""model_ent = DecisionTreeClassifier(criterion='entropy', max_depth=5,random_state=4)
model_ent.fit(train_x, train_y)

model_gini = DecisionTreeClassifier(criterion='gini', max_depth=5,random_state=4)
model_gini.fit(train_x, train_y)

pred_y_ent = model_ent.predict(test_x)
print("Accuracy using Entropy: %0.2f%%" %(accuracy_score(test_y, pred_y_ent) * 100))
print("Prediction: ", pred_y_ent)

pred_y_gini = model_gini.predict(test_x)
print("Accuracy using Gini: %0.2f%%" %(accuracy_score(test_y, pred_y_gini) * 100))
print("Prediction: ", pred_y_gini)"""
