from riotwatcher import LolWatcher, ApiError
from itertools import permutations
import csv
import json
import requests
import pandas
import time


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

"""
88master88/Bora Tandircioglu
McFarlane007/Ediz Oguz
kartoffelsalat/Yigit Bugu
AutumnWindRaider/Mert Murathanoglu
"""

token = "?api_key=RGAPI-2ae800bb-b6aa-44f6-a94a-f903ae906481"
host = "https://tr1.api.riotgames.com/"

account_names = []
with open("sample.txt") as sample:
    account_names = [ line[:line.find("/")] for line in sample.readlines() ]

class LOL_Extension:
    def __init__(self,host=None,token=None):
        self.host = host
        self.token = token

    def GetEncryptedAccountIds(self,data):
        encrypted_ids = { i : requests.get(self.host+"/lol/summoner/v4/summoners/by-name/"+i+self.token).json()["accountId"] for i in data }
        return encrypted_ids

    def GetMatches(self,data):
        matches = { i : requests.get(self.host+"/lol/match/v4/matchlists/by-account/"+data[i]+self.token).json()["matches"] for i in data }
        return matches

    def GetMatchInfosandProcess(self,data,username): #data = { match_ids: champion_ids}
        result = []
        labels = []
        dataa = {match["gameId"] : match["champion"] for match in data[username]}
        counter = 0
        for match_id in dataa: #for each match
            counter+=1
            if counter % 9 == 0:
                #break
                print("--------------------5 second break--------------------")
                time.sleep(10)
                print("--------------------Break is over--------------------")
            print("Counter: ", counter)
            resp = requests.get(self.host+"/lol/match/v4/matches/"+str(match_id)+self.token).json()
            print("Response: ", resp)
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
lol = LOL_Extension(host,token)
#lol.GetMatchInfosandProcess(matches

champions = lol.GetChampions()

encrypted_ids = lol.GetEncryptedAccountIds(account_names)
matches = lol.GetMatches(encrypted_ids)
infos,target = lol.GetMatchInfosandProcess(matches,"EforsuzBasit")
data = [match["Allies"] for match in infos ]

def ProcessData(data,target):
    result_data = []
    result_target = []
    index = 0
    for d in data:
        perms = permutations(d,len(d))
        for perm in perms:
            result_data.append(perm)
            result_target.append(target[index])
        index += 1
    return result_data,result_target
d,t = ProcessData(data,target)

train_x,test_x,train_y,test_y = train_test_split(d,t,test_size=0.1)
print(train_x)
print(train_y)
print(test_x)
print(test_y)

model_ent = DecisionTreeClassifier(criterion='entropy', max_depth=5,random_state=4)
model_ent.fit(train_x, train_y)

model_gini = DecisionTreeClassifier(criterion='gini', max_depth=5,random_state=4)
model_gini.fit(train_x, train_y)

pred_y_ent = model_ent.predict(test_x)
print("Accuracy using Entropy: %0.2f%%" %(accuracy_score(test_y, pred_y_ent) * 100))
print("Prediction: ", pred_y_ent)

pred_y_gini = model_gini.predict(test_x)
print("Accuracy using Gini: %0.2f%%" %(accuracy_score(test_y, pred_y_gini) * 100))
print("Prediction: ", pred_y_gini)