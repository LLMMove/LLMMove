import openai
import random
import os
import json
from tqdm import tqdm

api_key  = YOUR_API_KEY
from openai import OpenAI
client = OpenAI(api_key=api_key)

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def openaiAPIcall(**kwargs):
    return client.chat.completions.create(**kwargs)

from math import radians, sin, cos, sqrt, atan2
def haversine_distance(lat1, lon1, lat2, lon2):
    lat1 = eval(lat1)
    lon1 = eval(lon1)
    lat2 = eval(lat2)
    lon2 = eval(lon2)
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    radius = 6371.0
    distance = radius * c
    return distance
    

class LLMMove():
    def run(self, data, datasetName):
        self.datasetName = datasetName
        self.longs, self.recents, self.targets, self.poiInfos, self.traj2u = data
        poiList = list(self.poiInfos.keys())
        hit1 = 0
        hit5 = 0
        hit10 = 0
        rr = 0
        err = list()
        for trajectory, groundTruth in tqdm(self.targets.items()):
            seed_value = eval(trajectory)
            random.seed(seed_value)
            negSample = random.sample(poiList, 100)
            candidateSet = negSample + [groundTruth[0]]
            try:
                prediction = self.runeach(trajectory, candidateSet, groundTruth)
                if groundTruth[0] in prediction:
                    index = prediction.index(groundTruth[0]) + 1
                    if index == 1:
                        hit1 += 1
                    if index <= 5:
                        hit5 += 1
                    hit10 += 1
                    rr += 1 / index
                else:
                    err.append(eval(trajectory))
            except Exception as e:
                print(repr(e))
        err = list(sorted(err))
        with open('./testERR', 'a') as file:
            file.write(str(err))
        num_trajectories = len(self.targets)
        acc1 = hit1 / num_trajectories
        acc5 = hit5 / num_trajectories
        acc10 = hit10 / num_trajectories
        mrr = rr / num_trajectories
        print(f'acc@1: {acc1}, acc@5: {acc5}, acc@10: {acc10}, mrr@10: {mrr}')
        return acc1, acc10, mrr
    
    def runeach(self, trajectory, candidateSet, groundTruth):
        u = self.traj2u[trajectory]
        long = self.longs[u]
        rec = self.recents[trajectory]
        path = './output/LLMMove/{}/{}'.format(self.datasetName, trajectory)
        if os.path.exists(path):
            with open(path, 'r') as file:
                response = file.read()
                res_content = eval(response)
                prediction = res_content["response"]["recommendation"]
        else:
            output = dict()
            mostrec = rec[-1][0]
            longterm = [(poi, self.poiInfos[poi]["category"]) for poi, _ in long]
            longterm = longterm[-40:]

            recent = [(poi, self.poiInfos[poi]["category"]) for poi, _ in rec]
            recent = recent[-5:]
            
            candidates = [(poi, haversine_distance(self.poiInfos[poi]["latitude"], self.poiInfos[poi]["longitude"], self.poiInfos[mostrec]["latitude"], self.poiInfos[mostrec]["longitude"]), self.poiInfos[poi]["category"]) for poi in candidateSet]
            candidates.sort(key=lambda x:x[1])

            prompt = f"""\
<long-term check-ins> [Format: (POIID, Category)]: {longterm}
<recent check-ins> [Format: (POIID, Category)]: {recent}
<candidate set> [Format: (POIID, Distance, Category)]: {candidates}
Your task is to recommend a user's next point-of-interest (POI) from <candidate set> based on his/her trajectory information.
The trajectory information is made of a sequence of the user's <long-term check-ins> and a sequence of the user's <recent check-ins> in chronological order.
Now I explain the elements in the format. "POIID" refers to the unique id of the POI, "Distance" indicates the distance (kilometers) between the user and the POI, and "Category" shows the semantic information of the POI.

Requirements:
1. Consider the long-term check-ins to extract users' long-term preferences since people tend to revisit their frequent visits.
2. Consider the recent check-ins to extract users' current perferences.
3. Consider the "Distance" since people tend to visit nearby pois.
4. Consider which "Category" the user would go next for long-term check-ins indicates sequential transitions the user prefer.

Please organize your answer in a JSON object containing following keys:
"recommendation" (10 distinct POIIDs of the ten most probable places in <candidate set> in descending order of probability), and "reason" (a concise explanation that supports your recommendation according to the requirements). Do not include line breaks in your output.
"""
            output["prompt"] = prompt
            messages = [{"role": "user", "content": prompt}]
            response = openaiAPIcall(
                model = 'gpt-3.5-turbo',
                messages=messages,
                temperature=0,
            )
            res_content = response.choices[0].message.content
            res_content = eval(res_content)
            output["response"] = res_content
            prediction = res_content["recommendation"]
            output["groundTruth"] = groundTruth[0]
            self.outputResponse(output, trajectory)
        return prediction

    def outputResponse(self, response, trajectory):
        path = './output/LLMMove/{}/{}'.format(self.datasetName, trajectory)
        with open(path, 'w') as file:
            file.write(json.dumps(response, indent='\t'))