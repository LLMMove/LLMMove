import argparse
import os

case_num = YOUR_CASE_NUMBER

def readTrain(filePath):
    longs = dict()
    pois = dict()
    with open(filePath, 'r') as file:
        lines = file.readlines()
    for line in lines[1:]:
        data = line.split(',')
        time, u, lati, longi, i, category = data[1], data[5], data[6], data[7], data[8], data[10]
        if i not in pois:
            pois[i] = {"latitude": lati, "longitude": longi, "category": category}
        if u not in longs:
            longs[u] = list()
        longs[u].append((i, time))
    return longs, pois

def readTest(filePath):
    recents = dict()
    pois = dict()
    targets = dict()
    traj2u = dict()
    with open(filePath, 'r') as file:
        lines = file.readlines()
    for line in lines[1:]:
        data = line.split(',')
        time, trajectory, u, lati, longi, i, category = data[1], data[3],data[5], data[6], data[7], data[8], data[10]
        if i not in pois:
            pois[i] = dict()
            pois[i]["latitude"] = lati
            pois[i]["longitude"] = longi
            pois[i]["category"] = category
        if trajectory not in traj2u:
            traj2u[trajectory] = u
        if trajectory not in recents:
            recents[trajectory] = list()
            recents[trajectory].append((i, time))
        else:
            if trajectory in targets:
                recents[trajectory].append(targets[trajectory])
            targets[trajectory] = (i, time)
    return recents, pois, targets, traj2u

def getData(datasetName):
    if datasetName == 'nyc':
        filePath = './data/nyc/{}_sample.csv'
    elif datasetName == 'tky':
        filePath = './data/tky/{}_sample.csv'
    else:
        raise NotImplementedError
    trainPath = filePath.format('train')
    testPath = filePath.format('test')

    longs, poiInfos = readTrain(trainPath)
    recents, testPoi, targets, traj2u = readTest(testPath)
    poiInfos.update(testPoi)

    targets = dict(list(targets.items())[:case_num])

    return longs, recents, targets, poiInfos, traj2u

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, help='The model to be run.', required=True)
    parser.add_argument('-d', '--datasetName', type=str, choices=['nyc', 'tky'], default='nyc',help='nyc/tky')
    args = parser.parse_args()

    data = getData(args.datasetName)
    path = './output/{}/{}'.format(args.model, args.datasetName)
    if not os.path.exists(path):
        os.makedirs(path)

    if args.model == 'LLMMove':
        from models.LLMMove import LLMMove
        model = LLMMove()
    else:
        raise NotImplementedError

    results = model.run(data, args.datasetName)
    results = 'ACC@1: {}, ACC@10: {}, MRR: {}'.format(results[0], results[1], results[2])
    resultPath = './results/{}_{}'.format(args.model, args.datasetName)
    with open(resultPath, 'w') as file:
        file.write(results)