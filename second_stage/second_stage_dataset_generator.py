import json
import csv
import random
from numpy import genfromtxt

def writeUtil(datalist):
    util = []
    for i in range(len(datalist)):
        if (datalist[i][0] > 800):
            util.append(random.randrange(75, 100) / 100)
        elif (datalist[i][0] > 600):
            if (datalist[i][1] > 0.001):
                util.append(random.randrange(50, 100) / 100)
            else:
                util.append(random.randrange(30, 60) / 100)
        elif (datalist[i][0] > 400):
            if (datalist[i][1] > 0.001):
                util.append(random.randrange(50, 75) / 100)
            else:
                util.append(random.randrange(30, 50) / 100)
        elif (datalist[i][0] > 200):
            if (datalist[i][1] > 0.01):
                util.append(random.randrange(50, 75) / 100)
            else:
                util.append(random.randrange(20, 50) / 100)
        else:
            if (datalist[i][1] > 0.01):
                util.append(random.randrange(50, 75) / 100)
            else:
                util.append(random.randrange(0, 25) / 100)
    return util
def labelData(utildata):
    label = []
    utildatacompiled = []
    for i in range(len(utildata[0])):
        utildatacompiled.append([utildata[0][i], utildata[1][i], utildata[2][i], utildata[3][i]])
    for i in range(len(utildata[0])):
        labels_encoded = [0,0,0,0]
        labels_encoded[utildatacompiled[i].index(min(utildatacompiled[i]))] = 1
        label.append(labels_encoded)
    return label

#grab raw data
d0 = genfromtxt('networkdataraw.csv', delimiter=',', skip_footer = 0, skip_header = 10000, encoding="utf-8-sig", dtype='float32').tolist()
d1 = genfromtxt('networkdataraw.csv', delimiter=',', skip_footer = 0, skip_header = 10000, encoding="utf-8-sig", dtype='float32').tolist()
d2 = genfromtxt('networkdataraw.csv', delimiter=',', skip_footer = 0, skip_header = 10000, encoding="utf-8-sig", dtype='float32').tolist()
d3 = genfromtxt('networkdataraw.csv', delimiter=',', skip_footer = 0, skip_header = 10000, encoding="utf-8-sig", dtype='float32').tolist()
print(type(d0))
print(d0)
print(len(d0))
random.shuffle(d0)
random.shuffle(d1)
random.shuffle(d2)
random.shuffle(d3)

u0 = writeUtil(d0)
u1 = writeUtil(d1)
u2 = writeUtil(d2)
u3 = writeUtil(d3)

utillist = [u0, u1, u2, u3]
labels = labelData(utillist)
print(len(labels))

with open('networkvalidationdata.csv', 'w', ) as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
    for i in range(len(d0)):
        wr.writerow([d0[i][0], d1[i][0], d2[i][0], d3[i][0], d0[i][1], d1[i][1], d2[i][1], d3[i][1], d0[i][2], d1[i][2], d2[i][2], d3[i][2], u0[i], u1[i], u2[i], u3[i], labels[i][0], labels[i][1], labels[i][2], labels[i][3]])