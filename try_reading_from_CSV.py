##reading from a CSV file
import os
import csv
l = [None]
with open('train_landmark_csv.csv','r') as f_csv:
	spamreader = csv.reader(f_csv, delimiter=',')
	for row in spamreader:
		l.append(row)


occurences = []

for line in l:
        if(line):
                occurences.append(line[1])

print(occurences)
emotions =  ['happy', 'disappointed', 'passive', 'curious']

for e in emotions:
        
        print('e' + occurences.count(e))



