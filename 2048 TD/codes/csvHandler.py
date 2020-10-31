import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

csvPath = '../csv/'
csvFiles = []
for root, dirs, files in os.walk(csvPath, topdown=False):
    csvFiles = files
    
avg = []
high = []

csvFiles = list(map(lambda x : int(x.split('.')[0]), csvFiles))
csvFiles.sort()
csvFiles = list(map(lambda x : str(x)+'.csv', csvFiles))

for f in csvFiles:
    df = pd.read_csv(csvPath + f, skipinitialspace=True)
    avg.append(df['average'][0])
    high.append(df['highest'][0])
    
print(avg)
print(high)


fig = plt.figure(figsize=(5,4))
plt.plot(range(len(avg)), avg)
plt.xlabel('Episode (x1000)'), plt.ylabel('Average Score'), plt.title('Average Score')
plt.savefig('../results/Average.png'), plt.show()

fig = plt.figure(figsize=(5,4))
plt.plot(range(len(high)), high)
plt.xlabel('Episode (x1000)'), plt.ylabel('Highest Score'), plt.title('Highest Score')
plt.savefig('../results/Highest.png'), plt.show()