import numpy as np
from scipy.stats import kendalltau, pearsonr
from glob import glob

f1 = 'test/allscore.txt'
f2 = 'test/pred.txt'


loss1, loss2 = [], []
with open(f1, 'r') as f_1, open(f2, 'r') as f_2:
    for ls in zip(f_1, f_2):
        loss1.append(float(ls[0][:-1].split(',')[-1]))
        loss2.append(float(ls[1].split()[-1]))

loss1 = np.array(loss1).flatten()[:]
loss2 = np.array(loss2).flatten()[:]
print(f1, '-----', f2)
print(kendalltau(loss1, loss2)[0])
print(pearsonr(loss1, loss2)[0])
