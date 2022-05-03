import pandas as pd
import numpy as np

df = pd.read_csv(r'../lane_detection2/Arrays/histogram')

midpoint = 640

left_indices = df.iloc[640:, 0]
left_indices = left_indices[left_indices>0]
left = left_indices.index[len(left_indices)//2]

right_indices = df.iloc[:640, 0]
right_indices = right_indices[right_indices>0]
right = right_indices.index[len(right_indices)//2]

a = np.array([ 0,  3,  5,  5,  0, 10, 14, 15, 56,  0,  0,  0, 12, 23, 45, 23, 12,
       45,  0,  1,  0,  2,  3,  4,  0,  0,  0])

# idx = np.where(a!=0)[0]
# print(idx)
# print(a[idx])
#
# out = np.split(a, [3, 5, 5, 10, 14, 15, 56])
# print(out)

idx = np.where(a!=0)[0]
print(a[idx])
print(np.where(np.diff(idx)!=1)[0]+1)
nonzero = np.split(a[idx], np.where(np.diff(idx)!=1)[0]+1)

left = nonzero[0]
right = nonzero[-1]
print(left, right)

a = np.array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0])

numbers = np.array([1222, 232, 2, 232342, 22])
n = 2

idx = np.argpartition(numbers, -n)[-n:]

print(idx)

# key = []
# for idx, val in enumerate(a):
#     print(idx)
#     # if val==0 or a[idx-1]==0:
#     #     key.append()
#
# idx = np.where(a!=0)[0]
# key = np.where(np.diff(idx)!=1)[0]+1
# # out = np.split(a[idx],)
#
# print(key)