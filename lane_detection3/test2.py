a = 1
b = 1
list0 = []

if a or b:
    if a:
        list0.append('a')
    if b:
        list0.append('b')
else:
    print('not appending')

print(list0)