import sys
import math
import numpy as np

data_files = sys.argv[1]
f = open(data_files,'r')
test_data = []
x = f.readline()

while(x!=''):
    a = x.split()
    y = []
    for j in range(0, len(a), 1):
        y.append(float(a[j]))
    test_data.append(y)
    x = f.readline()

row_count = len(test_data)
column_count = len(test_data[0])

#print(column_count)
#print(row_count)
#print(test_data[0])

def info_dataset(data, verbose=True):
    label1, label2 = 0, 0
    data_size = len(data)
    for datum in data:
        if datum[-1] == 1:
            label1 += 1
        else:
            label2 += 1
    if verbose:
        print('Total No. of samples: %d' % data_size)
        print('Total label 1: %d' % label1)
        print('Total label 0: %d' % label2)
    return [len(test_data), label1, label2]

print("Data File Imported successfully !!")
info_dataset(test_data)

p = 0.6
_, label1, label2 = info_dataset(test_data,False)

train_set, test_set = [], []
max_label1, max_label2 = int(p * label1), int(p * label2)
total_label1, total_label2 = 0, 0
for sample in test_data:
    if (total_label1 + total_label2) < (max_label1 + max_label2):
        train_set.append(sample)
        if sample[-1] == 1 and total_label1 < max_label1:
            total_label1 += 1
        else:
            total_label2 += 1
    else:
        test_set.append(sample)

print("Train and Test Data split done with 60:40 ratio")
print("Train Set details: ")
info_dataset(train_set)

print("Test Set details: ")
info_dataset(test_set)

train_set_a = np.asarray(train_set)
test_set_a = np.asarray(test_set)


def euclidian_dist(p1, p2):
    dim, sum_ = len(p1), 0
    for index in range(dim - 1):
        sum_ += math.pow(p1[index] - p2[index], 2)
    return math.sqrt(sum_)

def knn(train_set, new_sample, K):
    dists, train_size = {}, len(train_set)
    
    for i in range(train_size):
        d = euclidian_dist(train_set[i], new_sample)
        dists[i] = d
    
    k_neighbors = sorted(dists, key=dists.get)[:K]
    
    qty_label1, qty_label2 = 0, 0
    for index in k_neighbors:
        if train_set[index][-1] == 1:
            qty_label1 += 1
        else:
            qty_label2 += 1
            
    if qty_label1 < qty_label2:
        return 0
    else:
        return 1



print("Predicting the labels using kNN algorithm using k = 7")
correct, k = 0 ,7
for sample in test_set_a:
	label = knn(train_set_a, sample, k)
	#print (label, sample[-1])
	if sample[-1] == label:
		correct += 1
acc = 100 * (correct/len(test_set))


print("Train set size: %d" % len(train_set))
print("Test set size: %d" % len(test_set))
print("Correct predicitons: %d" % correct)
print('Accuracy: %.3f' % acc)
