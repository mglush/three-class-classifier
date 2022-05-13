# hw2.py for UCSB CS165B Spring 2022.
# Michael Glushchenko, 9403890

import numpy as np

def run_train_test(training_input, testing_input):
    # extract training values from the input and split training data into 3 classes.
    [d, n1, n2, n3] = training_input[0]
    [a_class, b_class, c_class] = [training_input[1:1+n1], training_input[1+n1:1+2*n1], training_input[1+2*n1:]]

    # calculating centroids of each class.
    a_centroid = [sum(i)/len(i) for i in zip(*a_class)]
    b_centroid = [sum(i)/len(i) for i in zip(*b_class)]
    c_centroid = [sum(i)/len(i) for i in zip(*c_class)]

    # we use lists for each variable because d can be any number.
    [a_to_b, b_to_c, a_to_c] = [[], [], []]
    [a_to_b_mid, b_to_c_mid, a_to_c_mid] = [[], [], []]

    # calculating the discriminant function lines.
    for i in range(d):
        a_to_b.append(a_centroid[i] - b_centroid[i])
        b_to_c.append(b_centroid[i] - c_centroid[i])
        a_to_c.append(a_centroid[i] - c_centroid[i])
        a_to_b_mid.append((a_centroid[i] + b_centroid[i]) / 2)
        b_to_c_mid.append((b_centroid[i] + c_centroid[i]) / 2)
        a_to_c_mid.append((a_centroid[i] + c_centroid[i]) / 2)

    # calculating the threshold values between each class.
    [a_to_b_threshold, b_to_c_threshold, a_to_c_threshold] = [np.dot(a_to_b, a_to_b_mid), np.dot(b_to_c, b_to_c_mid), np.dot(a_to_c, a_to_c_mid)]

    # start processing testing_input.
    result = []
    [d, n1, n2, n3] = testing_input[0]
    testing_input = testing_input[1:]

    for i in testing_input:
        if np.dot(i, a_to_b) >= a_to_b_threshold:
            if np.dot(i, a_to_c) >= a_to_c_threshold:
                result.append('A')
            else: result.append('C')
        else:
            if np.dot(i, b_to_c) >= b_to_c_threshold:
                result.append('B')
            else: result.append('C')

    [a_res, b_res, c_res] = [result[0:n1], result[n1:2*n1], result[2*n1:]]
    
    # creating the confusion matrix.
    [a_table, b_table, c_table] = [[0,0,0,0], [0,0,0,0], [0,0,0,0]]
    for i in a_res:
        if i == 'A':
            a_table[0] += 1
            b_table[3] += 1
            c_table[3] += 1
        if i == 'B':
            a_table[1] += 1
            b_table[2] += 1
            c_table[3] += 1
        if i == 'C':
            a_table[1] += 1
            b_table[3] += 1
            c_table[2] += 1

    for i in b_res:
        if i == 'A':
            b_table[1] += 1
            a_table[2] += 1
            c_table[3] += 1
        if i == 'B':
            b_table[0] += 1
            a_table[3] += 1
            c_table[3] += 1
        if i == 'C':
            b_table[1] += 1
            a_table[3] += 1
            c_table[2] += 1

    for i in c_res:
        if i == 'A':
            c_table[1] += 1
            a_table[2] += 1
            b_table[3] += 1
        if i == 'B':
            c_table[1] += 1
            a_table[3] += 1
            b_table[2] += 1
        if i == 'C':
            c_table[0] += 1
            a_table[3] += 1
            b_table[3] += 1

    # calculating the sought after results (tpr, fpr, etc.).
    tpr = (a_table[0] + b_table[0] + c_table[0]) / (n1) / 3
    fpr = (a_table[2] + b_table[2] + c_table[2]) / (n1 * 2) / 3
    error_rate = (a_table[1] + a_table[2] + b_table[1] + b_table[2] + c_table[1] + c_table[2]) /  (n1 * 3) / 3
    accuracy = (a_table[0] + a_table[3] + b_table[0] + b_table[3] + c_table[0] + c_table[3]) /  (n1 * 3) / 3
    precision = (a_table[0]/(a_table[0] + a_table[2]) + b_table[0]/(b_table[0] + b_table[2]) + c_table[0]/(c_table[0] + c_table[2])) / 3


    return  {
                "tpr": float('{:.2f}'.format(tpr)),
                "fpr": float('{:.2f}'.format(fpr)),
                "error_rate": float('{:.2f}'.format(error_rate)),
                "accuracy": float('{:.2f}'.format(accuracy)),
                "precision": float('{:.2f}'.format(precision))
            }

#######
# The following functions are provided for you to test your classifier.
######
def parse_file(filename):
    with open(filename, "r") as f:
        data = [[float(y) for y in x.strip().split(" ")] for x in f]
        data[0] = [int(x) for x in data[0]]

        return data

if __name__ == "__main__":
    import sys

    training_input = parse_file(sys.argv[1])
    testing_input = parse_file(sys.argv[2])

    run_train_test(training_input, testing_input)

