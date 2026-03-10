import numpy as np
import math
import time

def load_data(filename):
    # Reads in the data from the file and returns the features and labels as separate lists
    data = []

    with open(filename, 'r') as fHandle:
        for line in fHandle:
            num_as_strings = line.split()

            row = []
            for i in num_as_strings:
                row.append(float(i))
            
            data.append(row)
        
    data = np.array(data)

    # The first column will represent the labels
    labels = []
    for i in range(len(data)):
        labels.append(int(data[i][0]))

    # Everything else after the first column will represent the features
    features = data[:, 1:]

    return features, labels

# Calculates the distance between the object and the neighbor using the features in the feature set
def euclidean_distance(object_to_classify, neighbor, feature_set):
    total = 0
    for f in feature_set:
        total += (object_to_classify[f] - neighbor[f]) ** 2
    return math.sqrt(total)

def nearest_neighbor(features, labels, feature_set):
    number_correctly_classified = 0

    for i in range(len(features)):
        object_to_classify = features[i]
        label_object_to_classify = labels[i]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')

        # "Leave one out" method: Leaving out the object to classify and finding the nearest neighbor from the rest of the objects
        for k in range(len(features)):
            if k != i:
                distance = euclidean_distance(object_to_classify, features[k], feature_set)

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = labels[nearest_neighbor_location]
        
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1

    # Returns the accuracy as a percentage between 0 and 1
    return number_correctly_classified / len(features)

def main():
    print ("Welcome to the Feature Selector!")
    filename = input("Type in the name of the file to test: ").strip()

    print("\nType in the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination\n")
    choice = input("Enter your choice: ").strip()

    if choice == "1":
        print("You selected Forward Selection.")
    elif choice == "2":
        print("You selected Backward Elimination.")
    else:
        print("Invalid choice. Please type 1 or 2.")
        return

    features, labels = load_data(filename)

    n_instances = len(features)
    n_features = len(features[0])

    print(f"\nThis dataset has {n_features} features (not including the class attribute), with {n_instances} instances.")

if __name__ == "__main__":
    main()