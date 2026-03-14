import numpy as np
import math
import matplotlib.pyplot as plt
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
    diff = object_to_classify[feature_set] - neighbor[feature_set]
    return np.sqrt(np.sum(diff * diff))

def nearest_neighbor(features, labels, feature_set, best_so_far_accuracy = 0):    
    number_correctly_classified = 0
    n_instances = len(features)

    for i in range(len(features)):
        object_to_classify = features[i]
        label_object_to_classify = labels[i]

        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = None
        nearest_neighbor_label = None

        # "Leave one out" method: Leaving out the object to classify and finding the nearest neighbor from the rest of the objects
        for k in range(len(features)):
            if k != i:
                neighbor = features[k]
                label_neighbor = labels[k]
                distance = euclidean_distance(object_to_classify, neighbor, feature_set)

                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = label_neighbor
        
        if label_object_to_classify == nearest_neighbor_label:
            number_correctly_classified += 1

        # Early abandoning method from P2_hints
        instances_remaining = n_instances - i - 1
        max_possible_accuracy = (number_correctly_classified + instances_remaining) / n_instances
        if max_possible_accuracy < best_so_far_accuracy:
            return -1

    # Returns the accuracy as a percentage between 0 and 1
    return number_correctly_classified / n_instances

def forward_selection(features, labels, n_features):
    # Starts with empty sets of features and adds the best feature at each level
    current_set_of_features = []
    best_overall_accuracy = 0
    best_overall_features = []
    graph_labels = ["{}"]
    graph_accuracies = [0]

    for i in range(n_features):
        feature_to_add_at_this_level = None
        level_best_accuracy = 0

        # Adding each feature that is not already in the current set of features and testing the accuracy of the new set of features
        for k in range(n_features):
            if k not in current_set_of_features:
                considered_set = current_set_of_features + [k]
                accuracy = nearest_neighbor(features, labels, considered_set, level_best_accuracy)

                if accuracy == -1:
                    continue

                # For display purposes, we add 1 to each feature number since the features are at index 0 but features start at 1
                display_considered = []
                for f in considered_set:
                    display_considered.append(f + 1)

                print(f"Using feature(s) {display_considered} accuracy is {round(accuracy * 100, 1)}%")

                if accuracy > level_best_accuracy:
                    level_best_accuracy = accuracy
                    feature_to_add_at_this_level = k
            
        # Adds the best feature found at this level to the current set of features
        current_set_of_features.append(feature_to_add_at_this_level)

        display_current = []
        for f in current_set_of_features:
            display_current.append(f + 1)

        graph_labels.append(str(display_current))
        graph_accuracies.append(round(level_best_accuracy * 100, 1))

        # Tracks the best overall in case the accuracy drops later
        if level_best_accuracy > best_overall_accuracy:
            best_overall_accuracy = level_best_accuracy
            best_overall_features = list(current_set_of_features)
        else:
            print(f"\n(Warning: Accuracy has decreased! Continuing search in case of local maxima)")
        
        print(f"Feature set {display_current} was best, accuracy is {round(level_best_accuracy * 100, 1)}%\n")

    display_best = []
    for f in best_overall_features:
        display_best.append(f + 1)
    
    print(f"Finished search!! The best feature subset is {display_best}, which has an accuracy of {round(best_overall_accuracy * 100, 1)}%")
    
    return best_overall_features, best_overall_accuracy, graph_labels, graph_accuracies

def backward_elimination(features, labels, n_features):
    # Starts with all the features and removes the worst features at each level
    current_set_of_features = list(range(n_features))
    best_overall_accuracy = 0
    best_overall_features = []
    graph_labels = []
    graph_accuracies = []
    
    # n_features - 1 will make sure I will stop and prevent checking the empty set
    for i in range(n_features - 1):
        feature_to_remove_at_this_level = None
        level_best_accuracy = 0

        # Testing the accuracy of the current set
        for k in range(n_features):
            if k in current_set_of_features:
                considered_set = []
                for f in current_set_of_features:
                    if f != k:
                        considered_set.append(f)
                
                accuracy = nearest_neighbor(features, labels, considered_set, level_best_accuracy)

                if accuracy == -1:
                    continue

                display_considered = []
                for f in considered_set:
                    display_considered.append(f + 1)

                print(f"Using feature(s) {display_considered} accuracy is {round(accuracy * 100, 1)}%")

                if accuracy > level_best_accuracy:
                    level_best_accuracy = accuracy
                    # Tracking which to remove, instead of adding features
                    feature_to_remove_at_this_level = k
        
        # Removal
        current_set_of_features.remove(feature_to_remove_at_this_level)

        display_current = []
        for f in current_set_of_features:
            display_current.append(f + 1)
        
        graph_labels.append(str(display_current))
        graph_accuracies.append(round(level_best_accuracy * 100, 1))

        if level_best_accuracy > best_overall_accuracy:
            best_overall_accuracy = level_best_accuracy
            best_overall_features = list(current_set_of_features)
        else:
            print(f"\n(Warning: Accuracy has decreased! Continuing search in case of local maxima)")

        print(f"Feature set {display_current} was best, accuracy is {round(level_best_accuracy * 100, 1)}%\n")

    display_best = []
    for f in best_overall_features:
        display_best.append(f + 1)
    
    print(f"Finished search!! The best feature subset is {display_best}, which has an accuracy of {round(best_overall_accuracy * 100, 1)}%")

    return best_overall_features, best_overall_accuracy, graph_labels, graph_accuracies

    
# Graph plotting without needing to manually create graph
def plot_graph(graph_labels, graph_accuracies, title):
    plt.figure(figsize = (30, 6))
    plt.bar(graph_labels, graph_accuracies, color = 'gray')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.title(title)
    plt.xticks(rotation = 45, ha = 'right')
    plt.show()


def main():
    print ("Welcome to the Feature Selector!")
    filename = input("Type in the name of the file to test: ").strip()

    print("\nType in the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination\n")
    choice = input("Enter your choice: ").strip()

    features, labels = load_data(filename)

    n_instances = len(features)
    n_features = len(features[0])

    print(f"\nThis dataset has {n_features} features (not including the class attribute), with {n_instances} instances.")

    all_features = list(range(n_features))
    initial_accuracy = nearest_neighbor(features, labels, all_features)
    print(f"Running nearest neighbor with all {n_features} features, using \"leaving-one-out\" evaluation, I get an accuracy of {round(initial_accuracy * 100, 1)}%")
    print("Beginning search.\n")

    # Recording the time to add into the report
    if choice == "1":
        start_time = time.time()
        best_features, best_accuracy, graph_labels, graph_accuracies = forward_selection(features, labels, n_features)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Forward Selection took {round(total_time, 2)} seconds to run.")
        plot_graph(graph_labels, graph_accuracies, "Forward Selection")
    elif choice == "2":
        start_time = time.time()
        best_features, best_accuracy, graph_labels, graph_accuracies = backward_elimination(features, labels, n_features)
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Backward Elimination took {round(total_time, 2)} seconds to run.")
        plot_graph(graph_labels, graph_accuracies, "Backward Elimination")
    else:
        print("Invalid choice. Please type 1 or 2.")
        return

if __name__ == "__main__":
    main()  