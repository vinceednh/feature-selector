import numpy as np
import time

def load_data(filename):
    data = []

    with open(filename, 'r') as fHandle:
        for line in fHandle:
            num_as_strings = line.split()

            row = []
            for i in num_as_strings:
                row.append(float(i))
            
            data.append(row)
        
        data = np.array(data)

        labels = []
        for i in range(len(data)):
            labels.append(int(data[i][0]))

        features = data[:, 1:]

        return features, labels
    
def main():
    print ("Welcome to the Feature Selector!")
    filename = input("Type in the name of the file to test: ").strip()

    print("\n Type in the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination \n")
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

    print(f"This dataset has {n_features} features (not including the class attribute), with {n_instances} instances.")

if __name__ == "__main__":
    main()