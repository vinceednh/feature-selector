import numpy as np
import time

def load_data(filename):
    data = []

    with open(filename, 'r') as fHandle:
        for line in fHandle:
            if line.strip() == '':
                continue

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

    features, labels = load_data(filename)

    n_instances = len(features)
    n_features = len(features[0])

    print(f"This dataset has {n_features} features (not including the class attribute), with {n_instances} instances.")

if __name__ == "__main__":
    main()