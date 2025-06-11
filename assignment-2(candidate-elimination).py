import numpy as np
import pandas as pd

# Load dataset
data = pd.read_csv('/content/drive/MyDrive/Day-1(7th april)/enjoysport.csv')
concepts = np.array(data.iloc[:, :-1])
target = np.array(data.iloc[:, -1])
print("Concepts:\n", concepts)
print("Target:\n", target)

def learn(concepts, target):
    specific_h = concepts[0].copy()
    general_h = [["?" for _ in range(len(specific_h))] for _ in range(len(specific_h))]

    print("\nInitialization of specific_h and general_h")
    print("Specific_h:", specific_h)
    print("General_h:", general_h)

    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
            print("\nInstance {} is Positive.".format(i + 1))
            print("Specific_h:", specific_h)
        elif target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'
            print("\nInstance {} is Negative.".format(i + 1))
            print("General_h:", general_h)

    # Remove overly general hypotheses
    general_h = [h for h in general_h if h != ['?' for _ in range(len(specific_h))]]

    return specific_h, general_h

s_final, g_final = learn(concepts, target)

print("\nFinal Specific_h:", s_final)
print("Final General_h:", g_final)
