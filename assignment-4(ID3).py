import math
import csv

# Load CSV file
def load_csv(filename):
    with open(filename, "r") as f:
        lines = csv.reader(f)
        dataset = list(lines)
    headers = dataset.pop(0)
    return dataset, headers

# Node class for the decision tree
class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ""

# Function to split dataset based on attribute
def subtables(data, col, delete):
    dic = {}
    coldata = [row[col] for row in data]
    attr = list(set(coldata))
    counts = [0] * len(attr)
    r = len(data)
    c = len(data[0])

    for x in range(len(attr)):
        for y in range(r):
            if data[y][col] == attr[x]:
                counts[x] += 1

    for x in range(len(attr)):
        dic[attr[x]] = [[0 for _ in range(c)] for _ in range(counts[x])]
        pos = 0
        for y in range(r):
            if data[y][col] == attr[x]:
                reduced_row = data[y][:col] + data[y][col+1:] if delete else data[y]
                dic[attr[x]][pos] = reduced_row
                pos += 1
    return attr, dic

# Calculate entropy
def entropy(S):
    attr = list(set(S))
    if len(attr) == 1:
        return 0
    counts = [S.count(val) / len(S) for val in attr]
    return sum([-p * math.log(p, 2) for p in counts if p != 0])

# Compute information gain for an attribute
def compute_gain(data, col):
    attr, dic = subtables(data, col, delete=False)
    total_entropy = entropy([row[-1] for row in data])
    for x in attr:
        ratio = len(dic[x]) / len(data)
        total_entropy -= ratio * entropy([row[-1] for row in dic[x]])
    return total_entropy

# Build decision tree using ID3
def build_tree(data, features):
    lastcol = [row[-1] for row in data]
    if len(set(lastcol)) == 1:
        leaf = Node("")
        leaf.answer = lastcol[0]
        return leaf

    n = len(data[0]) - 1
    gains = [compute_gain(data, col) for col in range(n)]
    split = gains.index(max(gains))
    node = Node(features[split])
    fea = features[:split] + features[split+1:]

    attr, dic = subtables(data, split, delete=True)
    for x in attr:
        child = build_tree(dic[x], fea)
        node.children.append((x, child))
    return node

# Print the decision tree
def print_tree(node, level=0):
    if node.answer != "":
        print("  " * level + "->", node.answer)
        return
    print("  " * level + f"[{node.attribute}]")
    for value, child in node.children:
        print("  " * (level+1) + f"({value})")
        print_tree(child, level + 2)

# Classify a test instance
def classify(node, x_test, features):
    if node.answer != "":
        return node.answer
    pos = features.index(node.attribute)
    for value, child in node.children:
        if pos >= len(x_test):
            return "Invalid input"
        if x_test[pos] == value:
            return classify(child, x_test, features)
    return "Unknown"

# Mount Google Drive (if not already mounted)
from google.colab import drive
drive.mount('/content/drive')

# Load training data
train_path = '/content/drive/MyDrive/Day-1(7th april)/tennisdata.csv'
train_data, features = load_csv(train_path)

# Build tree
tree = build_tree(train_data, features)

# Display the tree
print("Decision Tree built using ID3:\n")
print_tree(tree)

# Optional: Create clean test CSV file
test_csv = """Outlook,Temperature,Humidity,Wind
Rain,Cool,Normal,Strong
Sunny,Mild,Normal,Strong
"""
with open('/content/drive/MyDrive/Day-1(7th april)/tennisdata_test.csv', 'w') as f:
    f.write(test_csv)

# Load test data
test_path = '/content/drive/MyDrive/Day-1(7th april)/tennisdata_test.csv'
test_data, test_features = load_csv(test_path)

# Debug: Print test data shape
print("\n✅ Test Data Loaded:")
print("Test Features:", test_features)
for i, inst in enumerate(test_data):
    print(f"{i}: {inst} (len={len(inst)})")

# Predict and print results
print("\n🔍 Predictions on Test Data:\n")
for instance in test_data:
    if len(instance) != len(test_features):
        print(f"⚠️ Skipping malformed instance: {instance}")
        continue
    print(f"Instance: {instance}")
    label = classify(tree, instance, test_features)
    print(f"Predicted Label: {label}\n")
