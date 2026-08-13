import os

dataset_path = "dataset/archive2"

count = 0

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".wav"):
            count += 1

print("Total Audio Files:", count)