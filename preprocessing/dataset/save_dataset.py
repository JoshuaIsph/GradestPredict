
# preprocessing/dataset/save_dataset.py
import csv

def save_dataset_csv(dataset, filename="climb_dataset.csv"):
    # Detect all columns from dataset
    if not dataset:
        print("Dataset is empty!")
        return

    fieldnames = list(dataset[0].keys())
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in dataset:
            writer.writerow(row)
    print(f"Dataset saved to {filename}")
