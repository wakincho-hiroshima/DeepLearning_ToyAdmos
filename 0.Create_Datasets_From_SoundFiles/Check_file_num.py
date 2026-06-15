import os

ch_num = "1"

file_types = ["test_anomaly", "test_normal", "train_normal"]

# location of the dataset
DATASET_DIR = f"/run/media/wakincho/SSPQ-USC/Projects/Datasets/toy_admos_project/Training_SoundFiles/ToyCar/ch{ch_num}/"




for file_types in file_types:
    target_path=os.path.join(DATASET_DIR, file_types)
    
    if os.path.exists(target_path):
        items = [f for f in os.listdir(target_path) if os.path.isfile(os.path.join(target_path, f))]
        file_count = len(items)
        print(f"Dataset found at: {target_path}")
        print(f"Total number of files: {file_count}")
        
    else:
        print(f"Dataset not found at: {target_path}")
        print("Please check the path and ensure the dataset is properly mounted.")
        exit(1)