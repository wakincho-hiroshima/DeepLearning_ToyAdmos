import os

file_type = "test_anomaly"
#file_type = "test_normal" 
#file_type = "train_normal"

# location of the dataset
DATASET_DIR = f"/run/media/wakincho/SSPQ-USC/Projects/0.Datasets/toy_admos_project/C01_create_small_INT_dataset/exp1_dataset_ToyCar/{file_type}/"

if os.path.exists(DATASET_DIR):
    items = [f for f in os.listdir(DATASET_DIR) if os.path.isfile(os.path.join(DATASET_DIR, f))]
    file_count = len(items)
    
    print(f"Dataset found at: {DATASET_DIR}")
    print(f"Total number of files: {file_count}")
  
else:
    print(f"Dataset not found at: {DATASET_DIR}")
    print("Please check the path and ensure the dataset is properly mounted.")
    exit(1)