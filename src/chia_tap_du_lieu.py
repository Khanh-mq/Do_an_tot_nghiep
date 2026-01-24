import os
import shutil

source_dir = "data/wav_en"
out_dir = "data/source"

train_ratio = 0.8
valid_ratio = 0.1

os.makedirs(out_dir + "/train", exist_ok=True)
os.makedirs(out_dir + "/valid", exist_ok=True)
os.makedirs(out_dir + "/test", exist_ok=True)

files = sorted([f for f in os.listdir(source_dir) if f.endswith(".wav")])
n = len(files)
print(f'so luong n =  {n}')
n_train = int(n * train_ratio)
n_valid = int(n * valid_ratio)

train_files = files[:n_train]
valid_files = files[n_train:n_train + n_valid]
test_files  = files[n_train + n_valid:]

def copy(files, split):
    for f in files:
        shutil.copy(
            os.path.join(source_dir, f),
            os.path.join(out_dir, split, f)
        )

copy(train_files, "train")
copy(valid_files, "valid")
copy(test_files, "test")

print(f"Train: {len(train_files)}")
print(f"Valid: {len(valid_files)}")
print(f"Test : {len(test_files)}")
