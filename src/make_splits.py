import pandas as pd
import os

os.makedirs(r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\splits", exist_ok=True)
df = pd.read_csv(r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\meta\esc50.csv")

# Fold 1 is Validation, Folds 2,3,4,5 are Train
val_df = df[df['fold'] == 1]
train_df = df[df['fold'] != 1]

train_df.to_csv(r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\splits\train_fold1.csv", index=False)
val_df.to_csv(r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\splits\val_fold1.csv", index=False)

print(f"Saved Splits! Train size: {len(train_df)} | Val size: {len(val_df)}")