import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\TRIVIKRAM\Python VSCode\SRF\data\raw\ESC-50-master\meta\esc50.csv")

# 1. Verify Class Balance
class_counts = df['category'].value_counts()
print("Class Balance Verification:")
print(f"Min samples per class: {class_counts.min()} | Max: {class_counts.max()}")

# 2. Plot Distribution Histogram
plt.figure(figsize=(12, 6))
class_counts.plot(kind='bar')
plt.title("ESC-50 Class Distribution (Should be perfectly flat)")
plt.xlabel("Classes")
plt.ylabel("Number of Audio Clips")
plt.tight_layout()
plt.savefig(r"C:\Users\TRIVIKRAM\Python VSCode\SRF\results\class_distribution.png")
print("Saved histogram to results/class_distribution.png")