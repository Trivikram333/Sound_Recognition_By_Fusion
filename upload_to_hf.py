import os
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

load_dotenv()
# 1. Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_NAME = "ESC-50_SRF"  
REPO_TYPE = "model"

api = HfApi(token=HF_TOKEN)

# 2. Get your Hugging Face username automatically
user_info = api.whoami()
username = user_info["name"]
repo_id = f"{username}/{REPO_NAME}"

print(f"Target repository: {repo_id}")

# 3. Create the remote model repository if it doesn't already exist
create_repo(
    repo_id=repo_id,
    token=HF_TOKEN,
    repo_type=REPO_TYPE,
    exist_ok=True,
    private=False  # Set to True if you want the weights private
)
print("Remote repository confirmed!")

# 4. Upload all files from your local weights/ directory
weights_dir = "weights"

if not os.path.exists(weights_dir) or not os.listdir(weights_dir):
    raise FileNotFoundError(f"'{weights_dir}' folder is empty or does not exist!")

print(f"Uploading files from '{weights_dir}' to Hugging Face...")
api.upload_folder(
    folder_path=weights_dir,
    repo_id=repo_id,
    repo_type=REPO_TYPE,
    path_in_repo="weights"  # Preserves the weights/ folder structure remotely
)

print("\n" + "=" * 55)
print("Upload Complete!")
print(f"View your weights at: https://huggingface.co/{repo_id}")
print("=" * 55)