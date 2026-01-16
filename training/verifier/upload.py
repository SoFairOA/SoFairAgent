from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="best_large/", repo_id="SoFairOA/SoFairVerifier", repo_type="model")
