from huggingface_hub import HfApi
from config.settings import (
    DATASET_ID,
    DATASET_VECTOR_STORE_PATH,
    DATASET_CHAT_HISTORY_PATH,
    DATASET_FINE_TUNED_PATH,
    DATASET_ANNOTATIONS_PATH,
    HF_TOKEN
)

api = HfApi(token=HF_TOKEN)
dataset_name = DATASET_ID

# Initialize dataset structure
directories = [
    DATASET_VECTOR_STORE_PATH,
    DATASET_CHAT_HISTORY_PATH,
    DATASET_FINE_TUNED_PATH,
    DATASET_ANNOTATIONS_PATH
]

try:
    for directory in directories:
        api.upload_file(
            path_or_fileobj=b"",
            path_in_repo=f"{directory}/.gitkeep",
            repo_id=dataset_name,
            repo_type="dataset"
        )
        print(f"✓ Created directory: {directory}")

    print("\nDataset structure successfully initialized!")

except Exception as e:
    print(f"Error occurred: {str(e)}")

