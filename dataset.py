from huggingface_hub import HfApi
api = HfApi()

# Имя существующего датасета
dataset_name = "Rulga/status-law-knowledge-base"

# Создаем структуру с пустыми файлами
try:
    # Создаем .gitkeep в vector_store
    api.upload_file(
        path_or_fileobj=b"",  # пустой файл
        path_in_repo="vector_store/.gitkeep",
        repo_id=dataset_name,
        repo_type="dataset"
    )
    print("✓ Создана папка vector_store")

    # Создаем .gitkeep в chat_history
    api.upload_file(
        path_or_fileobj=b"",
        path_in_repo="chat_history/.gitkeep",
        repo_id=dataset_name,
        repo_type="dataset"
    )
    print("✓ Создана папка chat_history")

    print("\nСтруктура датасета успешно создана!")

except Exception as e:
    print(f"Произошла ошибка: {str(e)}")

