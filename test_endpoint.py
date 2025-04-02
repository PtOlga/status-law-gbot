from huggingface_hub import HfApi, InferenceClient
import os

# Ваш текущий токен
token = os.getenv("HUGGINGFACE_TOKEN")

# Проверка типа доступа
api = HfApi(token=token)
try:
    # Проверяем информацию об аккаунте
    user_info = api.whoami()
    print("Account type:", user_info.get("type"))
    print("Plan:", user_info.get("plan"))
    
    # Проверяем доступные эндпоинты
    endpoints = api.list_endpoints()
    print("\nAvailable endpoints:")
    for endpoint in endpoints:
        print(f"- {endpoint.name}: {endpoint.url}")
except Exception as e:
    print(f"Error checking endpoints: {e}")

# Проверяем текущий клиент
client = InferenceClient(
    "HuggingFaceH4/zephyr-7b-beta",  # или ваша текущая модель
    token=token
)

# Проверяем тип подключения
print("\nClient information:")
print("API Base URL:", client.api_url)
print("Headers:", client.headers)