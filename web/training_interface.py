"""
Веб-интерфейс для управления моделями и запуска дообучения
"""

import os
import json
import gradio as gr
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

from src.analytics.chat_analyzer import ChatAnalyzer
from src.training.fine_tuner import FineTuner, finetune_from_chat_history
from src.training.model_manager import ModelManager
from config.settings import MODEL_PATH, TRAINING_OUTPUT_DIR

# Инициализация менеджеров
model_manager = ModelManager()
chat_analyzer = ChatAnalyzer()

def get_models_df():
    """
    Получение датафрейма с моделями из реестра
    
    Returns:
        pandas.DataFrame: Датафрейм с моделями
    """
    models = model_manager.list_models()
    
    if not models:
        return pd.DataFrame(columns=["model_id", "version", "description", "is_active", "registration_date"])
    
    # Создаем датафрейм
    df = pd.DataFrame(models)
    
    # Выбираем нужные колонки
    columns = ["model_id", "version", "description", "is_active", "registration_date"]
    df = df[columns]
    
    # Сортируем по model_id и registration_date
    df = df.sort_values(by=["model_id", "registration_date"], ascending=[True, False])
    
    return df

def generate_chat_analysis():
    """
    Генерация аналитического отчета по истории чатов
    
    Returns:
        str: HTML-отчет
    """
    report = chat_analyzer.generate_analytics_report()
    
    if not report or report.get("total_conversations", 0) == 0:
        return "### Нет данных для анализа\nИстория чатов пуста или не может быть загружена."
    
    # Формируем HTML-отчет
    html = f"""
    ### Аналитический отчет по истории чатов
    
    #### Основные метрики
    - **Всего диалогов:** {report['total_conversations']}
    - **Пар вопрос-ответ для обучения:** {report['qa_pairs_count']}
    - **Вопросы без ответов:** {report['failed_questions_count']}
    
    #### Метрики удовлетворенности
    - **Среднее число сообщений в диалоге:** {report['satisfaction_metrics']['avg_messages_per_conversation']:.2f}
    - **Процент диалогов с дополнительными вопросами:** {report['satisfaction_metrics']['follow_up_questions_rate']:.2f}%
    """
    
    # Популярные вопросы
    if report.get('common_questions'):
        html += "\n\n#### Популярные вопросы\n"
        for i, (question, count) in enumerate(report['common_questions'][:10], 1):
            html += f"{i}. \"{question}\" ({count} раз)\n"
    
    # Вопросы без ответов
    if report.get('failed_questions'):
        html += "\n\n#### Примеры вопросов без ответов\n"
        for i, question in enumerate(report['failed_questions'][:5], 1):
            html += f"{i}. \"{question}\"\n"
    
    return html

def register_model_action(model_id, version, source, description, set_active):
    """
    Действие регистрации модели
    
    Args:
        model_id: Идентификатор модели
        version: Версия модели
        source: Источник модели
        description: Описание модели
        set_active: Установить как активную
        
    Returns:
        str: Результат операции
    """
    # Проверка входных данных
    if not model_id or not version or not source:
        return "Ошибка: все поля обязательны для заполнения"
    
    # Регистрация модели
    success, message = model_manager.register_model(
        model_id=model_id,
        version=version,
        source=source,
        description=description,
        is_active=set_active
    )
    
    if not success:
        return f"Ошибка: {message}"
    
    # Если установлена опция загрузки модели, загружаем её
    if source.startswith("hf://"):
        success, download_message = model_manager.download_model(model_id, version)
        if not success:
            return f"Модель зарегистрирована, но не загружена: {download_message}"
        message += f"\n{download_message}"
    
    return message

def import_local_model_action(source_path, model_id, version, description, set_active):
    """
    Действие импорта локальной модели
    
    Args:
        source_path: Путь к директории с моделью
        model_id: Идентификатор модели
        version: Версия модели
        description: Описание модели
        set_active: Установить как активную
        
    Returns:
        str: Результат операции
    """
    # Проверка входных данных
    if not source_path or not model_id or not version:
        return "Ошибка: все поля обязательны для заполнения"
    
    # Проверка существования директории
    if not os.path.exists(source_path):
        return f"Ошибка: директория {source_path} не существует"
    
    # Импорт модели
    success, message = model_manager.import_local_model(
        source_path=source_path,
        model_id=model_id,
        version=version,
        description=description,
        is_active=set_active
    )
    
    return message

def set_active_model_action(model_row_index, models_df):
    """
    Действие установки активной модели
    
    Args:
        model_row_index: Индекс строки модели в датафрейме
        models_df: Датафрейм с моделями
        
    Returns:
        str: Результат операции
    """
    try:
        # Получаем информацию о выбранной модели
        model_row = models_df.iloc[model_row_index]
        model_id = model_row["model_id"]
        version = model_row["version"]
        
        # Устанавливаем как активную
        success, message = model_manager.set_active_model(model_id, version)
        
        return message
    except Exception as e:
        return f"Ошибка: {str(e)}"

def delete_model_action(model_row_index, models_df):
    """
    Действие удаления модели
    
    Args:
        model_row_index: Индекс строки модели в датафрейме
        models_df: Датафрейм с моделями
        
    Returns:
        str: Результат операции
    """
    try:
        # Получаем информацию о выбранной модели
        model_row = models_df.iloc[model_row_index]
        model_id = model_row["model_id"]
        version = model_row["version"]
        
        # Удаляем модель
        success, message = model_manager.delete_model(model_id, version)
        
        return message
    except Exception as e:
        return f"Ошибка: {str(e)}"

def start_finetune_action(
    epochs, 
    batch_size, 
    learning_rate, 
    base_model_id,
    new_model_id,
    new_version,
    description,
    set_active
):
    """
    Действие запуска дообучения модели
    
    Args:
        epochs: Количество эпох обучения
        batch_size: Размер батча
        learning_rate: Скорость обучения
        base_model_id: Ид
        """