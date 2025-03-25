"""
Модуль для анализа истории чатов и извлечения полезных данных для обучения
"""

import json
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter, defaultdict
import re
from datetime import datetime
from src.knowledge_base.dataset import DatasetManager

class ChatAnalyzer:
    def __init__(self, dataset_manager: Optional[DatasetManager] = None):
        """
        Инициализация анализатора чатов
        
        Args:
            dataset_manager: Менеджер датасетов для получения истории чатов
        """
        self.dataset_manager = dataset_manager or DatasetManager()
        
    def get_chat_data(self) -> List[Dict[str, Any]]:
        """
        Получение всех данных чатов из датасета
        
        Returns:
            Список историй чатов
        """
        success, chat_data = self.dataset_manager.get_chat_history()
        if not success or not chat_data:
            return []
        return chat_data
    
    def extract_question_answer_pairs(self, min_question_length: int = 10) -> List[Dict[str, str]]:
        """
        Извлечение пар вопрос-ответ из истории чатов
        
        Args:
            min_question_length: Минимальная длина вопроса для включения в выборку
            
        Returns:
            Список пар вопрос-ответ в формате [{"question": "...", "answer": "..."}]
        """
        chat_data = self.get_chat_data()
        qa_pairs = []
        
        for chat in chat_data:
            messages = chat.get("messages", [])
            
            # Проходим по сообщениям и собираем пары вопрос-ответ
            for i in range(len(messages) - 1):
                if messages[i].get("role") == "user" and messages[i+1].get("role") == "assistant":
                    question = messages[i].get("content", "").strip()
                    answer = messages[i+1].get("content", "").strip()
                    
                    # Фильтруем по длине вопроса
                    if len(question) >= min_question_length and answer:
                        qa_pairs.append({
                            "question": question,
                            "answer": answer
                        })
        
        return qa_pairs
    
    def analyze_common_questions(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Анализ наиболее часто задаваемых вопросов
        
        Args:
            top_n: Количество самых популярных вопросов для возврата
            
        Returns:
            Список кортежей (вопрос, количество)
        """
        qa_pairs = self.extract_question_answer_pairs()
        
        # Извлекаем только вопросы
        questions = [qa["question"] for qa in qa_pairs]
        
        # Предварительная обработка вопросов для лучшего группирования
        processed_questions = []
        for q in questions:
            # Преобразуем в нижний регистр
            q = q.lower()
            # Удаляем пунктуацию и лишние пробелы
            q = re.sub(r'[^\w\s]', ' ', q)
            q = re.sub(r'\s+', ' ', q).strip()
            processed_questions.append(q)
        
        # Подсчет частоты вопросов
        question_counter = Counter(processed_questions)
        
        # Получаем top_n самых частых вопросов
        return question_counter.most_common(top_n)
    
    def analyze_user_satisfaction(self) -> Dict[str, Any]:
        """
        Анализ удовлетворенности пользователей на основе истории чатов
        
        Returns:
            Словарь с метриками удовлетворенности
        """
        chat_data = self.get_chat_data()
        
        # Инициализация метрик
        metrics = {
            "total_conversations": len(chat_data),
            "avg_messages_per_conversation": 0,
            "avg_conversation_duration": 0,  # в секундах
            "follow_up_questions_rate": 0,   # процент диалогов с дополнительными вопросами
        }
        
        if not chat_data:
            return metrics
        
        # Подсчет общего количества сообщений и длительности диалогов
        total_messages = 0
        conversations_with_followups = 0
        total_duration = 0
        
        for chat in chat_data:
            messages = chat.get("messages", [])
            total_messages += len(messages)
            
            # Проверка наличия дополнительных вопросов от пользователя
            user_messages = [m for m in messages if m.get("role") == "user"]
            if len(user_messages) > 1:
                conversations_with_followups += 1
            
            # Расчет длительности диалога, если есть временные метки
            if len(messages) >= 2 and all(["timestamp" in m for m in [messages[0], messages[-1]]]):
                try:
                    start_time = datetime.fromisoformat(messages[0]["timestamp"])
                    end_time = datetime.fromisoformat(messages[-1]["timestamp"])
                    duration = (end_time - start_time).total_seconds()
                    total_duration += duration
                except (ValueError, KeyError):
                    pass
        
        # Расчет средних значений
        metrics["avg_messages_per_conversation"] = total_messages / len(chat_data)
        metrics["follow_up_questions_rate"] = conversations_with_followups / len(chat_data) * 100
        
        # Расчет средней длительности, если есть данные
        if total_duration > 0:
            metrics["avg_conversation_duration"] = total_duration / len(chat_data)
        
        return metrics
    
    def extract_failed_questions(self) -> List[str]:
        """
        Извлечение вопросов, на которые бот не смог дать удовлетворительный ответ
        
        Returns:
            Список вопросов, требующих улучшения
        """
        chat_data = self.get_chat_data()
        failed_questions = []
        
        # Ключевые слова, указывающие на неудовлетворительный ответ
        failure_indicators = [
            "не знаю", "не могу ответить", "затрудняюсь ответить", 
            "у меня нет информации", "не имею данных"
        ]
        
        for chat in chat_data:
            messages = chat.get("messages", [])
            
            for i in range(len(messages) - 1):
                if messages[i].get("role") == "user" and messages[i+1].get("role") == "assistant":
                    question = messages[i].get("content", "").strip()
                    answer = messages[i+1].get("content", "").strip().lower()
                    
                    # Проверяем, содержит ли ответ индикаторы неудачи
                    if any(indicator in answer for indicator in failure_indicators):
                        failed_questions.append(question)
        
        return failed_questions
    
    def export_training_data(self, output_file: str) -> Tuple[bool, str]:
        """
        Экспорт данных для обучения в формате JSONL
        
        Args:
            output_file: Путь к выходному файлу
            
        Returns:
            (успех, сообщение)
        """
        try:
            qa_pairs = self.extract_question_answer_pairs()
            
            if not qa_pairs:
                return False, "Нет достаточного количества данных для экспорта"
            
            with open(output_file, "w", encoding="utf-8") as f:
                for pair in qa_pairs:
                    training_example = {
                        "messages": [
                            {"role": "user", "content": pair["question"]},
                            {"role": "assistant", "content": pair["answer"]}
                        ]
                    }
                    f.write(json.dumps(training_example, ensure_ascii=False) + "\n")
            
            return True, f"Данные для обучения успешно экспортированы в {output_file}. Экспортировано {len(qa_pairs)} примеров."
        except Exception as e:
            return False, f"Ошибка при экспорте данных для обучения: {str(e)}"
    
    def generate_analytics_report(self) -> Dict[str, Any]:
        """
        Генерация полного аналитического отчета
        
        Returns:
            Словарь с различными метриками и анализом
        """
        report = {}
        
        # Базовые метрики
        chat_data = self.get_chat_data()
        report["total_conversations"] = len(chat_data)
        
        # Удовлетворенность пользователей
        report["satisfaction_metrics"] = self.analyze_user_satisfaction()
        
        # Частые вопросы
        report["common_questions"] = self.analyze_common_questions(top_n=20)
        
        # Вопросы без ответов
        report["failed_questions"] = self.extract_failed_questions()
        report["failed_questions_count"] = len(report["failed_questions"])
        
        # Статистика по количеству пар вопрос-ответ
        qa_pairs = self.extract_question_answer_pairs()
        report["qa_pairs_count"] = len(qa_pairs)
        
        return report

if __name__ == "__main__":
    # Пример использования
    analyzer = ChatAnalyzer()
    report = analyzer.generate_analytics_report()
    print(f"Всего диалогов: {report['total_conversations']}")
    print(f"Пар вопрос-ответ для обучения: {report['qa_pairs_count']}")
    print("\nСамые популярные вопросы:")
    for question, count in report['common_questions'][:5]:
        print(f" - {question} ({count} раз)")