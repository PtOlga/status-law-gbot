from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
from datetime import datetime
import re
import json

from src.knowledge_base.dataset import DatasetManager

class ChatAnalyzer:
    """Chat history analyzer"""
    
    def __init__(self, dataset_manager: Optional['DatasetManager'] = None):
        """
        Initialize chat analyzer
        
        Args:
            dataset_manager: Dataset manager for getting chat history
        """
        self.dataset_manager = dataset_manager or DatasetManager()
        self.history = []

    def analyze_chats(self) -> str:
        """
        Analyzes chat history and returns a report
        """
        try:
            success, history = self.dataset_manager.get_chat_history()  # Changed from load_chat_history to get_chat_history
            
            if not success:
                return "Failed to load chat history"
            
            if not history:
                return "No chat history available for analysis"
            
            # Basic analysis
            total_chats = len(history)
            total_messages = sum(len(chat.get("messages", [])) for chat in history)
            avg_messages = total_messages / total_chats if total_chats > 0 else 0
            
            report = f"""
### Chat Analysis Report

- Total conversations: {total_chats}
- Total messages: {total_messages}
- Average messages per conversation: {avg_messages:.1f}
            """
            
            return report
            
        except Exception as e:
            return f"Error during analysis: {str(e)}"

    def extract_question_answer_pairs(self, min_question_length: int = 10) -> List[Dict[str, str]]:
        """
        Extract question-answer pairs from chat history
        
        Args:
            min_question_length: Minimum question length to include in the sample
            
        Returns:
            List of question-answer pairs in format [{"question": "...", "answer": "..."}]
        """
        chat_data = self.get_chat_data()
        qa_pairs = []
        
        for chat in chat_data:
            messages = chat.get("messages", [])
            
            # Go through messages and collect question-answer pairs
            for i in range(len(messages) - 1):
                if messages[i].get("role") == "user" and messages[i+1].get("role") == "assistant":
                    question = messages[i].get("content", "").strip()
                    answer = messages[i+1].get("content", "").strip()
                    
                    # Filter by question length
                    if len(question) >= min_question_length and answer:
                        qa_pairs.append({
                            "question": question,
                            "answer": answer
                        })
        
        return qa_pairs
    
    def analyze_common_questions(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Analysis of most frequently asked questions
        
        Args:
            top_n: Number of most popular questions to return
            
        Returns:
            List of tuples (question, count)
        """
        qa_pairs = self.extract_question_answer_pairs()
        
        # Extract only questions
        questions = [qa["question"] for qa in qa_pairs]
        
        # Preprocess questions for better grouping
        processed_questions = []
        for q in questions:
            # Convert to lowercase
            q = q.lower()
            # Remove punctuation and extra spaces
            q = re.sub(r'[^\w\s]', ' ', q)
            q = re.sub(r'\s+', ' ', q).strip()
            processed_questions.append(q)
        
        # Count question frequency
        question_counter = Counter(processed_questions)
        
        # Get top_n most frequent questions
        return question_counter.most_common(top_n)
    
    def analyze_user_satisfaction(self) -> Dict[str, Any]:
        """
        Analysis of user satisfaction based on chat history
        
        Returns:
            Dictionary with satisfaction metrics
        """
        chat_data = self.get_chat_data()
        
        # Initialize metrics
        metrics = {
            "total_conversations": len(chat_data),
            "avg_messages_per_conversation": 0,
            "avg_conversation_duration": 0,  # in seconds
            "follow_up_questions_rate": 0,   # percentage of dialogs with follow-up questions
        }
        
        if not chat_data:
            return metrics
        
        # Calculate averages
        metrics["avg_messages_per_conversation"] = total_messages / len(chat_data)
        metrics["follow_up_questions_rate"] = conversations_with_followups / len(chat_data) * 100
        
        # Calculate average duration if data exists
        if total_duration > 0:
            metrics["avg_conversation_duration"] = total_duration / len(chat_data)
        
        return metrics
    
    def extract_failed_questions(self) -> List[str]:
        """
        Extract questions that the bot failed to answer satisfactorily
        
        Returns:
            List of questions that need improvement
        """
        chat_data = self.get_chat_data()
        failed_questions = []
        
        # Keywords indicating unsatisfactory response
        failure_indicators = [
            "don't know", "cannot answer", "unable to answer", 
            "I don't have information", "no data available"
        ]
        
        for chat in chat_data:
            messages = chat.get("messages", [])
            
            for i in range(len(messages) - 1):
                if messages[i].get("role") == "user" and messages[i+1].get("role") == "assistant":
                    question = messages[i].get("content", "").strip()
                    answer = messages[i+1].get("content", "").strip().lower()
                    
                    # Check if answer contains failure indicators
                    if any(indicator in answer for indicator in failure_indicators):
                        failed_questions.append(question)
        
        return failed_questions
    
    def export_training_data(self, output_file: str) -> Tuple[bool, str]:
        """
        Export training data in JSONL format
        
        Args:
            output_file: Path to output file
            
        Returns:
            (success, message)
        """
        try:
            qa_pairs = self.extract_question_answer_pairs()
            
            if not qa_pairs:
                return False, "Not enough data for export"
            
            with open(output_file, "w", encoding="utf-8") as f:
                for pair in qa_pairs:
                    training_example = {
                        "messages": [
                            {"role": "user", "content": pair["question"]},
                            {"role": "assistant", "content": pair["answer"]}
                        ]
                    }
                    f.write(json.dumps(training_example, ensure_ascii=False) + "\n")
            
            return True, f"Training data successfully exported to {output_file}. Exported {len(qa_pairs)} examples."
        except Exception as e:
            return False, f"Error exporting training data: {str(e)}"



