"""
Module for evaluation and annotation of bot responses
"""

import json
import os
import datetime
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from src.knowledge_base.dataset import DatasetManager
from huggingface_hub import HfApi

class ChatEvaluator:
    def __init__(self, 
                 dataset_manager: Optional[DatasetManager] = None, 
                 hf_token: str = None,
                 dataset_id: str = None,
                 chat_history_path: str = None):
        """
        Initialize chat evaluator
        
        Args:
            dataset_manager: Dataset manager for retrieving chat history
            hf_token: Hugging Face token for uploading annotations
            dataset_id: Hugging Face dataset ID
            chat_history_path: Path to local chat history directory
        """
        self.dataset_manager = dataset_manager or DatasetManager()
        self.hf_token = hf_token
        self.dataset_id = dataset_id
        self.chat_history_path = chat_history_path
        self.annotations_dir = os.path.join(os.path.dirname(chat_history_path), "annotations") if chat_history_path else None
        
        # Create annotations directory if it doesn't exist
        if self.annotations_dir:
            os.makedirs(self.annotations_dir, exist_ok=True)
    
    def get_chat_history(self) -> List[Dict[str, Any]]:
        """
        Get all chat history data from local files and dataset
        
        Returns:
            List of chat histories
        """
        success, chat_data = self.dataset_manager.get_chat_history()
        if not success or not chat_data:
            return []
        return chat_data
    
    def get_qa_pairs_for_evaluation(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Extract question-answer pairs for evaluation
        
        Args:
            limit: Maximum number of pairs to return
            
        Returns:
            List of QA pairs with metadata
        """
        chat_data = self.get_chat_history()
        qa_pairs = []
        
        for chat in chat_data:
            conversation_id = chat.get("conversation_id", "unknown")
            timestamp = chat.get("timestamp", "")
            history = chat.get("history", [])
            
            # Find user-assistant pairs in history
            for i in range(len(history) - 1):
                if history[i].get("role") == "user" and history[i+1].get("role") == "assistant":
                    question = history[i].get("content", "").strip()
                    answer = history[i+1].get("content", "").strip()
                    
                    # Only include non-empty pairs
                    if question and answer:
                        qa_pairs.append({
                            "conversation_id": conversation_id,
                            "timestamp": timestamp,
                            "question": question,
                            "original_answer": answer,
                            "question_timestamp": history[i].get("timestamp", ""),
                            "answer_timestamp": history[i+1].get("timestamp", "")
                        })
                        
                        # Check if we've reached the limit
                        if len(qa_pairs) >= limit:
                            return qa_pairs
        
        return qa_pairs
    
    def get_evaluation_status(self) -> Dict[str, int]:
        """
        Get status of evaluated QA pairs
        
        Returns:
            Dictionary with counts of evaluated and unevaluated QA pairs
        """
        all_pairs = self.get_qa_pairs_for_evaluation(limit=1000)  # Get a large sample
        evaluated_pairs = self.get_annotations()
        
        # Count evaluated conversation IDs
        evaluated_ids = set(item.get("conversation_id") for item in evaluated_pairs)
        
        return {
            "total_qa_pairs": len(all_pairs),
            "evaluated_pairs": len(evaluated_pairs),
            "unevaluated_pairs": len(all_pairs) - len(evaluated_pairs),
            "evaluated_conversations": len(evaluated_ids)
        }
    
    def save_annotation(self, 
                       conversation_id: str,
                       question: str,
                       original_answer: str,
                       improved_answer: str,
                       ratings: Dict[str, int],
                       notes: str = "") -> Tuple[bool, str]:
        """
        Save evaluation annotation
        
        Args:
            conversation_id: ID of the conversation
            question: User question
            original_answer: Original bot answer
            improved_answer: Improved answer (gold standard)
            ratings: Dictionary with ratings for different criteria
            notes: Optional evaluator notes
            
        Returns:
            (success, message)
        """
        if not self.annotations_dir:
            return False, "Annotations directory not configured"
        
        try:
            # Create annotation object
            annotation = {
                "conversation_id": conversation_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "question": question,
                "original_answer": original_answer,
                "improved_answer": improved_answer,
                "ratings": ratings,
                "notes": notes
            }
            
            # Create filename with conversation_id
            filename = f"annotation_{conversation_id}.json"
            filepath = os.path.join(self.annotations_dir, filename)
            
            # Save to local file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(annotation, f, ensure_ascii=False, indent=2)
            
            # Upload to HuggingFace dataset if configured
            if self.hf_token and self.dataset_id:
                try:
                    api = HfApi(token=self.hf_token)
                    
                    # Extract just the directory name from annotations_dir
                    dir_name = os.path.basename(self.annotations_dir)
                    target_path = f"{dir_name}/{filename}"
                    
                    # Upload the file to the dataset
                    api.upload_file(
                        path_or_fileobj=filepath,
                        path_in_repo=target_path,
                        repo_id=self.dataset_id,
                        repo_type="dataset"
                    )
                    
                except Exception as e:
                    return True, f"Saved locally but failed to upload to dataset: {str(e)}"
            
            return True, "Annotation saved successfully"
        except Exception as e:
            return False, f"Error saving annotation: {str(e)}"
    
    def get_annotations(self) -> List[Dict[str, Any]]:
        """
        Get all saved annotations
        
        Returns:
            List of annotation objects
        """
        if not self.annotations_dir or not os.path.exists(self.annotations_dir):
            return []
        
        annotations = []
        for filename in os.listdir(self.annotations_dir):
            if filename.startswith("annotation_") and filename.endswith(".json"):
                try:
                    filepath = os.path.join(self.annotations_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        annotation = json.load(f)
                        annotations.append(annotation)
                except Exception as e:
                    print(f"Error loading annotation {filename}: {str(e)}")
        
        # Sort by timestamp (newest first)
        annotations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return annotations
    
    def get_annotation_by_conversation_id(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get annotation for a specific conversation
        
        Args:
            conversation_id: Conversation ID to look for
            
        Returns:
            Annotation object or None if not found
        """
        if not self.annotations_dir:
            return None
        
        filepath = os.path.join(self.annotations_dir, f"annotation_{conversation_id}.json")
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading annotation for {conversation_id}: {str(e)}")
        
        return None
    
    def export_training_data(self, output_file: str, min_rating: int = 4) -> Tuple[bool, str]:
        """
        Export high-quality annotated data for fine-tuning
        
        Args:
            output_file: Path to output file
            min_rating: Minimum average rating to include in training data
            
        Returns:
            (success, message)
        """
        annotations = self.get_annotations()
        
        if not annotations:
            return False, "No annotations available for export"
        
        try:
            # Filter annotations by quality
            high_quality_examples = []
            
            for annotation in annotations:
                ratings = annotation.get("ratings", {})
                
                # Calculate average rating
                if ratings:
                    avg_rating = sum(ratings.values()) / len(ratings)
                    
                    # Include only high-quality examples
                    if avg_rating >= min_rating:
                        high_quality_examples.append({
                            "messages": [
                                {"role": "user", "content": annotation.get("question", "")},
                                {"role": "assistant", "content": annotation.get("improved_answer", "")}
                            ]
                        })
            
            if not high_quality_examples:
                return False, f"No examples meet the minimum quality threshold of {min_rating}"
            
            # Save to JSONL format
            with open(output_file, "w", encoding="utf-8") as f:
                for example in high_quality_examples:
                    f.write(json.dumps(example, ensure_ascii=False) + "\n")
            
            return True, f"Successfully exported {len(high_quality_examples)} high-quality examples for training"
        except Exception as e:
            return False, f"Error exporting training data: {str(e)}"
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """
        Generate evaluation summary report
        
        Returns:
            Dictionary with evaluation metrics
        """
        annotations = self.get_annotations()
        
        if not annotations:
            return {
                "total_evaluations": 0,
                "message": "No evaluations available"
            }
        
        # Initialize metrics
        criteria = set()
        for annotation in annotations:
            criteria.update(annotation.get("ratings", {}).keys())
        
        metrics = {
            "total_evaluations": len(annotations),
            "criteria_averages": {},
            "overall_average": 0,
            "improvement_rate": 0  # Percentage of answers that were improved
        }
        
        # Calculate averages for each criterion
        for criterion in criteria:
            values = [a.get("ratings", {}).get(criterion, 0) for a in annotations if criterion in a.get("ratings", {})]
            if values:
                metrics["criteria_averages"][criterion] = sum(values) / len(values)
        
        # Calculate overall average
        all_ratings = []
        for annotation in annotations:
            all_ratings.extend(annotation.get("ratings", {}).values())
        
        if all_ratings:
            metrics["overall_average"] = sum(all_ratings) / len(all_ratings)
        
        # Calculate improvement rate
        improved_count = sum(1 for a in annotations if a.get("original_answer") != a.get("improved_answer"))
        metrics["improvement_rate"] = (improved_count / len(annotations)) * 100
        
        return metrics
