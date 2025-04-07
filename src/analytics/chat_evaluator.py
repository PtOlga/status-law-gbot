"""
Module for evaluation and annotation of bot responses
"""

import json
import os
import datetime
from typing import List, Dict, Any, Tuple, Optional
import io
import logging
from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

from config.settings import (
    DATASET_ID,
    HF_TOKEN,
    CHAT_HISTORY_PATH,
    DATASET_CHAT_HISTORY_PATH,
    DATASET_ANNOTATIONS_PATH
)

class ChatEvaluator:
    def __init__(self, hf_token: str = None, dataset_id: str = None):
        """
        Initialize chat evaluator
        
        Args:
            hf_token: Hugging Face token
            dataset_id: Dataset ID on Hugging Face
        """
        self.hf_token = hf_token or HF_TOKEN
        self.dataset_id = dataset_id or DATASET_ID
        self.api = HfApi(token=self.hf_token)
        
        # Используем пути из settings
        self.chat_history_path = DATASET_CHAT_HISTORY_PATH
        self.annotations_path = DATASET_ANNOTATIONS_PATH
        
        # Ensure directories exist in dataset
        try:
            self._ensure_dataset_structure()
        except Exception as e:
            logger.error(f"Failed to ensure dataset structure: {e}")

    def _ensure_dataset_structure(self):
        """Ensure required directories exist in dataset"""
        try:
            files = self.api.list_repo_files(self.dataset_id, repo_type="dataset")
            
            # Check and create chat history directory
            if self.chat_history_path not in files:
                self.api.upload_file(
                    path_or_fileobj=io.BytesIO(b""),
                    path_in_repo=f"{self.chat_history_path}/.gitkeep",
                    repo_id=self.dataset_id,
                    repo_type="dataset"
                )
            
            # Check and create annotations directory
            if self.annotations_path not in files:
                self.api.upload_file(
                    path_or_fileobj=io.BytesIO(b""),
                    path_in_repo=f"{self.annotations_path}/.gitkeep",
                    repo_id=self.dataset_id,
                    repo_type="dataset"
                )
        except Exception as e:
            logger.error(f"Error ensuring dataset structure: {e}")
            raise

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Get all chat history data from dataset"""
        try:
            chat_data = []
            files = self.api.list_repo_files(self.dataset_id, repo_type="dataset")
            
            # Debug print all files
            logger.info(f"All files in dataset:")
            for f in files:
                logger.info(f"  - {f}")
            
            logger.info(f"Looking for files that start with: {self.chat_history_path}/")
            
            # Filter chat history files
            chat_files = [f for f in files if f.startswith(f"{self.chat_history_path}/") 
                         and f.endswith('.json')]
            
            # Debug print
            logger.info(f"Found chat files: {len(chat_files)}")
            logger.info(f"Chat files: {chat_files}")
            
            return chat_data
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
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
        
        print(f"Debug - Processing {len(chat_data)} chat histories")  # Debug print
        
        for chat in chat_data:
            conversation_id = chat.get("conversation_id", "unknown")
            timestamp = chat.get("timestamp", "")
            messages = chat.get("messages", [])
            
            # Find user-assistant pairs in messages
            for i in range(len(messages) - 1):
                if (messages[i].get("role") == "user" and 
                    messages[i+1].get("role") == "assistant"):
                    question = messages[i].get("content", "").strip()
                    answer = messages[i+1].get("content", "").strip()
                    
                    # Only include non-empty pairs
                    if question and answer:
                        qa_pairs.append({
                            "conversation_id": conversation_id,
                            "timestamp": timestamp,
                            "question": question,
                            "original_answer": answer,
                            "question_timestamp": messages[i].get("timestamp", ""),
                            "answer_timestamp": messages[i+1].get("timestamp", "")
                        })
                        
                        # Check if we've reached the limit
                        if len(qa_pairs) >= limit:
                            print(f"Debug - Reached limit of {limit} QA pairs")  # Debug print
                            return qa_pairs
        
        print(f"Debug - Extracted {len(qa_pairs)} QA pairs")  # Debug print
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
            filename = f"{self.annotations_path}/annotation_{conversation_id}.json"
            
            # Convert to JSON string
            json_content = json.dumps(annotation, ensure_ascii=False, indent=2)
            
            # Upload to dataset
            self.api.upload_file(
                path_or_fileobj=io.StringIO(json_content),
                path_in_repo=filename,
                repo_id=self.dataset_id,
                repo_type="dataset"
            )
            
            return True, "Annotation saved successfully"
            
        except Exception as e:
            logger.error(f"Error saving annotation: {e}")
            return False, f"Failed to save annotation: {str(e)}"
    
    def get_annotations(self) -> List[Dict[str, Any]]:
        """
        Get all saved annotations from dataset
        """
        try:
            annotations = []
            files = self.api.list_repo_files(self.dataset_id, repo_type="dataset")
            
            for file in files:
                if file.startswith(f"{self.annotations_path}/annotation_") and file.endswith(".json"):
                    try:
                        # Download and parse annotation file
                        content = self.api.hf_hub_download(
                            repo_id=self.dataset_id,
                            filename=file,
                            repo_type="dataset"
                        )
                        with open(content, 'r', encoding='utf-8') as f:
                            annotation = json.load(f)
                            annotations.append(annotation)
                    except Exception as e:
                        logger.error(f"Error loading annotation {file}: {e}")
            
            # Sort by timestamp (newest first)
            annotations.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return annotations
            
        except Exception as e:
            logger.error(f"Error getting annotations: {e}")
            return []
    
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













