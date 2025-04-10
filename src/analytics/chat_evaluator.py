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
        Initialize chat evaluator with lazy loading
        
        Args:
            hf_token: Hugging Face token
            dataset_id: Dataset ID on Hugging Face
        """
        self.hf_token = hf_token or HF_TOKEN
        self.dataset_id = dataset_id or DATASET_ID
        self.api = HfApi(token=self.hf_token)
        
        # Using paths from settings
        self.chat_history_path = DATASET_CHAT_HISTORY_PATH
        self.annotations_path = DATASET_ANNOTATIONS_PATH
        
        # Cache for chat histories and QA pairs
        self._chat_histories = None
        self._qa_pairs = None
        self._annotations = None
        
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

    def reset_cache(self):
        """
        Reset the cache to force reload of data
        """
        self._chat_histories = None
        self._qa_pairs = None
        self._annotations = None
        logger.info("Chat evaluator cache has been reset")

    def get_chat_history(self, force_reload=False) -> List[Dict[str, Any]]:
        """
        Get all chat histories from the dataset
        
        Args:
            force_reload: If True, ignore cache and reload from dataset
        """
        # Return cached data if available and not forcing reload
        if self._chat_histories is not None and not force_reload:
            logger.debug("Returning cached chat histories")
            return self._chat_histories
        
        try:
            # Get list of all files in chat history directory
            files = self.api.list_repo_files(self.dataset_id, repo_type="dataset")  
            
            # Filter for chat history files
            chat_path = f"{self.chat_history_path}/"
            chat_files = [f for f in files if f.startswith(chat_path) and f.endswith('.json')]
            logger.debug(f"Found {len(chat_files)} chat files")  # More compact log

            histories = []
            for file in chat_files:
                try:
                    # Download and parse each chat file
                    content = self.api.hf_hub_download(
                        repo_id=self.dataset_id,
                        filename=file,
                        repo_type="dataset"
                    )
                    with open(content, 'r', encoding='utf-8') as f:
                        chat_data = json.load(f)
                        if isinstance(chat_data, dict) and 'history' in chat_data:
                            histories.append(chat_data)
                        else:
                            logger.warning(f"Invalid chat history format in {file}")
                except Exception as e:
                    logger.error(f"Error processing chat file {file}: {e}")
                    continue

            # Cache the results
            self._chat_histories = histories
            return histories

        except Exception as e:
            logger.error(f"Failed to get chat histories: {e}")
            return []

    def extract_qa_pairs(self, histories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract question-answer pairs from chat histories
        """
        qa_pairs = []
        
        for history in histories:
            messages = history.get('history', [])
            current_question = None
            
            for msg in messages:
                if msg.get('role') == 'user':
                    current_question = msg.get('content')
                elif msg.get('role') == 'assistant' and current_question:
                    qa_pairs.append({
                        'conversation_id': history.get('conversation_id'),
                        'question': current_question,
                        'answer': msg.get('content'),
                        'timestamp': history.get('timestamp')
                    })
                    current_question = None

        logger.debug(f"Extracted {len(qa_pairs)} QA pairs")
        return qa_pairs

    def get_qa_pairs_for_evaluation(self, limit: int = 50, force_reload=False) -> List[Dict[str, Any]]:
        """
        Extract question-answer pairs for evaluation
        
        Args:
            limit: Maximum number of pairs to return
            force_reload: If True, force reload from dataset
            
        Returns:
            List of QA pairs with metadata
        """
        # Return cached data if available and not forcing reload
        if self._qa_pairs is not None and not force_reload:
            logger.debug("Returning cached QA pairs")
            return self._qa_pairs[:limit]  # Respect the limit parameter
        
        chat_data = self.get_chat_history(force_reload=force_reload)
        qa_pairs = []
        
        logger.debug(f"Processing {len(chat_data)} chat histories")
        
        for chat in chat_data:
            conversation_id = chat.get("conversation_id", "unknown")
            timestamp = chat.get("timestamp", "")
            messages = chat.get("history", [])
            
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
        
        # Cache the results
        self._qa_pairs = qa_pairs
        
        logger.debug(f"Extracted {len(qa_pairs)} QA pairs")
        # Return up to the limit
        return qa_pairs[:limit]
    
    def get_evaluation_status(self, force_reload=False) -> Dict[str, int]:
        """
        Get status of evaluated QA pairs
        
        Args:
            force_reload: If True, force reload from dataset
            
        Returns:
            Dictionary with counts of evaluated and unevaluated QA pairs
        """
        all_pairs = self.get_qa_pairs_for_evaluation(limit=1000, force_reload=force_reload)  # Get a large sample
        evaluated_pairs = self.get_annotations(force_reload=force_reload)
        
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
            
            # Convert to JSON bytes
            json_content = json.dumps(annotation, ensure_ascii=False, indent=2).encode('utf-8')
            
            # Upload to dataset using bytes buffer
            self.api.upload_file(
                path_or_fileobj=io.BytesIO(json_content),
                path_in_repo=filename,
                repo_id=self.dataset_id,
                repo_type="dataset"
            )
            
            # Reset annotations cache
            self._annotations = None
            
            return True, "Annotation saved successfully"
            
        except Exception as e:
            logger.error(f"Error saving annotation: {e}")
            return False, f"Failed to save annotation: {str(e)}"
    
    def get_annotations(self, force_reload=False) -> List[Dict[str, Any]]:
        """
        Get all saved annotations from dataset
        
        Args:
            force_reload: If True, force reload from dataset
        """
        # Return cached data if available and not forcing reload
        if self._annotations is not None and not force_reload:
            logger.debug("Returning cached annotations")
            return self._annotations
        
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
            
            # Cache the results
            self._annotations = annotations
            
            return annotations
            
        except Exception as e:
            logger.error(f"Error getting annotations: {e}")
            return []
    
    def get_annotation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Get specific annotation by conversation ID
        """
        try:
            # First check if annotations are loaded
            if self._annotations is not None:
                for annotation in self._annotations:
                    if annotation.get("conversation_id") == conversation_id:
                        return annotation
            
            # If not found in cache, try direct file access
            filename = f"{self.annotations_path}/annotation_{conversation_id}.json"
            try:
                content = self.api.hf_hub_download(
                    repo_id=self.dataset_id,
                    filename=filename,
                    repo_type="dataset"
                )
                
                with open(content, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading annotation for {conversation_id}: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting annotation: {e}")
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

