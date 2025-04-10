"""
Interface components for chat evaluation
"""

import gradio as gr
import pandas as pd
from src.analytics.chat_evaluator import ChatEvaluator
import json
import os
from typing import Dict, Any, List, Tuple

def get_evaluation_status(evaluator, force_reload=False):
    """
    Get evaluation status as formatted string and refresh QA data
    
    Args:
        evaluator: ChatEvaluator instance
        force_reload: If True, force reload data from dataset
        
    Returns:
        Status message, updated QA table and refresh message
    """
    try:
        # First, reset cache if forcing reload
        if force_reload:
            evaluator.reset_cache()
            
        # Get status data
        status = evaluator.get_evaluation_status(force_reload=force_reload)
        
        # Get updated QA table
        qa_table = get_qa_pairs_dataframe(evaluator, show_evaluated=False, force_reload=force_reload)
        
        status_message = f"""
        Total QA Pairs: {status['total_qa_pairs']}
        Evaluated Pairs: {status['evaluated_pairs']}
        Unevaluated Pairs: {status['unevaluated_pairs']}
        Evaluated Conversations: {status['evaluated_conversations']}
        """
        
        refresh_message = "Data refreshed successfully" if force_reload else ""
        
        return status_message, qa_table, refresh_message
    except Exception as e:
        logger.error(f"Error getting evaluation status: {e}")
        
        # Import pandas here to avoid circular imports
        import pandas as pd
        empty_df = pd.DataFrame(columns=["Conversation ID", "Question", "Answer", "Evaluated"])
        
        return f"Error getting status: {str(e)}", empty_df, f"Error: {str(e)}"

def get_qa_pairs_dataframe(evaluator, show_evaluated=False, force_reload=False):
    """
    Get QA pairs as DataFrame for the evaluation interface
    
    Args:
        evaluator: ChatEvaluator instance
        show_evaluated: If True, include already evaluated pairs
        force_reload: If True, force reload from dataset
        
    Returns:
        DataFrame with QA pairs
    """
    try:
        # Get QA pairs with potential force reload
        qa_pairs = evaluator.get_qa_pairs_for_evaluation(limit=100, force_reload=force_reload)
        
        # Get annotations
        annotations = evaluator.get_annotations(force_reload=force_reload)
        evaluated_ids = {a.get("conversation_id") for a in annotations}
        
        # Filter out already evaluated pairs if needed
        if not show_evaluated:
            qa_pairs = [qa for qa in qa_pairs if qa["conversation_id"] not in evaluated_ids]
        
        # Convert to DataFrame
        if qa_pairs:
            import pandas as pd
            
            df = pd.DataFrame([
                {
                    "Conversation ID": qa["conversation_id"],
                    "Question": qa["question"][:50] + "..." if len(qa["question"]) > 50 else qa["question"],
                    "Answer": qa["original_answer"][:100] + "..." if len(qa["original_answer"]) > 100 else qa["original_answer"],
                    "Evaluated": "Yes" if qa["conversation_id"] in evaluated_ids else "No"
                }
                for qa in qa_pairs
            ])
            return df
        else:
            import pandas as pd
            return pd.DataFrame(columns=["Conversation ID", "Question", "Answer", "Evaluated"])
    except Exception as e:
        logger.error(f"Error getting QA pairs dataframe: {e}")
        import pandas as pd
        return pd.DataFrame(columns=["Conversation ID", "Question", "Answer", "Evaluated"])

def load_qa_pair_for_evaluation(conversation_id: str, evaluator: ChatEvaluator) -> Tuple[str, str, str, int, int, int, int, int, str]:
    """
    Load a QA pair for evaluation
    
    Args:
        conversation_id: ID of the conversation to load
        evaluator: ChatEvaluator instance
        
    Returns:
        Tuple of (question, original_answer, improved_answer, accuracy, completeness, 
                 relevance, clarity, legal_correctness, notes)
    """
    # Get all QA pairs
    qa_pairs = evaluator.get_qa_pairs_for_evaluation(limit=1000)
    
    # Get existing annotation if any
    annotation = evaluator.get_annotation(conversation_id)
    
    if annotation:
        return (
            annotation.get("question", ""),
            annotation.get("original_answer", ""),  # Changed from original_answer
            annotation.get("improved_answer", ""),  # Changed from improved_answer
            annotation.get("ratings", {}).get("accuracy", 1),
            annotation.get("ratings", {}).get("completeness", 1),
            annotation.get("ratings", {}).get("relevance", 1),
            annotation.get("ratings", {}).get("clarity", 1),
            annotation.get("ratings", {}).get("legal_correctness", 1),
            annotation.get("notes", "")
        )
    
    # If no annotation exists, find the conversation in QA pairs
    for qa_pair in qa_pairs:
        if qa_pair.get("conversation_id") == conversation_id:
            return (
                qa_pair.get("question", ""),
                qa_pair.get("original_answer", ""),  # Changed from answer
                "",  # Empty improved answer
                1,   # Default ratings
                1,
                1,
                1,
                1,
                ""   # Empty notes
            )
            
    return "", "", "", 1, 1, 1, 1, 1, ""  # Return empty values if not found

def save_evaluation(
    conversation_id: str,
    question: str,
    original_answer: str,
    improved_answer: str,
    accuracy: int,
    completeness: int,
    relevance: int,
    clarity: int,
    legal_correctness: int,
    notes: str,
    evaluator: ChatEvaluator
) -> str:
    """
    Save evaluation to file and dataset
    
    Args:
        evaluator: ChatEvaluator instance
        conversation_id: ID of the conversation
        question: User question
        original_answer: Original bot answer
        improved_answer: Improved answer
        accuracy: Rating for factual accuracy (1-5)
        completeness: Rating for completeness (1-5)
        relevance: Rating for relevance (1-5)
        clarity: Rating for clarity (1-5)
        legal_correctness: Rating for legal correctness (1-5)
        notes: Evaluator notes
        
    Returns:
        Status message
    """
    # Create ratings dictionary
    ratings = {
        "accuracy": accuracy,
        "completeness": completeness,
        "relevance": relevance,
        "clarity": clarity,
        "legal_correctness": legal_correctness
    }
    
    # Save annotation
    success, message = evaluator.save_annotation(
        conversation_id=conversation_id,
        question=question,
        original_answer=original_answer,
        improved_answer=improved_answer,
        ratings=ratings,
        notes=notes
    )
    
    return message

def generate_evaluation_report_html(evaluator: ChatEvaluator) -> str:
    """
    Generate HTML report of evaluation metrics
    
    Args:
        evaluator: ChatEvaluator instance
        
    Returns:
        HTML string with report
    """
    report = evaluator.generate_evaluation_report()
    
    if report["total_evaluations"] == 0:
        return "<p>No evaluations available yet.</p>"
    
    # Format criteria averages
    criteria_html = ""
    for criterion, avg in report["criteria_averages"].items():
        # Calculate stars representation (1-5)
        stars = "★" * int(avg) + "☆" * (5 - int(avg))
        criteria_html += f"""
        <tr>
            <td>{criterion.capitalize()}</td>
            <td>{avg:.2f}/5.0</td>
            <td>{stars}</td>
        </tr>
        """
    
    # Overall stars representation
    overall_stars = "★" * int(report["overall_average"]) + "☆" * (5 - int(report["overall_average"]))
    
    html = f"""
    <div style="padding: 15px; border: 1px solid #ccc; border-radius: 5px; margin-top: 10px;">
        <h3>Evaluation Report</h3>
        
        <p><strong>Total Evaluations:</strong> {report["total_evaluations"]}</p>
        <p><strong>Overall Average Rating:</strong> {report["overall_average"]:.2f}/5.0 {overall_stars}</p>
        <p><strong>Improvement Rate:</strong> {report["improvement_rate"]:.1f}% of responses were improved</p>
        
        <h4>Criteria Ratings:</h4>
        <table style="width: 100%; border-collapse: collapse;">
            <tr>
                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Criterion</th>
                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Average Score</th>
                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #ddd;">Rating</th>
            </tr>
            {criteria_html}
        </table>
    </div>
    """
    
    return html

def export_training_data_action(evaluator: ChatEvaluator, min_rating: int, output_file: str) -> str:
    """
    Action for exporting training data
    
    Args:
        evaluator: ChatEvaluator instance
        min_rating: Minimum average rating (1-5)
        output_file: Output file path
        
    Returns:
        Status message
    """
    if not output_file:
        output_file = os.path.join(os.path.dirname(evaluator.annotations_dir), "training_data.jsonl")
    
    success, message = evaluator.export_training_data(output_file, min_rating)
    return message
