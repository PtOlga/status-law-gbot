"""
Interface components for chat evaluation
"""

import gradio as gr
import pandas as pd
from src.analytics.chat_evaluator import ChatEvaluator
import json
import os
from typing import Dict, Any, List, Tuple

def get_evaluation_status(evaluator: ChatEvaluator) -> str:
    """
    Format evaluation status for display
    
    Args:
        evaluator: ChatEvaluator instance
        
    Returns:
        Formatted markdown string with status information
    """
    status = evaluator.get_evaluation_status()
    
    status_md = f"""
    ## Evaluation Status
    
    - **Total QA Pairs:** {status['total_qa_pairs']}
    - **Evaluated Pairs:** {status['evaluated_pairs']} ({status['evaluated_pairs']/max(1, status['total_qa_pairs'])*100:.1f}%)
    - **Unevaluated Pairs:** {status['unevaluated_pairs']}
    - **Evaluated Conversations:** {status['evaluated_conversations']}
    """
    
    return status_md

def get_qa_pairs_dataframe(evaluator: ChatEvaluator, show_evaluated: bool = False, limit: int = 50) -> pd.DataFrame:
    """
    Get QA pairs as a pandas DataFrame for display
    
    Args:
        evaluator: ChatEvaluator instance
        show_evaluated: Whether to show already evaluated pairs
        limit: Maximum number of pairs to return
        
    Returns:
        DataFrame with QA pairs
    """
    qa_pairs = evaluator.get_qa_pairs_for_evaluation(limit=200)  # Get more than needed for filtering
    annotations = evaluator.get_annotations()
    
    # Create set of evaluated conversation IDs
    evaluated_ids = set(a.get("conversation_id") for a in annotations)
    
    # Filter QA pairs based on show_evaluated parameter
    if not show_evaluated:
        qa_pairs = [pair for pair in qa_pairs if pair.get("conversation_id") not in evaluated_ids]
    
    # Limit the results
    qa_pairs = qa_pairs[:limit]
    
    # Create DataFrame
    if qa_pairs:
        df = pd.DataFrame(qa_pairs)
        
        # Add "Evaluated" column
        df["evaluated"] = df["conversation_id"].apply(lambda x: "Yes" if x in evaluated_ids else "No")
        
        # Select and rename columns for display
        display_df = df[["conversation_id", "question", "original_answer", "evaluated"]].copy()
        display_df = display_df.rename(columns={
            "conversation_id": "ID",
            "question": "Question",
            "original_answer": "Answer",
            "evaluated": "Evaluated"
        })
        
        # Truncate long text for better display
        display_df["Question"] = display_df["Question"].apply(lambda x: (x[:150] + "...") if len(x) > 150 else x)
        display_df["Answer"] = display_df["Answer"].apply(lambda x: (x[:150] + "...") if len(x) > 150 else x)
        
        return display_df
    
    # Return empty DataFrame if no pairs
    return pd.DataFrame(columns=["ID", "Question", "Answer", "Evaluated"])

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
    
    # Find the requested pair
    for pair in qa_pairs:
        if pair.get("conversation_id") == conversation_id:
            question = pair.get("question", "")
            original_answer = pair.get("original_answer", "")
            
            # Check if there's an existing annotation
            annotation = evaluator.get_annotation_by_conversation_id(conversation_id)
            
            if annotation:
                ratings = annotation.get("ratings", {})
                improved_answer = annotation.get("improved_answer", original_answer)
                notes = annotation.get("notes", "")
                
                # Get individual ratings with default value of 3
                accuracy = ratings.get("accuracy", 3)
                completeness = ratings.get("completeness", 3)
                relevance = ratings.get("relevance", 3)
                clarity = ratings.get("clarity", 3)
                legal_correctness = ratings.get("legal_correctness", 3)
                
                return (question, original_answer, improved_answer, 
                        accuracy, completeness, relevance, clarity, 
                        legal_correctness, notes)
            
            # Return default values for new evaluation
            return (question, original_answer, original_answer, 
                    3, 3, 3, 3, 3, "")  # Default rating of 3 for all criteria
    
    # Return empty values if conversation not found
    return ("", "", "", 3, 3, 3, 3, 3, "")

def save_evaluation(
    evaluator: ChatEvaluator,
    conversation_id: str,
    question: str,
    original_answer: str,
    improved_answer: str,
    accuracy: int,
    completeness: int,
    relevance: int,
    clarity: int,
    legal_correctness: int,
    notes: str
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
