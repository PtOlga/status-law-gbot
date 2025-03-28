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

def load_qa_pair_for_evaluation(evaluator: ChatEvaluator, conversation_id: str) -> Tuple[str, str, Dict, str]:
    """
    Load a QA pair for evaluation
    
    Args:
        evaluator: ChatEvaluator instance
        conversation_id: ID of the conversation to load
        
    Returns:
        Tuple of (question, original_answer, existing_ratings, notes)
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
                existing_ratings = annotation.get("ratings", {})
                improved_answer = annotation.get("improved_answer", original_answer)
                notes = annotation.get("notes", "")
                return question, original_answer, improved_answer, existing_ratings, notes
            
            return question, original_answer, original_answer, {}, ""
    
    return "", "", "", {}, ""

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

def export_training_data_action(min_rating: int, output_file: str, evaluator: ChatEvaluator) -> str:
    """
    Action for exporting training data
    
    Args:
        min_rating: Minimum average rating (1-5)
        output_file: Output file path
        evaluator: ChatEvaluator instance
        
    Returns:
        Status message
    """
    if not output_file:
        output_file = os.path.join(os.path.dirname(evaluator.annotations_dir), "training_data.jsonl")
    
    success, message = evaluator.export_training_data(output_file, min_rating)
    return message



# Create a Gradio interface
with gr.Blocks() as interface:
    # Load the ChatEvaluator
    chat_evaluator = ChatEvaluator()
    
    # Evaluation status
    status = get_evaluation_status(chat_evaluator)
    status_display = gr.Markdown(value=status)
    
    # QA pairs table
    show_evaluated = gr.Checkbox(label="Show evaluated pairs", value=False)
    limit = gr.Slider(label="Number of pairs to show", minimum=1, maximum=100, value=50)
    qa_pairs_table = gr.DataFrame(value=get_qa_pairs_dataframe(chat_evaluator, show_evaluated.value, limit.value))
    
    # QA pair evaluation
    conversation_id = gr.Textbox(label="Conversation ID")
    load_btn = gr.Button("Load QA Pair")
    question = gr.Textbox(label="Question", interactive=False)
    original_answer = gr.Textbox(label="Original Answer", interactive=False)
    improved_answer = gr.Textbox(label="Improved Answer")
    accuracy = gr.Slider(label="Accuracy", minimum=1, maximum=5, value=3)
    completeness = gr.Slider(label="Completeness", minimum=1, maximum=5, value=3)
    relevance = gr.Slider(label="Relevance", minimum=1, maximum=5, value=3)
    clarity = gr.Slider(label="Clarity", minimum=1, maximum=5, value=3)
    legal_correctness = gr.Slider(label="Legal Correctness", minimum=1, maximum=5, value=3)
    notes = gr.Textbox(label="Notes")
    save_btn = gr.Button("Save Evaluation")
    
    # Evaluation report
    report_html = generate_evaluation_report_html(chat_evaluator)
    report_display = gr.HTML(value=report_html)
    
    # Export training data
    min_rating = gr.Slider(label="Minimum Average Rating", minimum=1, maximum=5, value=3)
    export_path = gr.Textbox(label="Output File Path")
    export_btn = gr.Button("Export Training Data")
    export_status = gr.Textbox(label="Export Status", interactive=False)
    
    # Event listeners
    load_btn.click(
        fn=lambda cid: load_qa_pair_for_evaluation(chat_evaluator, cid),
        inputs=[conversation_id],
        outputs=[question, original_answer, improved_answer, accuracy, completeness, relevance, clarity, legal_correctness, notes]
    )
    
    save_btn.click(
        fn=lambda cid, q, oa, ia, acc, comp, rel, cl, lc, n: save_evaluation(chat_evaluator, cid, q, oa, ia, acc, comp, rel, cl, lc, n),
        inputs=[conversation_id, question, original_answer, improved_answer, accuracy, completeness, relevance, clarity, legal_correctness, notes],
        outputs=[save_btn]
    )
    
    show_evaluated.change(
        fn=lambda se, l: get_qa_pairs_dataframe(chat_evaluator, se, l),
        inputs=[show_evaluated, limit],
        outputs=[qa_pairs_table]
    )
    
    limit.change(
        fn=lambda se, l: get_qa_pairs_dataframe(chat_evaluator, se, l),
        inputs=[show_evaluated, limit],
        outputs=[qa_pairs_table]
    )
    
    # Export training data
    export_btn.click(
        fn=lambda min_r, path: export_training_data_action(min_r, path, chat_evaluator),
        inputs=[min_rating, export_path],
        outputs=[export_status]
    )

# Launch the interface
interface.launch()

