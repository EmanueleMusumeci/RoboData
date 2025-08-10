import os
import json
import pandas as pd
import re
from collections import defaultdict
import statistics


def enhance_triple(triple_text: str, node_mapping: dict, edge_mapping: dict) -> str:
    """
    Replace node/relation IDs in a single triple with "LABEL (ID)" format.
    
    Parameters:
    - triple_text: str - triple in format (Q108297, P144, Q1452613)
    - node_mapping: dict - mapping of node IDs to labels
    - edge_mapping: dict - mapping of edge types to labels
    
    Returns:
    - str: enhanced triple with labels
    """
    # Pattern to match triples like (Q108297, P144, Q1452613)
    triple_pattern = r'\(([^,\)]+),\s*([^,\)]+),\s*([^,\)]+)\)'
    
    match = re.search(triple_pattern, triple_text)
    if not match:
        return triple_text
    
    node1_id = match.group(1).strip()
    relation_id = match.group(2).strip()
    node2_id = match.group(3).strip()
    
    # Get labels for nodes and relation
    node1_label = node_mapping.get(node1_id, node1_id)
    relation_label = edge_mapping.get(relation_id, relation_id)
    node2_label = node_mapping.get(node2_id, node2_id)
    
    # Format as "LABEL (ID)" but use angle brackets for triples
    node1_enhanced = f"{node1_label} ({node1_id})" if node1_label != node1_id else node1_id
    relation_enhanced = f"{relation_label} ({relation_id})" if relation_label != relation_id else relation_id
    node2_enhanced = f"{node2_label} ({node2_id})" if node2_label != node2_id else node2_id
    
    return f"<{node1_enhanced}, {relation_enhanced}, {node2_enhanced}>"
def improve_support_sets_with_singles(support_content: str, node_mapping: dict, edge_mapping: dict, support_num: str) -> str:
    """
    Improve support sets handling both triples and single nodes.
    
    Parameters:
    - support_content: str - the support set content
    - node_mapping: dict - mapping of node IDs to labels  
    - edge_mapping: dict - mapping of edge types to labels
    - support_num: str - support set number
    
    Returns:
    - str: improved support set with proper formatting
    """
    # Check if it's a triple (Q123, P456, Q789) pattern
    triple_pattern = r'\(([^,\)]+),\s*([^,\)]+),\s*([^,\)]+)\)'
    triple_matches = re.findall(triple_pattern, support_content)
    
    if triple_matches:
        # Handle triples
        enhanced_triples = []
        for match in triple_matches:
            node1_id, relation_id, node2_id = [item.strip() for item in match]
            triple_text = f"({node1_id}, {relation_id}, {node2_id})"
            enhanced_triple = enhance_triple(triple_text, node_mapping, edge_mapping)
            enhanced_triples.append(enhanced_triple)
        return f"{support_num} " + ", ".join(enhanced_triples)
    else:
        # Check if it's a single node (Q123) pattern
        single_node_pattern = r'\(([^,\)]+)\)'
        single_matches = re.findall(single_node_pattern, support_content)
        
        if single_matches:
            # Handle single nodes - try to extract description from the text
            enhanced_nodes = []
            for node_id in single_matches:
                node_id = node_id.strip()
                node_label = node_mapping.get(node_id, node_id)
                
                # Try to extract description from the support content
                # Look for patterns like "Q123 - description text"
                desc_pattern = rf'{re.escape(node_id)}\s*-\s*(.+?)(?:\.|$)'
                desc_match = re.search(desc_pattern, support_content)
                
                if desc_match:
                    description = desc_match.group(1).strip()
                    enhanced_node = f"{node_label} ({node_id}): '{description}'"
                else:
                    enhanced_node = f"{node_label} ({node_id})"
                
                enhanced_nodes.append(enhanced_node)
            return f"{support_num} " + ", ".join(enhanced_nodes)
        else:
            # If no specific pattern found, just add the support number
            return f"{support_num} {support_content}"


def parse_final_answer(final_answer: str, node_mapping: dict, edge_mapping: dict) -> tuple[str, str]:
    """
    Parse final answer to extract sentences and support sets separately.
    
    Parameters:
    - final_answer: str - the final answer text
    - node_mapping: dict - mapping of node IDs to labels
    - edge_mapping: dict - mapping of edge types to labels
    
    Returns:
    - tuple: (formatted_sentences, enhanced_support_sets)
    """
    if not final_answer:
        return "", ""
    
    # Split by lines and process
    lines = final_answer.split('\n')
    sentences = []
    support_sets = []
    
    sentence_counter = 1
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for sentence lines
        sentence_match = re.match(r'SENTENCE\s+\d+:\s*(.+)', line)
        if sentence_match:
            sentence_text = sentence_match.group(1).strip()
            # Ensure sentence ends with period
            if not sentence_text.endswith('.'):
                sentence_text += '.'
            sentences.append(f"{sentence_text} ({sentence_counter})")
            sentence_counter += 1
            continue
        
        # Check for support set lines
        support_match = re.match(r'SUPPORT\s+SET\s+(\d+):\s*(.+)', line)
        if support_match:
            support_num = f"({str(int(support_match.group(1)))})"  # Format as (1), (2), etc.
            support_content = support_match.group(2).strip()
            
            # Use the new function to handle both triples and single nodes
            enhanced_support = improve_support_sets_with_singles(support_content, node_mapping, edge_mapping, support_num)
            support_sets.append(enhanced_support)
    
    # Format output
    formatted_sentences = " ".join(sentences)
    formatted_support_sets = ", ".join(support_sets)
    
    return formatted_sentences, formatted_support_sets


def load_knowledge_graph(kg_path: str) -> tuple[dict, dict]:
    """
    Load knowledge graph from JSON file and return nodes/edges mapping.
    
    Parameters:
    - kg_path: str - path to knowledge_graph.json file
    
    Returns:
    - tuple: (node_mapping, edge_mapping)
    """
    if not os.path.exists(kg_path):
        return {}, {}
    
    try:
        with open(kg_path, 'r') as f:
            kg_data = json.load(f)
        
        # Create node ID to label mapping
        node_mapping = {}
        for node in kg_data.get('nodes', []):
            node_id = node.get('id', '')
            label = node.get('label', node_id)
            node_mapping[node_id] = label
        
        # Create edge mapping for relationship types
        edge_mapping = {}
        for edge in kg_data.get('edges', []):
            edge_type = edge.get('type', '')
            label = edge.get('label', edge_type)
            edge_mapping[edge_type] = label
            
        return node_mapping, edge_mapping
    except Exception as e:
        print(f"Error loading knowledge graph from {kg_path}: {e}")
        return {}, {}


def load_experiment_statistics(exp_path: str) -> dict | None:
    """
    Load the latest statistics JSON file from an experiment directory.
    
    Parameters:
    - exp_path: str - path to experiment directory
    
    Returns:
    - dict: statistics data or None if no file found
    """
    if not os.path.isdir(exp_path):
        return None
    
    # Look for the latest statistics JSON file
    json_files = sorted(
        [f for f in os.listdir(exp_path) if f.endswith(".json") and "statistics" in f],
        reverse=True
    )
    
    if not json_files:
        return None
    
    latest_json = json_files[0]
    try:
        with open(os.path.join(exp_path, latest_json)) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {latest_json} in {exp_path}: {e}")
        return None


def extract_batch_table(
    batch_letter: str,
    root_dir: str,
    output_csv: str | None = None
):
    """
    Creates a detailed table for batches A, B, and D with standard columns.
    
    Parameters:
    - batch_letter: str - batch identifier (e.g., 'A', 'B', 'D')
    - root_dir: str - root directory containing batch subdirectories (batch_a, batch_b, batch_d)
    - output_csv: str - output filename (if None, auto-generate)
    """
    if output_csv is None:
        output_csv = f"batch_Q{batch_letter}_table.csv"
    
    # Map batch letters to folder names
    batch_folder_map = {'A': 'batch_a', 'B': 'batch_b', 'D': 'batch_d'}
    batch_folder = batch_folder_map.get(batch_letter.upper())
    
    if batch_folder is None:
        raise ValueError(f"Unknown batch letter: {batch_letter}")
    
    # Auto-discover experiment directories for this batch
    experiment_dirs = []
    batch_path = os.path.join(root_dir, batch_folder)
    if os.path.isdir(batch_path):
        for item in os.listdir(batch_path):
            if os.path.isdir(os.path.join(batch_path, item)) and item.startswith(f'Q{batch_letter}'):
                experiment_dirs.append(item)
    
    experiment_dirs.sort()
    records = []

    for exp in experiment_dirs:
        exp_path = os.path.join(batch_path, exp)  # Use batch_path instead of root_dir
        data = load_experiment_statistics(exp_path)
        
        if data is None:
            print(f"No statistics found for {exp}")
            continue

        # Extract basic information
        experiment_id = exp  # Use directory name as experiment ID
        query = data.get("query", "")
        final_answer = data.get("final_answer", "")
        
        # Extract batch and query number from experiment ID (directory name)
        # Pattern: QA01 -> Batch: A, Query Number: 1
        id_match = re.match(r'Q([A-Z])(\d+)', experiment_id)
        if id_match:
            batch = id_match.group(1)
            query_number = str(int(id_match.group(2)))  # Convert to int then str to remove leading zeros
        else:
            batch = batch_letter
            query_number = "0"
        
        # Load knowledge graph for enhanced triples
        kg_path = os.path.join(exp_path, "knowledge_graph.json")
        node_mapping, edge_mapping = load_knowledge_graph(kg_path)
        
        # Parse final answer to separate sentences and support sets
        formatted_sentences, formatted_support_sets = parse_final_answer(final_answer, node_mapping, edge_mapping)
        
        # Calculate total tool calls
        tool_stats = data.get("tool_statistics", {})
        total_tool_calls = sum(stats.get("call_count", 0) for stats in tool_stats.values())
        
        # Extract inference statistics
        inference_stats = data.get("inference_statistics", {})
        orchestrator_stats = inference_stats.get("orchestrator", {})
        metacognition_stats = inference_stats.get("metacognition", {})
        
        # For batch D, include metacognition in totals
        if batch_letter == 'D' and metacognition_stats:
            total_tokens = (orchestrator_stats.get("total_tokens", 0) + 
                          metacognition_stats.get("total_tokens", 0))
            total_inference_time = (orchestrator_stats.get("total_time", 0) + 
                                  metacognition_stats.get("total_time", 0))
        else:
            total_tokens = orchestrator_stats.get("total_tokens", 0)
            total_inference_time = orchestrator_stats.get("total_time", 0)
        
        # Extract iterations (orchestrator steps)
        iterations = orchestrator_stats.get("count", 0)
        
        # Extract knowledge graph statistics
        graph_stats = data.get("graph_statistics", {})
        nodes_added = graph_stats.get("nodes_added", 0)
        edges_added = graph_stats.get("edges_added", 0)
        
        row = {
            "Batch": batch,
            "Query Number": query_number,
            "Query ID": experiment_id,
            "Query Answer": formatted_sentences,
            "Support Sets": formatted_support_sets,
            "Success": "",  # Leave empty as requested
            "Iterations": iterations,
            "Total Tool Calls": total_tool_calls,
            "Total Tokens": total_tokens,
            "Total Inference Time": round(total_inference_time, 2),
            "KG Nodes Added": nodes_added,
            "KG Edges Added": edges_added
        }
        
        records.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(records).fillna(0)

    # Save with wider column formatting for Query Answer
    df.to_csv(output_csv, index=False)
    print(f"[✓] Batch Q{batch_letter} table with {len(df)} experiments saved to {output_csv}")
    return df


def extract_consistency_test_table(
    root_dir: str,
    output_csv: str = "batch_QC_table.csv"
):
    """
    Creates a table for QC01 consistency test showing Min, Max, and Avg values.
    
    Parameters:
    - root_dir: str - root directory containing batch subdirectories
    - output_csv: str - output filename
    """
    # QC experiments might be in the old structure or need to be found
    # First try to find QC01 in the root directory, then in batch folders
    exp_path = os.path.join(root_dir, "QC01_consistency_test_canada_capital")
    
    if not os.path.isdir(exp_path):
        # Try looking in batch_c folder if it exists
        batch_c_path = os.path.join(root_dir, "batch_c")
        if os.path.isdir(batch_c_path):
            exp_path = os.path.join(batch_c_path, "QC01_consistency_test_canada_capital")
        
        if not os.path.isdir(exp_path):
            print(f"QC01 experiment directory not found in {root_dir} or batch_c")
            return None
    
    # Load all statistics files
    json_files = [f for f in os.listdir(exp_path) if f.endswith(".json") and "statistics" in f]
    
    if not json_files:
        print(f"No statistics files found in {exp_path}")
        return None
    
    all_data = []
    for json_file in json_files:
        try:
            with open(os.path.join(exp_path, json_file)) as f:
                data = json.load(f)
                all_data.append(data)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue
    
    if not all_data:
        print("No valid statistics data found")
        return None
    
    # Calculate metrics for each run
    metrics = []
    query_answers = []
    for data in all_data:
        # Collect query answers for analysis
        final_answer = data.get("final_answer", "")
        query_answers.append(final_answer)
        
        # Calculate total tool calls
        tool_stats = data.get("tool_statistics", {})
        total_tool_calls = sum(stats.get("call_count", 0) for stats in tool_stats.values())
        
        # Extract inference statistics
        inference_stats = data.get("inference_statistics", {})
        orchestrator_stats = inference_stats.get("orchestrator", {})
        
        # Extract iterations and other metrics
        iterations = orchestrator_stats.get("count", 0)
        total_tokens = orchestrator_stats.get("total_tokens", 0)
        total_inference_time = orchestrator_stats.get("total_time", 0)
        
        # Extract knowledge graph statistics
        graph_stats = data.get("graph_statistics", {})
        nodes_added = graph_stats.get("nodes_added", 0)
        edges_added = graph_stats.get("edges_added", 0)
        
        metrics.append({
            "Iterations": iterations,
            "Total Tool Calls": total_tool_calls,
            "Total Tokens": total_tokens,
            "Total Inference Time": total_inference_time,
            "KG Nodes Added": nodes_added,
            "KG Edges Added": edges_added
        })
    
    # Calculate Min, Max, Avg for each metric
    records = []
    metric_names = ["Iterations", "Total Tool Calls", "Total Tokens", 
                   "Total Inference Time", "KG Nodes Added", "KG Edges Added"]
    
    # Add query answer analysis
    # For consistency test, we want to see if all answers are the same
    unique_answers = list(set(query_answers))
    if len(unique_answers) == 1:
        query_consistency = f"Consistent: {unique_answers[0][:100]}{'...' if len(unique_answers[0]) > 100 else ''}"
    else:
        query_consistency = f"Inconsistent: {len(unique_answers)} different answers"
    
    # Add query answer row
    records.append({
        "Metric": "Query Answer",
        "Min": query_consistency,
        "Max": query_consistency,
        "Avg": query_consistency
    })
    
    for metric_name in metric_names:
        values = [m[metric_name] for m in metrics]
        
        record = {
            "Metric": metric_name,
            "Min": round(min(values), 2),
            "Max": round(max(values), 2),
            "Avg": round(statistics.mean(values), 2)
        }
        records.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"[✓] QC consistency test table with {len(metrics)} runs saved to {output_csv}")
    return df


def extract_qa_batch_table(root_dir: str, output_csv: str = "batch_QA_table.csv"):
    """Extract table for QA batch experiments."""
    return extract_batch_table('A', root_dir, output_csv)


def extract_qb_batch_table(root_dir: str, output_csv: str = "batch_QB_table.csv"):
    """Extract table for QB batch experiments."""
    return extract_batch_table('B', root_dir, output_csv)


def extract_qd_batch_table(root_dir: str, output_csv: str = "batch_QD_table.csv"):
    """Extract table for QD batch experiments."""
    return extract_batch_table('D', root_dir, output_csv)


def extract_overview_table(
    root_dir: str,
    output_csv: str = "experiment_overview.csv"
):
    """
    Creates an overview table with Query ID and Query text.
    
    Parameters:
    - root_dir: str - root directory containing batch subdirectories (batch_a, batch_b, batch_d)
    - output_csv: str - output filename for the table
    """
    # Auto-discover experiment directories in batch folders
    experiment_dirs = []
    batch_folders = ['batch_a', 'batch_b', 'batch_d']
    
    for batch_folder in batch_folders:
        batch_path = os.path.join(root_dir, batch_folder)
        if os.path.isdir(batch_path):
            for item in os.listdir(batch_path):
                item_path = os.path.join(batch_path, item)
                if os.path.isdir(item_path) and re.match(r'Q[A-Z]\d+', item):
                    experiment_dirs.append((batch_folder, item))
    
    # Also check for any experiments in the root directory (like QC01)
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if os.path.isdir(item_path) and re.match(r'Q[A-Z]\d+', item):
            experiment_dirs.append(('', item))
    
    experiment_dirs.sort(key=lambda x: x[1])  # Sort by experiment name
    records = []

    for batch_folder, exp in experiment_dirs:
        if batch_folder:
            exp_path = os.path.join(root_dir, batch_folder, exp)
        else:
            exp_path = os.path.join(root_dir, exp)
            
        data = load_experiment_statistics(exp_path)
        
        if data is None:
            continue

        experiment_id = exp  # Use directory name as experiment ID
        query = data.get("query", "")
        
        # Extract batch identifier and query number (Q[A-Z])
        batch_match = re.match(r'Q([A-Z])(\d+)', experiment_id)
        if batch_match:
            batch = batch_match.group(1)
            query_number = str(int(batch_match.group(2)))  # Convert to int then str to remove leading zeros
        else:
            batch = "Unknown"
            query_number = "0"
        
        record = {
            "Batch": batch,
            "Query Number": query_number,
            "Experiment ID": experiment_id,
            "Query": query
        }
        
        records.append(record)

    # Create overview DataFrame
    df = pd.DataFrame(records).fillna("")
    df = df.sort_values(["Batch", "Query Number"])

    # Save overview table with all columns
    df.to_csv(output_csv, index=False)
    print(f"[✓] Overview table saved to {output_csv}")
    
    return df


# Example usage
if __name__ == "__main__":
    # Configuration
    ROOT = "experiments"
    
    print("Extracting experiment tables with updated structure...")
    
    # Extract overview table (all experiments)
    print("\n1. Creating overview table...")
    overview_df = extract_overview_table(
        root_dir=ROOT,
        output_csv="experiment_overview.csv"
    )
    
    # Count experiments by batch
    batch_counts = overview_df['Batch'].value_counts().sort_index()
    print(f"Found {len(overview_df)} experiments across {len(batch_counts)} batches:")
    for batch, count in batch_counts.items():
        print(f"  - Batch Q{batch}: {count} experiments")
    
    # Extract tables for each batch
    print("\n2. Creating batch-specific tables...")
    
    # QA batch (A)
    if 'A' in batch_counts.index:
        print("Creating QA batch table...")
        qa_df = extract_qa_batch_table(ROOT)
        print(f"QA batch: {len(qa_df)} experiments")
    
    # QB batch (B)
    if 'B' in batch_counts.index:
        print("Creating QB batch table...")
        qb_df = extract_qb_batch_table(ROOT)
        print(f"QB batch: {len(qb_df)} experiments")
    
    # QC batch (C) - Special consistency test format
    print("Creating QC consistency test table...")
    qc_df = extract_consistency_test_table(ROOT)
    if qc_df is not None:
        print(f"QC consistency test: {len(qc_df)} metrics analyzed")
    
    # QD batch (D) - With metacognition
    if 'D' in batch_counts.index:
        print("Creating QD batch table (with metacognition)...")
        qd_df = extract_qd_batch_table(ROOT)
        print(f"QD batch: {len(qd_df)} experiments")
    
    print(f"\n[✓] All tables generated successfully!")
    print(f"    - Overview: experiment_overview.csv")
    for batch in batch_counts.index:
        if batch == 'C':
            print(f"    - Batch Q{batch}: batch_Q{batch}_table.csv (Min/Max/Avg format)")
        else:
            print(f"    - Batch Q{batch}: batch_Q{batch}_table.csv")
    
    # Display preview of overview
    print(f"\nOverview table preview:")
    print(overview_df.head(10))
