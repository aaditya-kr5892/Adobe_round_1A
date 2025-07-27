import os
import json
import yaml
from pathlib import Path
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def ensure_directories(directories):
    """Create directories if they don't exist"""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def save_json(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_json(filepath):
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_pdf_json_pairs(data_dir):
    """Get matching PDF-JSON file pairs from directory"""
    data_path = Path(data_dir)
    pdf_files = list(data_path.glob("*.pdf"))
    
    pairs = []
    for pdf_file in pdf_files:
        json_file = pdf_file.with_suffix('.json')
        if json_file.exists():
            pairs.append((str(pdf_file), str(json_file)))
    
    return pairs

def validate_file_exists(filepath):
    """Check if file exists and is readable"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if not os.access(filepath, os.R_OK):
        raise PermissionError(f"Cannot read file: {filepath}")
    return True

def clean_text(text):
    """Clean and normalize text"""
    import re
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\-\.,;:!?()]', '', text)
    return text

def calculate_similarity(text1, text2):
    """Calculate simple text similarity"""
    import re
    
    # Normalize texts
    text1 = re.sub(r'\s+', ' ', text1.lower().strip())
    text2 = re.sub(r'\s+', ' ', text2.lower().strip())
    
    if text1 == text2:
        return 1.0
    
    # Check if one contains the other
    if text1 in text2 or text2 in text1:
        return 0.9
    
    # Simple word overlap similarity
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    if len(words1) == 0 or len(words2) == 0:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0

def format_execution_time(start_time, end_time):
    """Format execution time in readable format"""
    duration = end_time - start_time
    if duration < 60:
        return f"{duration:.2f} seconds"
    else:
        minutes = duration // 60
        seconds = duration % 60
        return f"{int(minutes)}m {seconds:.2f}s"
