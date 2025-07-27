#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import time

# Add src to Python path
sys.path.insert(0, '/app')

def main():
    """Docker entry point for Adobe Hackathon 1A"""
    print("=== Adobe Hackathon 1A - Document Analysis ===")
    print("Starting document processing...")
    
    # Docker paths (these are fixed in the container)
    input_dir = "/app/input"
    output_dir = "/app/output"
    model_dir = "/app/models"
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Model directory: {model_dir}")
    
    # Validate required directories exist
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory {input_dir} not found!")
        sys.exit(1)
    
    if not os.path.exists(model_dir):
        print(f"ERROR: Model directory {model_dir} not found!")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for PDF files
    pdf_files = list(Path(input_dir).glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    if len(pdf_files) == 0:
        print("WARNING: No PDF files found in input directory")
        sys.exit(0)
    
    try:
        # Import and initialize analyzer
        from inference import DocumentAnalyzer
        
        print("Loading trained model...")
        analyzer = DocumentAnalyzer(model_dir)
        print("Model loaded successfully!")
        
        # Process all PDFs
        print("Starting PDF processing...")
        start_time = time.time()
        
        analyzer.process_directory(input_dir, output_dir)
        
        total_time = time.time() - start_time
        print(f"Processing completed in {total_time:.2f} seconds")
        
        # Validate outputs
        output_files = list(Path(output_dir).glob("*.json"))
        print(f"Generated {len(output_files)} JSON output files")
        
        if len(output_files) != len(pdf_files):
            print("WARNING: Output file count doesn't match input file count")
        
        print("=== Processing Complete ===")
        
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
