#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append('src')

from data_processor import PDFProcessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from utils import setup_logging, load_config, get_pdf_json_pairs, ensure_directories
import pandas as pd

def main():
    """Enhanced training pipeline with diagnostics"""
    logger = setup_logging()
    logger.info("=== Adobe Hackathon 1A Training Pipeline (Enhanced) ===")
    
    # Load configuration
    try:
        config = load_config("config/config.yaml")
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        config = {}
    
    # Setup paths
    data_dir = config.get('paths', {}).get('raw_data', 'data/raw')
    model_dir = config.get('paths', {}).get('models', 'data/models')
    
    ensure_directories([model_dir, 'data/processed'])
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Model directory: {model_dir}")
    
    # Get PDF-JSON pairs
    pdf_json_pairs = get_pdf_json_pairs(data_dir)
    logger.info(f"Found {len(pdf_json_pairs)} PDF-JSON pairs")
    
    if len(pdf_json_pairs) == 0:
        logger.error("No PDF-JSON pairs found. Please check your data directory.")
        sys.exit(1)
    
    # Initialize components
    processor = PDFProcessor(config)
    engineer = FeatureEngineer(config)
    trainer = ModelTrainer(config)
    
    # Process training data with detailed logging
    logger.info("Processing PDF-JSON pairs for training...")
    start_time = time.time()
    
    # Use first 100 pairs for training
    training_pairs = pdf_json_pairs[:min(100, len(pdf_json_pairs))]
    logger.info(f"Using {len(training_pairs)} files for training")
    
    all_training_data = processor.process_multiple_pdfs(training_pairs)
    
    if len(all_training_data) == 0:
        logger.error("No training data generated. Please check your PDF-JSON files.")
        sys.exit(1)
    
    processing_time = time.time() - start_time
    logger.info(f"Data processing completed in {processing_time:.2f}s")
    
    # Engineer features
    logger.info("Engineering features...")
    df = engineer.engineer_features(all_training_data)
    feature_columns = engineer.get_feature_columns()
    
    logger.info(f"Feature engineering completed. Dataset shape: {df.shape}")
    logger.info(f"Feature columns: {len(feature_columns)}")
    
    # Detailed label analysis
    label_counts = df['label'].value_counts()
    logger.info("=== DETAILED LABEL ANALYSIS ===")
    logger.info(f"Total samples: {len(df)}")
    
    for label, count in label_counts.items():
        percentage = (count / len(df) * 100)
        logger.info(f"  {label}: {count} samples ({percentage:.1f}%)")
    
    # Check if we have enough heading samples for good training
    heading_samples = sum(count for label, count in label_counts.items() if label in ['H1', 'H2', 'H3'])
    logger.info(f"Total heading samples: {heading_samples}")
    
    if heading_samples < 500:
        logger.warning("⚠️  LOW HEADING SAMPLES - Model may not learn headings well!")
        logger.warning("Consider:")
        logger.warning("  1. Lowering similarity threshold further")
        logger.warning("  2. Checking if JSON text matches PDF text")
        logger.warning("  3. Adding more training files")
    else:
        logger.info(f"✅ Good heading sample count: {heading_samples}")
    
    # Show sample headings found during training
    sample_headings = df[df['label'].isin(['H1', 'H2', 'H3'])].head(20)
    logger.info("=== SAMPLE HEADINGS FOUND IN TRAINING ===")
    for _, row in sample_headings.iterrows():
        logger.info(f"  {row['label']}: '{row['text'][:60]}...' (page {row['page']})")
    
    # Save processed data
    processed_data_path = "data/processed/training_data.pkl"
    df.to_pickle(processed_data_path)
    logger.info(f"Processed data saved to {processed_data_path}")
    
    # Train model
    logger.info("=== STARTING MODEL TRAINING ===")
    model = trainer.train_model(df, feature_columns)
    
    # Save model
    trainer.save_model(model_dir)
    
    # Feature importance analysis
    importance_df = engineer.get_feature_importance_summary(model, feature_columns)
    if importance_df is not None:
        importance_path = "data/processed/feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
    
    # Final training summary
    training_info = trainer.get_model_info()
    logger.info("=== TRAINING COMPLETED ===")
    logger.info(f"Model classes: {training_info['classes']}")
    logger.info(f"Feature count: {training_info['feature_count']}")
    
    # Quick test on a sample file
    if len(pdf_json_pairs) > 100:
        logger.info("=== QUICK TEST ON UNUSED FILE ===")
        test_pdf = pdf_json_pairs[100][0]  # First unused file
        
        try:
            from inference import DocumentAnalyzer
            analyzer = DocumentAnalyzer(model_dir, config)
            result = analyzer.analyze_pdf(test_pdf)
            
            logger.info(f"Test file: {Path(test_pdf).name}")
            logger.info(f"  Title: '{result['title'][:50]}...'")
            logger.info(f"  Headings found: {len(result['outline'])}")
            
            for i, heading in enumerate(result['outline'][:5]):
                logger.info(f"    {heading['level']}: '{heading['text'][:40]}...' (page {heading['page']})")
            
        except Exception as e:
            logger.error(f"Quick test failed: {e}")
    
    logger.info("🎉 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
