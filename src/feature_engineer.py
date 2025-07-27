import pandas as pd
import numpy as np
import re
from utils import setup_logging

class FeatureEngineer:
    def __init__(self, config=None):
        self.logger = setup_logging()
        self.config = config or {}
        self.feature_config = self.config.get('features', {})
        
        # Get thresholds from config
        self.font_quantiles = self.feature_config.get('font_size_quantiles', [0.25, 0.75])
        self.position_thresholds = self.feature_config.get('position_thresholds', {
            'top_quarter': 0.25,
            'center_left': 0.2,
            'center_right': 0.8
        })
        self.text_thresholds = self.feature_config.get('text_length_thresholds', {
            'short': 50,
            'very_short': 20
        })
    
    def engineer_features(self, training_data):
        """Engineer features from raw training data"""
        self.logger.info("Starting feature engineering...")
        
        df = pd.DataFrame(training_data)
        
        if len(df) == 0:
            self.logger.warning("Empty training data provided")
            return df
        
        # Basic text features
        df = self._add_text_features(df)
        
        # Font features
        df = self._add_font_features(df)
        
        # Position features
        df = self._add_position_features(df)
        
        # Text pattern features
        df = self._add_pattern_features(df)
        
        # Capitalization features
        df = self._add_capitalization_features(df)
        
        # Relative features within pages
        df = self._add_relative_features(df)
        
        # Fill any NaN values
        df = df.fillna(0)
        
        self.logger.info(f"Feature engineering completed. Shape: {df.shape}")
        self.logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        
        return df
    
    def _add_text_features(self, df):
        """Add basic text-based features"""
        df['text_clean'] = df['text'].str.strip()
        df['word_count'] = df['text_clean'].str.split().str.len().fillna(0)
        df['char_count'] = df['text_clean'].str.len()
        
        return df
    
    def _add_font_features(self, df):
        """Add font-related features"""
        # Normalized font sizes within each page
        df['font_size_normalized'] = df.groupby('page')['font_size'].transform(
            lambda x: x / x.max() if x.max() > 0 else 0
        )
        
        # Font size categories
        df['is_large_font'] = df['font_size'] > df['font_size'].quantile(self.font_quantiles[1])
        df['is_small_font'] = df['font_size'] < df['font_size'].quantile(self.font_quantiles[0])
        
        # Global font size percentiles
        df['font_size_percentile'] = df['font_size'].rank(pct=True)
        
        return df
    
    def _add_position_features(self, df):
        """Add position-based features"""
        # Normalize positions
        df['y_position_normalized'] = df['bbox_y'] / df['page_height']
        df['x_position_normalized'] = df['bbox_x'] / df['page_width']
        
        # Position categories
        df['is_top_quarter'] = df['y_position_normalized'] < self.position_thresholds['top_quarter']
        df['is_centered'] = (
            (df['x_position_normalized'] > self.position_thresholds['center_left']) & 
            (df['x_position_normalized'] < self.position_thresholds['center_right'])
        )
        
        # Distance from page edges
        df['distance_from_left'] = df['x_position_normalized']
        df['distance_from_top'] = df['y_position_normalized']
        
        return df
    
    def _add_pattern_features(self, df):
        """Add text pattern features"""
        df['is_all_caps'] = df['text_clean'].str.isupper()
        df['starts_with_number'] = df['text_clean'].str.match(r'^\d+\.?\s*', na=False)
        df['contains_chapter'] = df['text_clean'].str.lower().str.contains(
            r'chapter|section|part|appendix', na=False
        )
        
        # Length categories
        df['is_short'] = df['char_count'] < self.text_thresholds['short']
        df['is_very_short'] = df['char_count'] < self.text_thresholds['very_short']
        
        # Special patterns
        df['contains_colon'] = df['text_clean'].str.contains(':', na=False)
        df['ends_with_period'] = df['text_clean'].str.endswith('.', na=False)
        df['is_single_word'] = df['word_count'] == 1
        
        return df
    
    def _add_capitalization_features(self, df):
        """Add capitalization-based features"""
        df['capital_ratio'] = df['text_clean'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        df['title_case'] = df['text_clean'].apply(lambda x: x.istitle())
        df['first_word_capitalized'] = df['text_clean'].str.match(r'^[A-Z]', na=False)
        
        return df
    
    def _add_relative_features(self, df):
        """Add relative features within each page"""
        # Relative font size within page
        df['relative_font_size'] = df.groupby('page')['font_size'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0
        )
        
        # Is this the largest font on the page?
        df['is_largest_font_on_page'] = df.groupby('page')['font_size'].transform(
            lambda x: x == x.max()
        )
        
        # Position ranking within page (top to bottom)
        df['position_rank_on_page'] = df.groupby('page')['bbox_y'].rank(method='dense')
        df['position_percentile_on_page'] = df.groupby('page')['bbox_y'].rank(pct=True)
        
        # Font size ranking within page
        df['font_size_rank_on_page'] = df.groupby('page')['font_size'].rank(
            method='dense', ascending=False
        )
        
        return df
    
    def get_feature_columns(self):
        """Get list of feature columns for training"""
        return [
            # Basic features
            'font_size', 'font_size_normalized', 'is_bold', 'is_italic',
            'text_length', 'word_count', 'char_count',
            
            # Position features
            'y_position_normalized', 'x_position_normalized',
            'distance_from_left', 'distance_from_top',
            'is_top_quarter', 'is_centered',
            
            # Font features
            'is_large_font', 'is_small_font', 'font_size_percentile',
            
            # Pattern features
            'is_all_caps', 'starts_with_number', 'contains_chapter',
            'is_short', 'is_very_short', 'contains_colon', 'ends_with_period',
            'is_single_word',
            
            # Capitalization features
            'capital_ratio', 'title_case', 'first_word_capitalized',
            
            # Relative features
            'relative_font_size', 'is_largest_font_on_page',
            'position_rank_on_page', 'position_percentile_on_page',
            'font_size_rank_on_page'
        ]
    
    def get_feature_importance_summary(self, model, feature_columns):
        """Get feature importance summary from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.logger.info("Top 10 most important features:")
            for _, row in importance_df.head(10).iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return importance_df
        
        return None
