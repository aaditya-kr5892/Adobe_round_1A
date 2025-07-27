import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import pickle
import pandas as pd
import numpy as np
from utils import setup_logging, ensure_directories
import time

class ModelTrainer:
    def __init__(self, config=None):
        self.logger = setup_logging()
        self.config = config or {}
        self.model_config = self.config.get('model', {})
        self.training_config = self.config.get('training', {})
        
        # Initialize components
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.training_history = {}
    
    def train_model(self, df, feature_columns):
        """Train XGBoost model with comprehensive evaluation"""
        self.logger.info("Starting model training...")
        start_time = time.time()
        
        self.feature_columns = feature_columns
        
        # Prepare features and labels
        X = df[feature_columns].fillna(0)
        y = df['label']
        
        self.logger.info(f"Training data shape: {X.shape}")
        self.logger.info(f"Feature columns: {len(feature_columns)}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        self.logger.info(f"Label classes: {list(self.label_encoder.classes_)}")
        
        # Split data
        test_size = self.training_config.get('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=42, 
            stratify=y_encoded
        )
        
        self.logger.info(f"Training set size: {len(X_train)}")
        self.logger.info(f"Test set size: {len(X_test)}")
        
        # Initialize XGBoost with config parameters
        self.model = xgb.XGBClassifier(
            max_depth=self.model_config.get('max_depth', 6),
            n_estimators=self.model_config.get('n_estimators', 100),
            learning_rate=self.model_config.get('learning_rate', 0.1),
            random_state=self.model_config.get('random_state', 42),
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        # Train model with evaluation
        eval_set = [(X_train, y_train), (X_test, y_test)]
        
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Store training history
        self.training_history = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'training_time': time.time() - start_time,
            'feature_count': len(feature_columns),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # Log results
        self.logger.info(f"Training completed in {self.training_history['training_time']:.2f} seconds")
        self.logger.info(f"Training Accuracy: {train_score:.4f}")
        self.logger.info(f"Test Accuracy: {test_score:.4f}")
        
        # Detailed classification report
        self.logger.info("\nTest Set Classification Report:")
        report = classification_report(
            y_test, y_test_pred, 
            target_names=self.label_encoder.classes_,
            zero_division=0
        )
        print(report)
        
        # Confusion matrix
        self.logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_test_pred)
        cm_df = pd.DataFrame(cm, 
                           index=self.label_encoder.classes_, 
                           columns=self.label_encoder.classes_)
        print(cm_df)
        
        # Cross-validation
        if len(X_train) > 100:  # Only if we have enough data
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=3, scoring='accuracy')
            self.logger.info(f"Cross-validation scores: {cv_scores}")
            self.logger.info(f"CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.model
    
    def optimize_hyperparameters(self, df, feature_columns):
        """Quick hyperparameter optimization (optional)"""
        from sklearn.model_selection import GridSearchCV
        
        self.logger.info("Starting hyperparameter optimization...")
        
        X = df[feature_columns].fillna(0)
        y = self.label_encoder.fit_transform(df['label'])
        
        # Simple grid search
        param_grid = {
            'max_depth': [4, 6, 8],
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.05, 0.1, 0.15]
        }
        
        grid_search = GridSearchCV(
            xgb.XGBClassifier(random_state=42),
            param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.logger.info(f"Best parameters: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def save_model(self, model_dir):
        """Save trained model and related components"""
        ensure_directories([model_dir])
        
        model_path = f"{model_dir}/xgboost_model.pkl"
        encoder_path = f"{model_dir}/label_encoder.pkl"
        features_path = f"{model_dir}/feature_columns.pkl"
        history_path = f"{model_dir}/training_history.pkl"
        
        # Save model
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save label encoder
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save feature columns
        with open(features_path, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        
        # Save training history
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history, f)
        
        self.logger.info(f"Model saved to {model_dir}")
        
        # Log model size
        import os
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        self.logger.info(f"Model size: {model_size:.2f} MB")
        
        if model_size > 200:
            self.logger.warning("Model size exceeds 200MB limit!")
    
    def load_model(self, model_dir):
        """Load trained model and related components"""
        model_path = f"{model_dir}/xgboost_model.pkl"
        encoder_path = f"{model_dir}/label_encoder.pkl"
        features_path = f"{model_dir}/feature_columns.pkl"
        history_path = f"{model_dir}/training_history.pkl"
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load label encoder
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load feature columns
            with open(features_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            # Load training history
            try:
                with open(history_path, 'rb') as f:
                    self.training_history = pickle.load(f)
            except FileNotFoundError:
                self.training_history = {}
            
            self.logger.info(f"Model loaded from {model_dir}")
            self.logger.info(f"Model classes: {list(self.label_encoder.classes_)}")
            self.logger.info(f"Feature count: {len(self.feature_columns)}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
    
    def predict_batch(self, X):
        """Make predictions on batch of features"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        X_features = X[self.feature_columns].fillna(0)
        predictions = self.model.predict(X_features)
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        # Get prediction probabilities for confidence
        probabilities = self.model.predict_proba(X_features)
        confidence_scores = np.max(probabilities, axis=1)
        
        return predicted_labels, confidence_scores
    
    def get_model_info(self):
        """Get model information summary"""
        if not self.model:
            return "Model not loaded"
        
        info = {
            'model_type': 'XGBoost Classifier',
            'classes': list(self.label_encoder.classes_),
            'feature_count': len(self.feature_columns),
            'training_history': self.training_history
        }
        
        return info
