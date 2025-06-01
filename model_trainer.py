import argparse
import logging
from emotion_classifier import EmotionClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model():
    """Train the emotion classification model."""
    logger.info("Starting model training")
    
    # Initialize classifier
    classifier = EmotionClassifier()
    
    # Prepare data
    classifier.prepare_data()
    
    # Train model
    classifier.train_model()
    
    # Evaluate model
    metrics = classifier.get_evaluation_metrics()
    
    # Print evaluation results
    logger.info(f"Model training completed with accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1 score (weighted): {metrics['f1_weighted']:.4f}")
    logger.info(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
    logger.info(f"Recall (weighted): {metrics['recall_weighted']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Train the DistilBERT emotion classifier')
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train a new model'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Evaluate the existing model'
    )
    
    args = parser.parse_args()
    
    if args.train:
        train_model()
    elif args.evaluate:
        # Initialize classifier
        classifier = EmotionClassifier()
        
        # Load model
        classifier.load_model()
        
        # Prepare data (needed for evaluation)
        classifier.prepare_data()
        
        # Evaluate model
        metrics = classifier.get_evaluation_metrics()
        
        # Print evaluation results
        print(f"Model evaluation results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 score (weighted): {metrics['f1_weighted']:.4f}")
        print(f"Precision (weighted): {metrics['precision_weighted']:.4f}")
        print(f"Recall (weighted): {metrics['recall_weighted']:.4f}")
    else:
        logger.info("No action specified. Use --train or --evaluate.")

if __name__ == "__main__":
    main()
