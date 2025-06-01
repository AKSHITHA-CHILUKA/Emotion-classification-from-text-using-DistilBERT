import os
import torch
import numpy as np
import pandas as pd
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset, load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionClassifier:
    """Class for emotion classification using DistilBERT."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emotion_labels = ["joy", "sadness", "anger", "fear", "surprise", "neutral", "love"]
        self.max_length = 128
        self.model_checkpoint = "distilbert-base-uncased"
        self.model_dir = "emotion_model"
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.evaluation_metrics = None

    def load_or_train_model(self):
        """Load a pre-trained model or train if it doesn't exist."""
        if os.path.exists(f"{self.model_dir}/pytorch_model.bin"):
            logger.info("Loading pre-trained model")
            self.load_model()
        else:
            logger.info("No pre-trained model found. Training new model.")
            self.prepare_data()
            self.train_model()
        
        # Make evaluation metrics available
        self.evaluate_model()
        
        return self

    def prepare_data(self):
        """Load and prepare data for training."""
        logger.info("Preparing data for training")
        
        try:
            # Try to load the Hugging Face emotion dataset
            emotion_dataset = load_dataset("emotion")
            
            # Convert to pandas for easier handling
            train_df = pd.DataFrame(emotion_dataset["train"])
            val_df = pd.DataFrame(emotion_dataset["validation"])
            test_df = pd.DataFrame(emotion_dataset["test"])
            
            # Map numeric labels to text labels
            label_map = {
                0: "joy",
                1: "sadness",
                2: "anger",
                3: "fear",
                4: "love",
                5: "surprise",
                6: "neutral"
            }
            
            # Update emotion labels based on the dataset
            self.emotion_labels = list(label_map.values())
            
            # Convert numeric labels to text labels
            train_df["emotion"] = train_df["label"].map(label_map)
            val_df["emotion"] = val_df["label"].map(label_map)
            test_df["emotion"] = test_df["label"].map(label_map)
            
        except Exception as e:
            logger.warning(f"Error loading Hugging Face dataset: {e}")
            logger.info("Creating fallback dataset")
            
            # Create a fallback dataset
            data = [
                {"text": "I am so happy today!", "emotion": "joy"},
                {"text": "I feel sad and depressed.", "emotion": "sadness"},
                {"text": "I am furious about what happened.", "emotion": "anger"},
                {"text": "I am terrified of spiders.", "emotion": "fear"},
                {"text": "I was surprised by the unexpected gift.", "emotion": "surprise"},
                {"text": "I'm just walking to the store.", "emotion": "neutral"},
                {"text": "I love you so much!", "emotion": "love"},
                {"text": "That movie made me so happy.", "emotion": "joy"},
                {"text": "I'm feeling down today.", "emotion": "sadness"},
                {"text": "That makes me so angry!", "emotion": "anger"},
                {"text": "I'm afraid of the dark.", "emotion": "fear"},
                {"text": "Wow! That's amazing news!", "emotion": "surprise"},
                {"text": "I'm just sitting here.", "emotion": "neutral"},
                {"text": "I adore my family.", "emotion": "love"},
                {"text": "The weather is nice today.", "emotion": "neutral"},
                {"text": "I can't believe what happened!", "emotion": "surprise"},
                {"text": "I'm scared of failing the exam.", "emotion": "fear"},
                {"text": "How dare you say that to me!", "emotion": "anger"},
                {"text": "I'm crying because I'm so sad.", "emotion": "sadness"},
                {"text": "This brings me so much joy.", "emotion": "joy"}
            ]
            
            # Create a DataFrame from the sample data
            df = pd.DataFrame(data)
            
            # Split into train, validation, and test sets
            train_df = df.sample(frac=0.7, random_state=42)
            temp_df = df.drop(train_df.index)
            val_df = temp_df.sample(frac=0.5, random_state=42)
            test_df = temp_df.drop(val_df.index)
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_checkpoint)
        
        # Convert to Hugging Face datasets
        self.train_dataset = self._prepare_dataset(train_df)
        self.val_dataset = self._prepare_dataset(val_df)
        self.test_dataset = self._prepare_dataset(test_df)
        
        logger.info(f"Prepared datasets - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

    def _prepare_dataset(self, df):
        """Convert DataFrame to tokenized Dataset."""
        # Create label map
        label_map = {emotion: i for i, emotion in enumerate(self.emotion_labels)}
        
        # Add numeric labels
        df["label"] = df["emotion"].map(label_map)
        
        # Convert to Dataset
        dataset = Dataset.from_pandas(df)
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            lambda examples: self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            ),
            batched=True
        )
        
        return tokenized_dataset

    def train_model(self):
        """Train the DistilBERT model on the prepared dataset."""
        logger.info("Training DistilBERT model")
        
        # Initialize model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_checkpoint,
            num_labels=len(self.emotion_labels)
        )
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy"
        )
        
        # Define compute_metrics function
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="weighted"
            )
            acc = accuracy_score(labels, preds)
            return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train model
        trainer.train()
        
        # Save model and tokenizer
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        
        logger.info("Model training completed")

    def load_model(self):
        """Load a pre-trained model and tokenizer."""
        try:
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
            
            # Load emotion labels if available
            if os.path.exists(f"{self.model_dir}/emotion_labels.txt"):
                with open(f"{self.model_dir}/emotion_labels.txt", "r") as f:
                    self.emotion_labels = f.read().strip().split("\n")
            
            logger.info("Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Fall back to the pre-trained model
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_checkpoint,
                num_labels=len(self.emotion_labels)
            )
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_checkpoint)
            logger.info("Loaded default pre-trained model")

    def evaluate_model(self):
        """Evaluate the model on the test dataset."""
        if not self.model or not self.tokenizer:
            logger.error("Model or tokenizer not initialized")
            return
        
        # If we don't have a test dataset, create a simple one
        if not self.test_dataset:
            self.prepare_data()
        
        logger.info("Evaluating model")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Get predictions for test dataset
        all_preds = []
        all_labels = []
        
        # Process in batches
        for i in range(0, len(self.test_dataset), 16):
            batch = self.test_dataset[i:i+16]
            inputs = {k: torch.tensor(v) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
            with torch.no_grad():
                outputs = self.model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['label'])
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        # Generate classification report
        report = classification_report(
            all_labels, all_preds, 
            target_names=self.emotion_labels, 
            output_dict=True
        )
        
        # Convert report to DataFrame for better visualization
        report_df = pd.DataFrame(report).transpose()
        
        # Store metrics
        self.evaluation_metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'confusion_matrix': conf_matrix,
            'labels': self.emotion_labels,
            'classification_report': report_df
        }
        
        logger.info(f"Model evaluation completed. Accuracy: {accuracy:.4f}")

    def get_evaluation_metrics(self):
        """Return the evaluation metrics."""
        if not self.evaluation_metrics:
            self.evaluate_model()
        return self.evaluation_metrics

    def predict(self, text):
        """Predict the emotion for a given text."""
        if not self.model or not self.tokenizer:
            logger.error("Model or tokenizer not initialized")
            return "unknown", {}
        
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        )
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        
        # Get emotion label
        emotion = self.emotion_labels[pred_class]
        
        # Get probabilities for all emotions
        probabilities = {
            self.emotion_labels[i]: probs[0, i].item()
            for i in range(len(self.emotion_labels))
        }
        
        return emotion, probabilities
