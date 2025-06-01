import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """
    Preprocess text for emotion classification.
    
    Args:
        text (str): Input text to preprocess
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_emotion_color(emotion):
    """
    Get a color for each emotion category.
    
    Args:
        emotion (str): The emotion category
        
    Returns:
        str: Hex color code
    """
    emotion_colors = {
        'joy': '#FFD700',      # Gold
        'happiness': '#FFD700', # Gold (alternative name)
        'sadness': '#4682B4',  # Steel Blue
        'anger': '#FF4500',    # Orange Red
        'fear': '#800080',     # Purple
        'surprise': '#32CD32', # Lime Green
        'neutral': '#808080',  # Gray
        'love': '#FF69B4',     # Hot Pink
        'disgust': '#006400',  # Dark Green
        'shame': '#8B4513',    # Saddle Brown
        'guilt': '#483D8B'     # Dark Slate Blue
    }
    
    # Default color if emotion not found
    return emotion_colors.get(emotion.lower(), '#1E90FF')  # Default: Dodger Blue

def truncate_text(text, max_length=100):
    """
    Truncate text to a specified length and add ellipsis.
    
    Args:
        text (str): Input text to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + '...'

def format_probability(prob):
    """
    Format probability as percentage.
    
    Args:
        prob (float): Probability value (0-1)
        
    Returns:
        str: Formatted percentage
    """
    return f"{prob * 100:.2f}%"
