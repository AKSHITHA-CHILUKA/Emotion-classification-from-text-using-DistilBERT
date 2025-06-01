# Emotion Classification App 🎭

A powerful Streamlit web application that automatically classifies emotions in text using intelligent keyword-based analysis. The app identifies four core emotions: **Joy**, **Sadness**, **Anger**, and **Fear**.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.44+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### 🎯 Real-Time Text Analysis
- Instant emotion detection from any text input
- Confidence scoring for each emotion category
- Clean, intuitive interface with color-coded results

### 📊 Batch Processing
- Upload CSV files to analyze multiple texts at once
- Automatic processing and categorization
- Visual emotion distribution charts
- Downloadable results for further analysis

### 📈 Performance Metrics
- Model evaluation metrics display
- Confusion matrix visualization
- Classification accuracy insights

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AKSHITHA-CHILUKA/emotion-classification-app.git
cd emotion-classification-app

# Install dependencies
pip install streamlit pandas numpy
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 💡 Usage Examples

### Single Text Analysis
1. Navigate to the "Emotion Classifier" tab
2. Enter your text in the input area
3. Click "Classify Emotion" to see results
4. View the predicted emotion and confidence scores

### Batch Processing
1. Go to the "Batch Processing" tab
2. Upload a CSV file with a 'text' column
3. Watch the app process all texts automatically
4. Download the results or view the emotion distribution

### Sample CSV Format
```csv
text
I am so happy about this wonderful day!
This situation makes me feel really sad.
I'm absolutely furious about this issue.
I'm scared about what might happen next.
```

## 📁 Project Structure

```
emotion-classification-app/
├── app.py                 # Main Streamlit application
├── .streamlit/
│   └── config.toml       # Streamlit configuration
├── emotion_data.csv      # Sample data for testing
├── README.md            # Project documentation
└── requirements.txt     # Python dependencies
```

## 🛠️ How It Works

The app uses a sophisticated keyword-based approach:

1. **Text Preprocessing**: Cleans and normalizes input text
2. **Keyword Matching**: Identifies emotion-specific keywords
3. **Scoring Algorithm**: Calculates confidence scores for each emotion
4. **Classification**: Determines the dominant emotion
5. **Visualization**: Presents results with intuitive charts and colors

## 🎨 Emotion Categories

| Emotion | Color | Keywords |
|---------|-------|----------|
| **Joy** | 🟡 Gold | happy, wonderful, amazing, delighted |
| **Sadness** | 🔵 Blue | sad, unhappy, disappointed, miserable |
| **Anger** | 🔴 Red | angry, furious, frustrated, outraged |
| **Fear** | 🟣 Purple | afraid, scared, anxious, worried |

## 🏢 Business Applications

### Customer Feedback Analysis
- Process customer reviews and support tickets
- Identify emotional trends in user feedback
- Prioritize responses based on emotional urgency

### Content Strategy
- Analyze marketing copy emotional impact
- Evaluate social media sentiment
- Optimize messaging for target emotional responses

### Team Communication
- Monitor communication tone
- Identify potential issues early
- Track team morale through text analysis

## 🔮 Future Enhancements

- [ ] Integration with advanced NLP models (BERT, RoBERTa)
- [ ] Real-time API endpoints
- [ ] Custom emotion categories
- [ ] Multi-language support
- [ ] Integration with popular platforms (Slack, Discord)
- [ ] Advanced analytics dashboard

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙋‍♀️ Author

**Akshitha Chiluka**
- GitHub: [@AKSHITHA-CHILUKA](https://github.com/AKSHITHA-CHILUKA)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Inspired by the need for accessible emotion analysis tools
- Thanks to the open-source community for continuous inspiration

---

⭐ Star this repo if you find it helpful!