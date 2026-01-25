import json
import gradio as gr
from textblob import TextBlob

def sentiment_analysis(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment
    print(f" Sentiment Analysis Result: {text}, Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")
    return {
        "polarity": round(sentiment.polarity, 2),
        "subjectivity": round(sentiment.subjectivity, 2),
        "assessment": "Positive" if sentiment.polarity > 0 else "Negative" if sentiment.polarity < 0 else "Neutral"
    }

demo = gr.Interface(
    fn=sentiment_analysis,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs=gr.Textbox(),
    title="Sentiment Analysis",
    description="Enter text to analyze sentiment."
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, mcp_server=True)
