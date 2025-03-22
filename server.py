from flask import Flask, render_template, request, send_file,abort,send_from_directory
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PyPDF2 import PdfReader
from docx import Document
import re
import seaborn as sns
import matplotlib.pyplot as plt
import os
import requests
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load the fine-tuned Legal LED model
MODEL_NAME = "Izza-shahzad-13/legal-LED-final"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Function to generate summary
def generate_summary(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    outputs = model.generate(inputs, max_length=800, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function to calculate sentence importance scores
def calculate_sentence_importance(summary):
    sentences = summary.split(". ")
    scores = [len(sentence) for sentence in sentences]  # Score based on sentence length
    max_score = max(scores) if scores else 1
    normalized_scores = [score / max_score for score in scores]
    return sentences, normalized_scores

# Function to generate heatmap
def generate_heatmap(scores):
    plt.figure(figsize=(10, 2))
    sns.heatmap([scores], annot=True, cmap="coolwarm", xticklabels=False, yticklabels=False, cbar=True)
    plt.title("Sentence Importance Heatmap")
    plt.savefig("static/heatmap.png")  # Save heatmap image
    plt.close()

# Function to highlight sentences in the summary
def highlight_summary(sentences, scores):
    cmap = sns.color_palette("coolwarm", as_cmap=True)
    highlighted_summary = ""

    for sentence, score in zip(sentences, scores):
        color = cmap(score)
        rgb_color = f"rgb({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)})"
        highlighted_summary += f'<span style="background-color:{rgb_color};padding:2px;">{sentence}.</span> '

    return highlighted_summary

# Function to highlight legal terms
def highlight_keywords(text):
    patterns = {
        'act_with_year': r'\b([A-Za-z\s]+(?:\sAct(?:\s[\d]{4})?))\s*,\s*(\d{4})\b',
        'article': r'\bArticle\s\d{1,3}(-[A-Z])?\b',
        'section': r'\bSection\s\d{1,3}[-A-Za-z]?\(?[a-zA-Z]?\)?\b',
        'date': r'\b(?:[A-Za-z]+)\s\d{4}\b|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
        'persons': r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b',
        'ordinance': r'\b([A-Z][a-z\s]+Ordinance(?:,\s\d{4})?)\b',  # Example: PEMRA Ordinance, 2002
        'petition': r'\b(?:[A-Za-z\s]*Petition\sNo\.\s\d+/\d{4})\b',  # Example: Constitutional Petition No. 123/2024
        'act_with_year': r'\b([A-Za-z\s]+(?:\sAct(?:\s\d{4})?)),\s*(\d{4})\b',  # Example: Control of Narcotic Substances Act, 1997
        'article': r'\b(Article\s\d{1,3}(-[A-Z])?)\b',  # Example: Article 10-A
        'section': r'\b(Section\s\d{1,3}(\([a-zA-Z0-9]+\))?)\b',  # Example: Section 302(b), Section 9(c), Section 144-A
        'date': r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{1,2},?\s\d{4})\b',  
        # Examples: 15/07/2015, July 2015, March 5, 2021, 2023
        'person': r'\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)\b'  # Example: Justice Ali Raza
    }

    highlighted_text = text
    for pattern in patterns.values():
        highlighted_text = re.sub(pattern, lambda match: f'<span class="highlight">{match.group(0)}</span>', highlighted_text)

    return highlighted_text

# Function to read uploaded files
def read_file(file):
    if file.filename.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.filename.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        return " ".join(page.extract_text() for page in pdf_reader.pages)
    elif file.filename.endswith(".docx"):
        doc = Document(file)
        return " ".join(paragraph.text for paragraph in doc.paragraphs)
    return None

# Function to fetch text from a URL
def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("Content-Type", "")
        if "text/html" in content_type:  # If it's a webpage
            soup = BeautifulSoup(response.text, "html.parser")
            paragraphs = soup.find_all("p")  # Extract paragraph text
            return " ".join([p.get_text() for p in paragraphs])

        elif "text/plain" in content_type:  # If it's a plain text file
            return response.text

        else:
            return None
    except Exception as e:
        print("Error fetching URL:", e)
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    document_text = None
    summary = None
    heatmap_url = None

    if request.method == "POST":
        file = request.files.get("file")
        pasted_text = request.form.get("pasteText", "").strip()
        url = request.form.get("url", "").strip()

        if file and file.filename:
            document_text = read_file(file)
        elif pasted_text:
            document_text = pasted_text
        elif url:
            document_text = fetch_text_from_url(url)

        if document_text:
            summary = generate_summary(document_text)
            sentences, scores = calculate_sentence_importance(summary)

            generate_heatmap(scores)

            highlighted_summary = highlight_summary(sentences, scores)
            highlighted_summary = highlight_keywords(highlighted_summary)

            # Save the summary to a text file
            with open("summary.txt", "w", encoding="utf-8") as f:
                f.write(summary)

            return render_template("mainscreen.html", document_text=document_text, summary=highlighted_summary, heatmap_url="static/heatmap.png")

    return render_template("mainscreen.html", document_text=None, summary=None, heatmap_url=None)

@app.route("/download_summary")
def download_summary():
    file_path = os.path.join(os.getcwd(), "summary.txt")

    if not os.path.exists(file_path):
        return abort(404, description="File not found")

    return send_file(file_path, as_attachment=True, download_name="summary.txt", mimetype="text/plain")
@app.route("/home")
def home():
    return render_template("homepage.html")  # Homepage 
@app.route("/contact")
def contact():
    return render_template("contactpage.html")
@app.route("/about")
def about():
    return render_template("aboutpage.html") 
@app.route("/summarization")
def summarization():
    return render_template("mainscreen.html")
@app.route("/login")
def login():
    return render_template("loginpage.html")  # Login Page

@app.route("/signup")
def signup():
    return render_template("signuppage.html") 
@app.route('/forget-password')
def forget_password():
    return render_template('forgetpasswordpage.html')

@app.route('/lawbooks/<filename>')
def serve_pdf(filename):
    return send_from_directory('static/lawbooks', filename)

if __name__ == "__main__":
    app.run(debug=True)
