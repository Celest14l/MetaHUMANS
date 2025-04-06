from flask import Flask, request, jsonify, render_template_string
import requests
from datetime import datetime

app = Flask(__name__)

# NewsAPI key (get yours from https://newsapi.org/)
NEWS_API_KEY = "286319e491de4b5eb07b1f27224686bf"  # Replace with your free API key
BASE_URL = "https://newsapi.org/v2/everything"  # Changed to 'everything' for broader queries

# Supported categories (for reference, optional)
VALID_CATEGORIES = ["business", "entertainment", "general", "health", "science", "sports", "technology"]

# Helper function to fetch news based on user query
def fetch_news(query):
    params = {
        "apiKey": NEWS_API_KEY,
        "q": query,         # User-provided query (e.g., "AI", "sports", "Ukraine war")
        "language": "en",   # English for now
        "pageSize": 10,     # Limit to 10 articles
        "sortBy": "relevancy"  # Sort by relevance to query
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data["status"] == "ok" and data["totalResults"] > 0:
            articles = [{
                "title": article["title"],
                "source": article["source"]["name"],
                "url": article["url"],
                "publishedAt": article["publishedAt"]
            } for article in data["articles"]]
            return {"status": "success", "articles": articles, "query": query}
        else:
            return {"status": "error", "message": f"No articles found for '{query}'"}
    
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Failed to fetch news: {str(e)}"}

# Root endpoint with HTML form to ask for user input
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            return render_template_string(HTML_TEMPLATE, error="Please enter a topic or keyword.", articles=None)
        
        result = fetch_news(query)
        if result["status"] == "success":
            return render_template_string(HTML_TEMPLATE, articles=result["articles"], query=query)
        else:
            return render_template_string(HTML_TEMPLATE, error=result["message"], articles=None)
    
    # GET request: Show the form
    return render_template_string(HTML_TEMPLATE, articles=None)

# API endpoint for programmatic access (optional)
@app.route('/news', methods=['GET'])
def get_news():
    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({"status": "error", "message": "Query parameter is required"}), 400
    
    result = fetch_news(query)
    result["date"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    return jsonify(result), 200 if result["status"] == "success" else 500

# Simple HTML template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>News Fetcher</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .article { margin-bottom: 15px; }
        .error { color: red; }
    </style>
</head>
<body>
    <h1>News Fetcher</h1>
    <p>Enter a topic, keyword, or category (e.g., "AI", "sports", "Ukraine war"):</p>
    <form method="POST">
        <input type="text" name="query" placeholder="What do you want to know about?" style="width: 300px;">
        <input type="submit" value="Fetch News">
    </form>
    {% if error %}
        <p class="error">{{ error }}</p>
    {% endif %}
    {% if articles %}
        <h2>Results for "{{ query }}":</h2>
        {% for article in articles %}
            <div class="article">
                <strong>{{ article.title }}</strong><br>
                <small>Source: {{ article.source }} | Published: {{ article.publishedAt }}</small><br>
                <a href="{{ article.url }}" target="_blank">Read more</a>
            </div>
        {% endfor %}
    {% endif %}
</body>
</html>
"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)