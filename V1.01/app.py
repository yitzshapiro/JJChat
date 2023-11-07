from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import openai
import weaviate
import os
import re
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # This is to handle Cross-Origin Resource Sharing (CORS)

# Initialize the Sentence Transformer model for embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Initialize Weaviate client
weaviate_client = weaviate.Client("http://localhost:8080")

# Set the OpenAI API key from the environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/prompt', methods=['POST'])
def handle_prompt():
    system_prompt = "You are a martial arts instructor who is detailed and descriptive. Please add newline formatting to your output."
    data = request.json
    user_prompt = data['prompt']

    initial_prompt_embedding = get_embedding(user_prompt)
    similar_documents = search_weaviate(initial_prompt_embedding)
    model_response = query_openai(system_prompt, similar_documents['data']['Get']['Document'], user_prompt)
    
    youtube_urls = extract_youtube_urls(similar_documents['data']['Get']['Document'])
    
    return jsonify({"response": model_response, "youtube_urls": youtube_urls})

def get_embedding(text):
    return embedding_model.encode([text])[0]

def search_weaviate(query_embedding, limit=6):
    vector_query = {"vector": query_embedding.tolist()}
    return weaviate_client.query.get("Document", ["content", "_additional {id, certainty}"]) \
        .with_near_vector(vector_query) \
        .with_limit(limit) \
        .do()

def query_openai(system_message, documents, user_prompt):
    messages = [{"role": "system", "content": system_message}]
    for doc in documents:
        messages.append({"role": "system", "content": doc['content']})
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=messages
        )
        return response.choices[0].message['content'] if response.choices else "No response received"
    except openai.error.OpenAIError as e:
        return f"Error with OpenAI API call: {str(e)}"

def extract_youtube_urls(documents):
    youtube_urls = []
    url_pattern = re.compile(r'https?://(?:www\.)?youtube\.com/watch\?v=[\w-]+')
    for doc in documents:
        # Split the document content by newline and get the first line
        first_line = doc['content'].split('\n', 1)[0]
        # Find all URLs in the first line
        urls = url_pattern.findall(first_line)
        youtube_urls.extend(urls)  # Extend the list with the found URLs
    return youtube_urls

if __name__ == '__main__':
    app.run(debug=True)
