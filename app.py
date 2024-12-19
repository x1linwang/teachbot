from flask import Flask, request, jsonify, render_template, redirect, url_for, make_response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from werkzeug.utils import secure_filename
import os
import shutil
from fpdf import FPDF
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'uploads'
CORS(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    conversation_name = db.Column(db.String(150), nullable=False)
    messages = db.Column(db.Text, nullable=False)

# Initialize Database
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Functionality for Knowledge Base
def build_knowledge_base():
    if os.path.exists('faiss_index'):
        shutil.rmtree('faiss_index')
    embeddings = OpenAIEmbeddings()
    file_paths = []
    docs = []
    for root, _, files in os.walk(app.config['UPLOAD_FOLDER']):
        for file in files:
            path = os.path.join(root, file)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    docs.append(f.read())
                file_paths.append(file)
            except UnicodeDecodeError:
                print(f"Skipping non-UTF-8 file: {file}")
    if not docs:
        print("No valid documents found to index.")
        return
    metadata = [{'filename': file} for file in file_paths]
    vectorstore = FAISS.from_texts(texts=docs, embedding=embeddings, metadatas=metadata)
    vectorstore.save_local('faiss_index')

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    user_input = request.json.get('message')
    vectorstore = FAISS.load_local('faiss_index', OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever()
    
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(user_input)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    # Prepare the prompt with context
    full_prompt = f"Context:\n{context}\n\nUser: {user_input}\nChatbot:"
    
    # Use the chat model
    chat_model = ChatOpenAI()
    answer = chat_model.predict(full_prompt)
    
    # Save chat history
    chat_history = ChatHistory(user_id=current_user.id, conversation_name="New Conversation", messages="")
    chat_history.messages += f"User: {user_input}\nChatbot: {answer}\n"
    db.session.add(chat_history)
    db.session.commit()

    # Generate auto-naming for conversation
    summary_prompt = f"Summarize this conversation in a short name:\n{chat_history.messages}"
    chat_history.conversation_name = chat_model.predict(summary_prompt)
    db.session.commit()
    
    return jsonify({"response": answer, "chat_id": chat_history.id})

@app.route('/api/rename-conversation', methods=['POST'])
@login_required
def rename_conversation():
    data = request.json
    chat_id = data.get('chat_id')
    new_name = data.get('new_name')
    
    chat = ChatHistory.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    
    chat.conversation_name = new_name
    db.session.commit()
    return jsonify({"success": True})

@app.route('/api/download-chat', methods=['GET'])
@login_required
def download_chat():
    chat_id = request.args.get('id')
    chat = ChatHistory.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return "Chat not found", 404

    # Prepare PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=chat.conversation_name, ln=True, align='C')
    pdf.ln(10)

    messages = chat.messages.split("\n")
    for message in messages:
        if message.startswith("User:"):
            pdf.set_text_color(0, 0, 255)
            pdf.multi_cell(0, 10, f"User: {message.replace('User: ', '')}")
        elif message.startswith("Chatbot:"):
            pdf.set_text_color(0, 128, 0)
            pdf.multi_cell(0, 10, f"Chatbot: {message.replace('Chatbot: ', '')}")

    # Save PDF to file
    filename = f"chat_{chat_id}.pdf"
    filepath = os.path.join('static', filename)
    pdf.output(filepath)

    # Return the file response
    return jsonify({"download_url": url_for('static', filename=filename, _external=True)})

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    # Reindex the knowledge base after upload
    build_knowledge_base()

    return "File uploaded and indexed successfully", 200

@app.route('/reindex', methods=['POST'])
@login_required
def reindex():
    build_knowledge_base()
    return "Knowledge base reindexed successfully", 200

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            return "Username already exists", 400
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    analytics_data = {
        "total_chats_today": 37,
        "total_users": 120,
        "average_rating": 4.5,
    }
    return render_template('dashboard.html', analytics=analytics_data)

@app.route('/chatbot')
@login_required
def chatbot_ui():
    chat_histories = ChatHistory.query.filter_by(user_id=current_user.id).all()
    return render_template('chatbot.html', chat_histories=chat_histories)

@app.route('/api/chat-history', methods=['GET'])
@login_required
def get_chat_history():
    chat_id = request.args.get('id')
    chat = ChatHistory.query.filter_by(id=chat_id, user_id=current_user.id).first()
    if not chat:
        return jsonify({"error": "Chat not found"}), 404
    messages = chat.messages.split("\n")
    formatted_messages = []
    for message in messages:
        if message.startswith("User:"):
            formatted_messages.append({"type": "user", "text": message.replace("User: ", "")})
        elif message.startswith("Chatbot:"):
            formatted_messages.append({"type": "chatbot", "text": message.replace("Chatbot: ", "")})
    return jsonify({"messages": formatted_messages})

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    build_knowledge_base()  # Ensure knowledge base is indexed at startup
    app.run(debug=True)
