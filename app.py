import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

# ==================== CONFIG ====================
from dotenv import load_dotenv
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# ==================== FLASK APP ====================
app = Flask(__name__)
CORS(app)

# ==================== LOAD DATA ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Data.txt")
INDEX_PATH = os.path.join(BASE_DIR, "index")

with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_text = f.read()

# ==================== SPLIT & FILTER ====================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_text(raw_text)
docs = [d for d in docs if d and d.strip()]  # loại bỏ None hoặc rỗng

if not docs:
    raise ValueError("❌ Không có dữ liệu hợp lệ trong Data.txt!")

# ==================== EMBEDDING MODEL ====================
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # nếu có GPU đổi thành "cuda"
)

# ==================== BUILD / LOAD FAISS ====================
if not os.path.exists(INDEX_PATH):
    print("⚡ Chưa có index → Tạo mới...")
    vectorstore = FAISS.from_texts(docs, embedding_model)
    vectorstore.save_local(INDEX_PATH)
else:
    print("✅ Đã có index → Load nhanh...")
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
# ==================== RAG CHAT FUNCTION ====================
def rag_chat(query: str):
    if not query:
        return "❌ Câu hỏi rỗng."
    try:
        docs = retriever.get_relevant_documents(query)
        context = "\n".join([d.page_content for d in docs])  # truncate
        prompt = f"""
Bạn là chatbot Tosi hỗ trợ Aquaponics.
Giới thiệu bản thân khi người dùng chào.
Trả lời trực tiếp, không chào hỏi dư thừa.
Trả lời đầy đủ và chi tiết dựa trên ngữ cảnh dưới đây.
Tổng hợp tất cả thông tin liên quan, không bỏ sót.
Nếu liệt kê, mỗi ý một dòng bắt đầu bằng '-'.
Nếu không có thông tin, trả lời: "Xin lỗi. Điều này không có trong tài liệu"

Ngữ cảnh:
{context}

Câu hỏi: {query}
"""
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt, generation_config={"temperature":0.2})
        return response.text if response and response.text else "Xin lỗi, không nhận được phản hồi."
    except Exception as e:
        print("❌ Lỗi rag_chat:", e)
        return "❌ Bot gặp lỗi khi xử lý câu hỏi."

# ==================== ROUTES ====================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        query = data.get("query", "")
        answer = rag_chat(query)
        return jsonify({"response": answer})
    except Exception as e:
        print("❌ Lỗi /chat:", e)
        return jsonify({"response": f"❌ Server error: {str(e)}"})

# ==================== RUN ====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)