# 🚀 Quick Start Guide

Get your Smart RAG Document Assistant up and running in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# For local embeddings (FREE - no API key needed)
EMBEDDINGS_MODEL=sentence-transformers

# OR for OpenAI embeddings (requires API key)
OPENAI_API_KEY=your-api-key-here
EMBEDDINGS_MODEL=openai
OPENAI_MODEL=gpt-3.5-turbo
```

**Recommended for beginners:** Use `sentence-transformers` (free, no API key needed)

## Step 3: Run the Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`

## Step 4: Use the Application

1. **Upload a Document**: Click "Browse files" in the sidebar and select a PDF/DOCX/TXT file
2. **Process Document**: Click "Process & Index Document"
3. **Ask Questions**: Type your question in the chat interface
4. **Get Answers**: Receive answers with source citations!

## 🎯 Testing with Sample Documents

Try uploading:
- A PDF research paper
- A technical documentation PDF
- A text file with information
- A Word document

## 📊 Example Questions

- "What is the main topic of this document?"
- "Summarize the key points"
- "What are the important dates mentioned?"
- "Explain [concept] from the document"

## 🐳 Docker Alternative

If you prefer Docker:

```bash
docker-compose up -d
```

Access at `http://localhost:8501`

## ❓ Troubleshooting

### Issue: "No module named 'langchain'"
**Solution:** Run `pip install -r requirements.txt`

### Issue: "OpenAI API key required"
**Solution:** Either:
1. Set `EMBEDDINGS_MODEL=sentence-transformers` (free option)
2. Or get an OpenAI API key and set `OPENAI_API_KEY`

### Issue: "Port 8501 already in use"
**Solution:** Change the port:
```bash
streamlit run app.py --server.port 8502
```

## 🎓 Next Steps

1. **Customize**: Edit `utils/config.py` to adjust chunk sizes, models, etc.
2. **Deploy**: See README.md for deployment instructions
3. **Enhance**: Add more features (multi-document support, better UI, etc.)

---

**Happy building! 🚀**

