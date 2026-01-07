# 🚀 Deployment Guide for Smart RAG Assistant

This guide will help you deploy the Smart RAG Assistant online for free.

## 📋 Pre-Deployment Checklist

- [x] Project copied to `smart-rag-assistant-online`
- [x] Local data folders removed (chroma_db, data, uploads)
- [x] Cache files cleaned (__pycache__)
- [x] .gitignore created
- [x] Streamlit config created
- [ ] GitHub repository created
- [ ] Environment variables configured
- [ ] Deployed to hosting platform

## 🌐 Recommended: Streamlit Cloud (Easiest)

### Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click "New repository"
3. Name it: `smart-rag-assistant` (or any name you prefer)
4. Make it **Public** (required for free Streamlit Cloud)
5. **Don't** initialize with README, .gitignore, or license
6. Click "Create repository"

### Step 2: Push Code to GitHub

```bash
# Navigate to your online copy
cd C:\Users\DELL\Desktop\smart-rag-assistant-online

# Initialize git (if not already done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit for deployment"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/smart-rag-assistant.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Fill in:
   - **Repository**: Select your `smart-rag-assistant` repo
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click "Deploy!"

### Step 4: Configure Secrets

1. In your Streamlit Cloud app dashboard, click "Settings" (⚙️)
2. Go to "Secrets" tab
3. Add these secrets:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
EMBEDDINGS_MODEL = "openai"
OPENAI_MODEL = "gpt-3.5-turbo"
CHUNK_SIZE = "1000"
CHUNK_OVERLAP = "200"
TOP_K_RETRIEVAL = "3"
TEMPERATURE = "0.3"
MAX_TOKENS = "300"
```

4. Click "Save"
5. Your app will automatically redeploy

### Step 5: Access Your App

- Your app will be available at: `https://YOUR_APP_NAME.streamlit.app`
- Share this URL with others!

## 🔄 Alternative: Render (Docker Support)

### Step 1: Push to GitHub (same as above)

### Step 2: Deploy on Render

1. Go to [render.com](https://render.com)
2. Sign up/Sign in with GitHub
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name**: `smart-rag-assistant`
   - **Environment**: `Docker`
   - **Region**: Choose closest to you
   - **Branch**: `main`
6. Add Environment Variables:
   - `OPENAI_API_KEY`: your key
   - `EMBEDDINGS_MODEL`: `openai`
   - Other variables as needed
7. Click "Create Web Service"
8. Wait for deployment (5-10 minutes)

## 📝 Important Notes

### Data Persistence
- **Free tiers don't persist data** - ChromaDB data and chat history will reset on restart
- For production, consider:
  - Using cloud storage (S3, etc.) for ChromaDB
  - Using a database for chat history
  - Using paid hosting with persistent storage

### Environment Variables
- Never commit `.env` files to GitHub
- Use platform secrets management instead
- Streamlit Cloud: Use "Secrets" in dashboard
- Render: Use "Environment" tab

### Resource Limits
- Free tiers have limited CPU/memory
- May experience slower performance
- Consider upgrading for production use

### Auto-Deployment
- Both Streamlit Cloud and Render auto-deploy on git push
- Just push changes and they'll update automatically!

## 🐛 Troubleshooting

### App won't start
- Check logs in platform dashboard
- Verify all environment variables are set
- Ensure `requirements.txt` is up to date

### Import errors
- Check all dependencies in `requirements.txt`
- Verify Python version compatibility

### API errors
- Verify `OPENAI_API_KEY` is correct
- Check API quota/billing

### Data not persisting
- This is expected on free tiers
- Consider upgrading or using cloud storage

## 📊 Monitoring

- **Streamlit Cloud**: Check "Manage app" for logs
- **Render**: Check "Logs" tab in dashboard
- Monitor usage and performance

## 🔒 Security

- Never commit API keys
- Use platform secrets
- Keep repository public only if needed for free tier
- Consider private repo with paid hosting

## 🎉 Success!

Once deployed, you'll have:
- ✅ Public URL to share
- ✅ Auto-deployment on git push
- ✅ Free hosting (with limitations)
- ✅ Professional deployment

---

**Need help?** Check platform documentation or create an issue in your repository.

