# 🚀 Quick Deployment Steps

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `smart-rag-assistant`
3. Make it **Public** ✅
4. Click "Create repository"

## Step 2: Push Code

Open PowerShell in this directory and run:

```powershell
# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial deployment"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/smart-rag-assistant.git

# Push
git branch -M main
git push -u origin main
```

## Step 3: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file: `app.py`
6. Click "Deploy!"

## Step 4: Add Secrets

In Streamlit Cloud dashboard → Settings → Secrets:

```toml
OPENAI_API_KEY = "your-key-here"
EMBEDDINGS_MODEL = "openai"
```

## Step 5: Done! 🎉

Your app will be live at: `https://YOUR_APP_NAME.streamlit.app`

---

**That's it!** Your app is now online and will auto-update when you push to GitHub.

