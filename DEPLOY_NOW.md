# 🚀 Deploy Now - Step by Step

## ✅ Step 1: Create GitHub Repository

1. Go to **https://github.com/new**
2. Repository name: `smart-rag-assistant` (or any name you prefer)
3. Make it **Public** ✅ (required for free Streamlit Cloud)
4. **DO NOT** check "Initialize with README" (we already have files)
5. Click **"Create repository"**

## ✅ Step 2: Connect and Push to GitHub

After creating the repository, GitHub will show you commands. Use these:

```powershell
# Add your GitHub repository as remote
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/smart-rag-assistant.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**OR** if you prefer SSH:
```powershell
git remote add origin git@github.com:YOUR_USERNAME/smart-rag-assistant.git
git branch -M main
git push -u origin main
```

## ✅ Step 3: Deploy on Streamlit Cloud

1. Go to **https://share.streamlit.io**
2. Sign in with your **GitHub account**
3. Click **"New app"**
4. Fill in:
   - **Repository**: Select `smart-rag-assistant`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"**

## ✅ Step 4: Add Your API Key

1. In Streamlit Cloud dashboard, click **"⚙️ Settings"**
2. Go to **"Secrets"** tab
3. Add this (replace with your actual key):

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
EMBEDDINGS_MODEL = "openai"
OPENAI_MODEL = "gpt-3.5-turbo"
```

4. Click **"Save"**
5. Your app will automatically redeploy!

## ✅ Step 5: Access Your App! 🎉

Your app will be live at:
**https://YOUR_APP_NAME.streamlit.app**

Share this URL with anyone!

---

## 🆘 Need Help?

- **Git not installed?** Download from: https://git-scm.com/download/win
- **GitHub account?** Sign up at: https://github.com/signup
- **OpenAI API key?** Get from: https://platform.openai.com/api-keys

---

**Ready?** Start with Step 1 above! 🚀

