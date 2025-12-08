# 🚀 Corn Disease Detection App - Deployment Guide

## Option 1: Streamlit Community Cloud (FREE) ⭐ RECOMMENDED

### Prerequisites:
- GitHub account
- Your code in a GitHub repository

### Steps:

1. **Create GitHub Repository**
   ```bash
   cd /media/panda/Data1/leaf_ditection
   git init
   git add .
   git commit -m "Initial commit: Corn disease detection app"
   ```

2. **Push to GitHub**
   ```bash
   # Create a new repository on GitHub first, then:
   git remote add origin https://github.com/YOUR_USERNAME/corn-disease-detection.git
   git branch -M main
   git push -u origin main
   ```

3. **Deploy on Streamlit Cloud**
   - Go to: https://share.streamlit.io/
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Choose branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

4. **Important: Add Model File**
   - GitHub has 100MB file limit
   - Your .h5 model file is likely larger
   - Solutions:
     
     **A. Use Git LFS (Large File Storage)**
     ```bash
     git lfs install
     git lfs track "*.h5"
     git add .gitattributes
     git add *.h5
     git commit -m "Add model file with LFS"
     git push
     ```
     
     **B. Use Google Drive/Dropbox Link**
     Update `app.py` to download model from URL:
     ```python
     import gdown
     
     @st.cache_resource
     def download_model():
         url = "YOUR_GOOGLE_DRIVE_LINK"
         output = "model.h5"
         if not os.path.exists(output):
             gdown.download(url, output, quiet=False)
         return load_model(output)
     ```

---

## Option 2: Hugging Face Spaces (FREE)

### Steps:

1. **Create Account**
   - Go to: https://huggingface.co/
   - Sign up for free account

2. **Create New Space**
   - Click "New Space"
   - Name: `corn-disease-detection`
   - SDK: Streamlit
   - Visibility: Public or Private

3. **Upload Files**
   - Upload `app.py`
   - Upload `requirements.txt`
   - Upload model file (or use external link)

4. **Done!** Your app will be live at:
   `https://huggingface.co/spaces/YOUR_USERNAME/corn-disease-detection`

---

## Option 3: Railway.app (FREE Tier Available)

### Steps:

1. **Create Account**: https://railway.app/

2. **Create Project from GitHub**
   - Connect your GitHub repository
   - Railway auto-detects Streamlit

3. **Add Config File** (optional)
   Create `railway.toml`:
   ```toml
   [build]
   builder = "NIXPACKS"

   [deploy]
   startCommand = "streamlit run app.py --server.port $PORT"
   ```

4. **Deploy** - Automatic deployment on every push

---

## Option 4: Render (FREE Tier)

### Steps:

1. **Create Account**: https://render.com/

2. **New Web Service**
   - Connect GitHub repository
   - Name: `corn-disease-detection`
   - Environment: `Python`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

3. **Deploy** - Automatic!

---

## Option 5: PythonAnywhere (FREE Tier)

### Steps:

1. **Create Account**: https://www.pythonanywhere.com/

2. **Upload Files**
   - Use their file manager or git clone

3. **Create Web App**
   - Choose manual configuration
   - Python version 3.x
   - Setup virtual environment
   - Configure WSGI file for Streamlit

4. **Run**
   ```bash
   streamlit run app.py --server.port 8000
   ```

---

## 📦 Required Files for Deployment

Make sure you have these files:

### 1. `requirements.txt` ✅ (Already have)
```
streamlit
tensorflow
numpy
Pillow
```

### 2. `.gitignore` (Create this)
```
.venv/
__pycache__/
*.pyc
.env
.DS_Store
*.log
```

### 3. `README.md` ✅ (Already have)

### 4. `app.py` ✅ (Already have)

---

## 🎯 Quick Deploy Command (for Streamlit Cloud)

```bash
# 1. Initialize git
git init

# 2. Add files
git add app.py requirements.txt README.md

# 3. Commit
git commit -m "Initial deployment"

# 4. Add remote (replace with your repo)
git remote add origin https://github.com/YOUR_USERNAME/your-repo.git

# 5. Push
git push -u origin main
```

---

## ⚠️ Important Notes:

### Model File Size Issue:
Your `.h5` model file (~200-500MB) is too large for GitHub (100MB limit).

**Solutions:**

1. **Git LFS** (Recommended for Streamlit Cloud)
   ```bash
   git lfs install
   git lfs track "*.h5"
   git add .gitattributes
   git commit -m "Track model with LFS"
   ```

2. **Google Drive/Dropbox Link** (Easiest)
   - Upload model to Google Drive
   - Get shareable link
   - Download in app using `gdown` library
   
   Add to `requirements.txt`:
   ```
   gdown
   ```
   
   Update `app.py`:
   ```python
   import gdown
   
   MODEL_URL = "https://drive.google.com/uc?id=YOUR_FILE_ID"
   
   @st.cache_resource
   def load_trained_model():
       if not os.path.exists('model.h5'):
           with st.spinner('Downloading model...'):
               gdown.download(MODEL_URL, 'model.h5', quiet=False)
       return load_model('model.h5'), 'model.h5'
   ```

3. **Hugging Face Hub** (Best for ML models)
   - Upload model to Hugging Face
   - Download using `huggingface_hub`

---

## 🌟 RECOMMENDED: Streamlit Community Cloud

**Why?**
- ✅ Completely FREE
- ✅ Easy setup (3 clicks)
- ✅ Auto-deploy on git push
- ✅ Custom subdomain
- ✅ Perfect for Streamlit apps
- ✅ No credit card needed

**Live in 5 minutes!** 🚀

---

## 🆘 Need Help?

- Streamlit Docs: https://docs.streamlit.io/streamlit-community-cloud
- GitHub LFS: https://git-lfs.github.com/
- Contact: Check Streamlit community forum

---

**Choose Option 1 (Streamlit Cloud) for fastest & easiest deployment!** 🎉
