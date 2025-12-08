# Quick Deployment Script for Streamlit Cloud

echo "🚀 Deploying Corn Disease Detection App"
echo "========================================"
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "📦 Initializing Git repository..."
    git init
else
    echo "✅ Git repository already initialized"
fi

# Add .gitignore
echo "📝 Creating .gitignore..."
cat > .gitignore << 'EOF'
.venv/
__pycache__/
*.pyc
.env
.DS_Store
*.log
EOF

echo ""
echo "⚠️  IMPORTANT: Model File Size Issue"
echo "======================================"
echo "Your .h5 model file is too large for GitHub (>100MB limit)"
echo ""
echo "Choose an option:"
echo "1. Use Git LFS (requires Git LFS installed)"
echo "2. Upload model to Google Drive and use link (RECOMMENDED)"
echo "3. Upload model to Hugging Face"
echo ""
read -p "Enter choice (1/2/3): " choice

if [ "$choice" == "1" ]; then
    echo "Installing Git LFS tracking for .h5 files..."
    git lfs install
    git lfs track "*.h5"
    git add .gitattributes
    echo "✅ Git LFS configured"
elif [ "$choice" == "2" ]; then
    echo ""
    echo "📤 Please follow these steps:"
    echo "1. Upload your .h5 model to Google Drive"
    echo "2. Get shareable link (Anyone with link can view)"
    echo "3. Extract File ID from link"
    echo "4. Add gdown to requirements.txt"
    echo "5. Update app.py to download model"
    echo ""
    echo "Would you like me to update requirements.txt? (y/n)"
    read -p "> " update_req
    if [ "$update_req" == "y" ]; then
        echo "gdown" >> requirements.txt
        echo "✅ Added gdown to requirements.txt"
    fi
    echo ""
    echo "⚠️  Remember to update app.py with your Google Drive link!"
else
    echo "Please upload model to Hugging Face Hub manually"
fi

echo ""
echo "📦 Adding files to git..."
git add app.py requirements.txt README.md .gitignore

echo ""
echo "💾 Creating commit..."
git commit -m "Initial commit: Corn disease detection app"

echo ""
echo "✅ Repository ready for deployment!"
echo ""
echo "📋 Next Steps:"
echo "1. Create a new repository on GitHub"
echo "2. Run: git remote add origin https://github.com/YOUR_USERNAME/your-repo.git"
echo "3. Run: git branch -M main"
echo "4. Run: git push -u origin main"
echo "5. Go to https://share.streamlit.io/"
echo "6. Deploy your app!"
echo ""
echo "🎉 Good luck with your deployment!"
