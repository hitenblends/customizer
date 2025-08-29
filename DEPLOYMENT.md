# 🚀 Deploy to Render - Step by Step Guide

## Prerequisites
- GitHub account
- Render account (free tier available)

## Step 1: Push to GitHub
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

## Step 2: Deploy on Render

### Option A: Using render.yaml (Recommended)
1. Go to [render.com](https://render.com)
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect `render.yaml`
5. Click "Apply" to deploy

### Option B: Manual Setup
1. Go to [render.com](https://render.com)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `dtf-customizer-backend`
   - **Environment**: `Python 3`
   - **Build Command**: `cd backend && pip install -r requirements.txt`
   - **Start Command**: `cd backend && python render_start.py`
   - **Plan**: Free (or paid if you need more resources)

## Step 3: Update Frontend
1. After deployment, copy your Render URL
2. Update `dtf-customizer.html`:
   ```css
   :root {
       --backend-url: 'https://your-app-name.onrender.com';
       --use-local: false;
   }
   ```

## Step 4: Test
1. Open `dtf-customizer.html` in your browser
2. Click the "Render" toggle button
3. Test background removal and color extraction

## Environment Variables
- `PORT`: Automatically set by Render
- `PYTHON_VERSION`: Set to 3.9.16

## Troubleshooting
- Check Render logs for build errors
- Ensure all dependencies are in `requirements.txt`
- Verify Python version compatibility

## Cost
- **Free tier**: 750 hours/month, auto-sleep after 15 minutes
- **Paid plans**: Starting at $7/month for always-on service
