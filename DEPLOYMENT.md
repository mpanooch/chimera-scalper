# CHIMERA API Deployment Guide

## 🚨 Important: GitHub Pages Limitation
**GitHub Pages does NOT support PHP execution.** It only serves static files (HTML, CSS, JS).

## 📁 Files Ready for Upload:

### Core Files:
- `.htaccess` - URL rewriting (for PHP hosting)
- `api/chimera/index.php` - PHP API (for PHP hosting)
- `api/chimera/chimera_data.json` - Sample data
- `test_api.py` - Testing script

### GitHub Pages Compatible Files:
- `api/chimera/index.html` - Static HTML API simulation
- `api/chimera.js` - Vercel serverless function
- `.github/workflows/update-data.yml` - Auto-update workflow

## 🚀 Deployment Options:

### Option 1: GitHub Pages + Static API (Easiest)
1. Upload all files to GitHub
2. Enable GitHub Pages in repository settings
3. Your API will work at: `https://yourusername.github.io/yourrepo/api/chimera/index.html?action=stats`
4. **Limitation**: Data won't update automatically (read-only)

### Option 2: Vercel (Recommended)
1. Push code to GitHub
2. Connect GitHub repo to Vercel (vercel.com)
3. Deploy automatically
4. Custom domain: Point `tradingtoday.com.au` to Vercel
5. API will work at: `https://tradingtoday.com.au/api/chimera?action=stats`

### Option 3: Free PHP Hosting
**InfinityFree** (free):
1. Sign up at infinityfree.net
2. Upload PHP files via FTP
3. Point domain to their servers
4. Full PHP functionality

**000webhost** (free):
1. Sign up at 000webhost.com
2. Upload files via file manager
3. Custom domain support

### Option 4: Paid PHP Hosting
- **Hostinger** (~$2/month)
- **SiteGround** (~$3/month)
- **Bluehost** (~$3/month)

## 🧪 Testing Your Deployment:

### For GitHub Pages:
```bash
curl "https://yourusername.github.io/yourrepo/api/chimera/index.html?action=stats"
```

### For Vercel:
```bash
curl "https://your-project.vercel.app/api/chimera?action=stats"
```

### For PHP Hosting:
```bash
python3 test_api.py prod
```

## 📋 Quick Start Commands:

```bash
# 1. Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main

# 2. Enable GitHub Pages
# Go to Settings > Pages > Source: Deploy from branch > main

# 3. Test the static API
curl "https://YOUR_USERNAME.github.io/YOUR_REPO/api/chimera/index.html?action=stats"
```

## 🎯 Recommendation:

**For a professional trading dashboard**, I recommend **Vercel** because:
- ✅ Free tier available
- ✅ Custom domain support
- ✅ Serverless functions (like PHP)
- ✅ Automatic deployments from GitHub
- ✅ Can handle POST requests for data updates
- ✅ Fast global CDN

## 🔧 Next Steps:

1. **Choose your deployment method** from the options above
2. **Upload the appropriate files** for your chosen method
3. **Test the API** using the provided commands
4. **Point your domain** to the hosting service

Let me know which option you'd like to pursue!