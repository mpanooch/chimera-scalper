# CHIMERA API Deployment Guide

## Files Fixed and Ready for Deployment

### 1. Core API Files
- `api/chimera/index.php` - Main API endpoint (fixed PHP syntax errors)
- `api/chimera/chimera_data.json` - Sample data file
- `.htaccess` - URL rewriting configuration

### 2. Test File
- `test_api.py` - API testing script

## Deployment Steps

### For GitHub Pages with Custom Domain:

1. **Push to GitHub:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git branch -M main
   git push -u origin main
   ```

2. **GitHub Pages Settings:**
   - Go to your repository settings
   - Enable GitHub Pages
   - Set custom domain to `tradingtoday.com.au`

### Important Notes:

⚠️ **GitHub Pages Limitation:** GitHub Pages only serves static files (HTML, CSS, JS) and does NOT execute PHP code.

### Alternative Deployment Options:

1. **Web Hosting with PHP Support:**
   - Upload files to a web host that supports PHP
   - Ensure PHP 7.4+ is available
   - Set file permissions: `chmod 644` for files, `chmod 755` for directories
   - Make sure `chimera_data.json` is writable: `chmod 666`

2. **VPS/Cloud Server:**
   - Deploy to a server with Apache/Nginx + PHP
   - Configure virtual host for your domain
   - Ensure mod_rewrite is enabled for .htaccess

## Testing After Deployment

Run the test script:
```bash
python3 test_api.py prod
```

Expected results:
- POST /api/chimera/update: Status 200 (success)
- GET /api/chimera/stats: Status 200 with JSON data

## API Endpoints

- `POST /api/chimera/update` - Update trading data
- `GET /api/chimera/stats` - Get current stats
- `GET /api/chimera/health` - Health check

## Troubleshooting

If you get "Page not found" errors:
1. Ensure files are in the correct directory structure
2. Check that PHP is enabled on your hosting
3. Verify .htaccess is being processed
4. Check file permissions

If you get 405 errors:
1. Verify the web server supports the HTTP methods
2. Check .htaccess rewrite rules
3. Ensure PHP script is receiving the correct parameters