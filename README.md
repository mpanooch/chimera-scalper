# CHIMERA AI Trading Dashboard - Website Deployment

## 📁 Files for Your Website

### **Main Website Files**
- `index.html` - Main website page with integrated CHIMERA dashboard
- `.htaccess` - Apache URL rewriting for API endpoints

### **API Files**
- `api/chimera/index.php` - PHP API endpoint to receive/serve CHIMERA data

### **Local Testing**
- `test_api.py` - Test script to verify API functionality

## 🚀 Deployment Instructions

### **Step 1: Upload Files to Your Website**

Upload these files to your `tradingtoday.com.au` root directory:

```
tradingtoday.com.au/
├── index.html
├── .htaccess
└── api/
    └── chimera/
        └── index.php
```

### **Step 2: Set File Permissions**

Make sure the API directory is writable:
```bash
chmod 755 api/chimera/
chmod 644 api/chimera/index.php
```

### **Step 3: Test the API**

Visit these URLs to test:
- `https://tradingtoday.com.au/api/chimera/health` - Should return API status
- `https://tradingtoday.com.au/` - Your main website with dashboard

### **Step 4: Update CHIMERA System**

The CHIMERA system is already configured to send data to:
`https://tradingtoday.com.au/api/chimera/update`

## 🔧 API Endpoints

### **POST /api/chimera/update**
Receives data from CHIMERA system
- Content-Type: application/json
- Saves data to `chimera_data.json`

### **GET /api/chimera/stats**
Returns current trading data for dashboard
- Returns JSON with performance, market, regime data

### **GET /api/chimera/health**
Health check endpoint
- Returns API status and available endpoints

## 📊 Dashboard Features

### **Live Data Display**
- Real-time cryptocurrency prices (BTC, ETH, SOL)
- Trading performance metrics
- AI regime detection status
- System performance stats

### **Interactive Charts**
- PnL performance over time
- Market regime confidence levels
- Recent trade history

### **Professional Design**
- Responsive layout for mobile/desktop
- Trading-focused color scheme
- Professional branding for Trading Today

## 🧪 Local Testing

### **Test with PHP Built-in Server**
```bash
# Start local PHP server
php -S localhost:8000

# Test API endpoints
python test_api.py

# Open dashboard
http://localhost:8000/index.html
```

### **Test Production API**
```bash
# Test production endpoints
python test_api.py prod
```

## 🔄 How It Works

1. **CHIMERA System** → Sends live trading data to your API
2. **PHP API** → Receives and stores data in JSON file
3. **Dashboard** → Fetches data every 2 seconds and updates display
4. **Visitors** → See live trading performance on your website

## 📈 Current CHIMERA Status

The CHIMERA system is currently running and sending data to:
- **API URL**: `https://tradingtoday.com.au/api/chimera/update`
- **Update Frequency**: Every 2 seconds
- **Data Types**: Performance, market prices, regime detection, trades

## 🛠️ Troubleshooting

### **API Not Working**
- Check file permissions on `api/chimera/` directory
- Verify `.htaccess` is uploaded and working
- Check server error logs

### **Dashboard Not Updating**
- Verify API endpoints return data
- Check browser console for JavaScript errors
- Ensure CORS headers are working

### **CHIMERA Connection Issues**
- Check if `chimera_data.json` is being created/updated
- Review `chimera_log.txt` for API request logs
- Verify CHIMERA system is running

## 📞 Support

If you need help with deployment or encounter issues:
1. Check the API health endpoint first
2. Review server error logs
3. Test with the provided test script
4. Verify file permissions and uploads

## 🎯 Next Steps

Once deployed:
1. Visit your website to see the live dashboard
2. Monitor the CHIMERA system performance
3. Customize the design/branding as needed
4. Add additional features or pages

The dashboard will automatically start showing live data once the CHIMERA system connects to your API!