# 📑 Cashramp Market Rate Tracker – Self-Tuning Forecasting & Alerting System

A sophisticated Python application that automatically tracks Cashramp market rates, performs intelligent model selection between Prophet and ARIMA, and sends proactive email alerts when market conditions or model performance degrade.

## 🚀 Key Features

✅ **Self-Tuning Forecasting** – Automatically compares Prophet vs ARIMA models and selects the best performer
✅ **Directional Market Alerts** – Email notifications when rates go above/below user-defined thresholds
✅ **Model Performance Monitoring** – Alerts when forecast accuracy degrades (MAPE-based)
✅ **Email Notifications with CSV** – Automated alerts include detailed forecast data as attachments
✅ **Interactive Multi-Page Dashboard** – Main dashboard + dedicated model performance analytics
✅ **Automated Data Collection** – GitHub Actions fetch rates every 30 minutes
✅ **Local Test Harness** – Comprehensive testing framework before deployment
✅ **Data Retention Management** – Automatic pruning of old records (90-day retention)

## 📂 Project Structure

```
accrue_tracker/
├── 📊 Core Application
│   ├── app.py                      # Main Streamlit dashboard with alerts
│   ├── pages/
│   │   └── 1_Model_Performance.py  # Detailed model analytics & diagnostics
│
├── 🤖 ML & Forecasting
│   ├── model_selector.py           # Auto model training & selection (Prophet/ARIMA)
│   ├── fetch_rates.py              # Cashramp API data fetcher
│   └── db_utils.py                 # Database operations & utilities
│
├── 📧 Alerting System
│   └── alert_utils.py              # Email notifications with CSV attachments
│
├── 🧪 Testing & Development
│   ├── seed_data.py                # Generate realistic test data
│   ├── local_test.py               # Comprehensive test harness
│   └── market_rates.db             # SQLite database (auto-created)
│
├── ⚙️ Deployment
│   ├── .github/workflows/fetch.yml # Automated data collection (every 30min)
│   ├── pyproject.toml              # Poetry dependency management
│   └── poetry.lock                 # Locked dependencies
│
└── 📚 Documentation
    ├── README.md                   # This file
    ├── plan.md                     # Detailed implementation plan
    └── model_selection.log         # Auto-generated model performance logs
```

## 🛠️ Quick Start

### Prerequisites
- Python 3.11+
- Poetry (for dependency management)
- Git

### 1. Installation
```bash
git clone <your-repo-url>
cd accrue_tracker
poetry install
```

### 2. Local Testing (Recommended First)
```bash
# Run comprehensive test suite with sample data
poetry run python local_test.py

# Or manually seed test data
poetry run python seed_data.py --days 7 --test-scenarios --summary
```

### 3. Launch Dashboard
```bash
poetry run streamlit run app.py
```

### 4. Fetch Real Data (Optional)
```bash
poetry run python fetch_rates.py
```

## 📊 Dashboard Features

### Main Dashboard (`app.py`)
- **Real-time Metrics**: Latest deposit/withdrawal rates with timestamps
- **Historical Visualization**: Interactive time-series charts
- **Auto Model Selection**: Real-time Prophet vs ARIMA comparison
- **Smart Alerts**:
  - Model performance degradation warnings
  - Market rate threshold breach notifications
- **Email Integration**: SMTP configuration for automated alerts
- **Data Export**: Download forecasts as CSV

### Model Performance Page (`pages/1_Model_Performance.py`)
- **Model Comparison**: Side-by-side Prophet vs ARIMA metrics
- **Forecast vs Actual**: Visual accuracy assessment
- **Residual Analysis**: Statistical diagnostic plots
- **Prophet Components**: Trend and seasonality decomposition
- **Performance Reports**: Exportable analytics

## 🤖 Intelligent Model Selection

The system automatically:

1. **Trains Multiple Models**: Prophet (time-series) and ARIMA (statistical)
2. **Validates Performance**: Uses train/validation splits with MAE, RMSE, MAPE metrics
3. **Selects Best Model**: Automatically chooses the lowest-error performer
4. **Logs Decisions**: All selections logged to `model_selection.log`
5. **Adapts Over Time**: Re-evaluates model choice as new data arrives

### Model Comparison Metrics
- **MAE** (Mean Absolute Error): Average prediction error magnitude
- **RMSE** (Root Mean Square Error): Penalizes larger errors more heavily
- **MAPE** (Mean Absolute Percentage Error): Error as percentage of actual values

## 🚨 Advanced Alert System

### 1. Model Performance Alerts
- Triggered when MAPE exceeds user-defined threshold (default: 10%)
- Includes current metrics and retraining recommendations
- Helps detect model drift and data quality issues

### 2. Directional Market Rate Alerts
- **Below Threshold Mode**: Alert when rates drop (bad for buying)
- **Above Threshold Mode**: Alert when rates rise (bad for selling)
- **CSV Attachments**: Full forecast data for detailed analysis
- **Configurable Thresholds**: User-defined trigger levels

### Email Configuration
Supports standard SMTP providers:
- **Gmail**: `smtp.gmail.com:587`
- **Outlook**: `smtp-mail.outlook.com:587`
- **Custom SMTP**: Any standard SMTP service

## 🔄 Automated Data Collection

### GitHub Actions Workflow
- **Frequency**: Every 30 minutes (configurable)
- **Data Fetching**: Calls Cashramp GraphQL API
- **Database Updates**: Commits new data automatically
- **Data Retention**: Prunes records older than 90 days
- **Error Handling**: Continues operation even if single fetch fails
- **Monitoring**: GitHub Actions summary with database statistics

### Manual Triggers
```bash
# Force immediate data fetch
poetry run python fetch_rates.py

# Seed development data
poetry run python seed_data.py

# Clean old data
poetry run python -c "from db_utils import prune_old_data; print(f'Deleted {prune_old_data(30)} records')"
```

## 🧪 Testing Framework

### Comprehensive Test Suite (`local_test.py`)
```bash
poetry run python local_test.py
```

Tests include:
- **Database Operations**: Schema creation, data insertion/retrieval
- **Model Training**: Prophet and ARIMA training with validation
- **Forecast Generation**: Short and long-term prediction accuracy
- **Alert System**: Mock email testing (console output for safety)
- **CSV Export**: Data formatting and download functionality

### Test Data Generation (`seed_data.py`)
```bash
# Generate 7 days of realistic data
poetry run python seed_data.py --days 7 --countries GH NG --summary

# Add specific test scenarios for alerts
poetry run python seed_data.py --test-scenarios
```

## ☁️ Deployment

### Streamlit Cloud
1. Push repository to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set `app.py` as main file
4. Configure SMTP secrets in Streamlit Cloud environment
5. Deploy!

### Environment Variables (Optional)
For production email alerts, set these in Streamlit Cloud:
```
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
DEFAULT_RECIPIENT=alerts@yourcompany.com
DEFAULT_SENDER=cashramp-tracker@yourcompany.com
```

## 📈 Usage Examples

### Basic Forecasting
```python
from model_selector import select_best_model, generate_forecast
from db_utils import get_rates_data

# Load data and train best model
df = get_rates_data("GH")
model_name, model, metrics = select_best_model(df, ["prophet", "arima"])

# Generate 24-hour forecast
forecast = generate_forecast(model_name, model, 24, df["timestamp"].iloc[-1])
print(f"Using {model_name} model with MAPE: {metrics['mape']:.2f}%")
```

### Custom Alerts
```python
from alert_utils import send_rate_threshold_alert

# Configure SMTP
smtp_config = {
    "smtp_server": "smtp.gmail.com",
    "smtp_port": 587,
    "smtp_user": "your-email@gmail.com",
    "smtp_pass": "your-password"
}

# Send custom alert with forecast attachment
send_rate_threshold_alert(
    alert_mode="Below Threshold (bad for buying)",
    threshold=9.0,
    min_rate=8.5,
    max_rate=10.2,
    horizon=24,
    recipient="trader@company.com",
    sender="alerts@company.com",
    smtp_config=smtp_config,
    forecast_df=forecast_data
)
```

## 🔧 Configuration

### Model Settings
- **Validation Split**: Fraction of data for model validation (default: 0.2)
- **Models to Compare**: Enable/disable Prophet and ARIMA (default: both)
- **Performance Threshold**: MAPE level that triggers model alerts (default: 10%)

### Alert Settings
- **Rate Threshold**: Value that triggers market alerts (configurable)
- **Alert Mode**: "Below" or "Above" threshold monitoring
- **Email Recipients**: Multiple recipients supported
- **Forecast Horizon**: Hours ahead to forecast (12-168 hours)

### Data Settings
- **Collection Frequency**: GitHub Actions cron schedule (default: every 30min)
- **Retention Period**: Days of historical data to keep (default: 90)
- **Countries**: Market regions to monitor (default: ["GH"])

## 🔮 Future Enhancements

### Planned Features
- [ ] **Multi-Country Dashboard**: Comparative analysis across regions
- [ ] **Advanced Models**: XGBoost, LSTM neural networks
- [ ] **Slack Integration**: Alternative to email notifications
- [ ] **Real-time Streaming**: WebSocket-based live updates
- [ ] **API Endpoints**: REST API for external integrations
- [ ] **Cloud Database**: PostgreSQL/Supabase migration for scaling

### Model Improvements
- [ ] **Feature Engineering**: Technical indicators, economic data
- [ ] **Ensemble Methods**: Combine multiple model predictions
- [ ] **Hyperparameter Tuning**: Automated model optimization
- [ ] **Anomaly Detection**: Identify unusual market conditions

## 🐛 Troubleshooting

### Common Issues

#### No Data in Dashboard
```bash
# Check if database has data
poetry run python -c "from db_utils import get_rates_data; print(len(get_rates_data()))"

# Seed test data if empty
poetry run python seed_data.py --days 3 --summary
```

#### Model Training Errors
```bash
# Check data quality and quantity
poetry run python -c "
from db_utils import get_rates_data
df = get_rates_data()
print(f'Records: {len(df)}, Null values: {df.isnull().sum().sum()}')
"

# Need 20+ records for reliable model training
```

#### Email Alerts Not Working
1. Verify SMTP credentials in sidebar
2. Check firewall/network restrictions
3. Test with mock output: `poetry run python local_test.py`
4. Gmail users: Enable App Passwords instead of regular password

#### GitHub Actions Failing
1. Check repository permissions for GitHub Actions
2. Verify Poetry configuration in workflow
3. Review action logs for specific error messages
4. Ensure Cashramp API is accessible

### Performance Optimization

#### Large Datasets
```python
# Use data limits for faster testing
from db_utils import get_rates_data
df = get_rates_data("GH", limit=1000)  # Last 1000 records only
```

#### Memory Usage
```python
# Clear model cache if needed
import streamlit as st
st.cache_resource.clear()
st.cache_data.clear()
```

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`poetry run python local_test.py`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📞 Support

- **Documentation**: See `plan.md` for detailed implementation notes
- **Issues**: Report bugs on GitHub Issues
- **Performance**: Check `model_selection.log` for model training details
- **Testing**: Use `poetry run python local_test.py` for diagnostics

---

**Built with ❤️ using Prophet, ARIMA, Streamlit, and Poetry**