# NLC Resource Optimization System

A comprehensive machine learning-powered resource optimization system designed to optimize logistics operations by predicting driver fatigue, delivery delays, and vehicle maintenance risks.

## 🎯 Project Overview

The **NLC Resource Optimization System** leverages machine learning and combinatorial optimization to:

- **Driver Fatigue Detection**: Identify high-risk drivers using XGBoost classification
- **Delivery Delay Prediction**: Forecast potential delivery delays with Random Forest models
- **Vehicle Health Monitoring**: Assess vehicle maintenance risks using Gradient Boosting
- **Resource Optimization**: Optimize task-driver assignments using OR-Tools

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Requirements](#data-requirements)
- [Models](#models)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

### 1. Driver Fatigue Classification
- Analyzes sleep patterns, accident history, and continuous driving hours
- XGBoost classifier with 85%+ accuracy
- Prevents risky assignments of fatigued drivers

### 2. Delivery Delay Prediction
- Predicts delivery delays based on route and driver characteristics
- Random Forest ensemble model
- Proactive delay management

### 3. Vehicle Health Assessment
- Monitors mileage and maintenance history
- Gradient Boosting classifier
- Predictive maintenance scheduling

### 4. Optimization Engine
- Assignment of drivers to tasks
- Constraint-based optimization using Google OR-Tools
- Minimizes operational costs while respecting constraints

## 📁 Project Structure

```
NLC_Resource_Optimization_System/
├── README.md                          # Project documentation
├── LICENSE                           # MIT License
├── .gitignore                        # Git ignore rules
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package setup configuration
├── config.py                         # Configuration management
│
├── src/
│   ├── __init__.py
│   ├── main.py                       # Entry point
│   ├── data_preprocessing.py         # Data loading & preprocessing
│   ├── fatigue_model.py              # Driver fatigue classifier
│   ├── delivery_model.py             # Delivery delay predictor
│   ├── vehicle_model.py              # Vehicle health classifier
│   ├── delay_model.py                # Additional delay modeling
│   └── optimization_engine.py        # OR-Tools optimization
│
├── data/
│   ├── drivers.csv                   # Driver data
│   ├── vehicles.csv                  # Vehicle data
│   └── deliveries.csv                # Delivery records
│
├── models/
│   ├── fatigue_classifier.pkl        # Trained fatigue model
│   ├── delivery_rf.pkl               # Trained delivery model
│   └── vehicle_gb.pkl                # Trained vehicle model
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py         # Data preprocessing tests
│   ├── test_models.py                # Model tests
│   └── test_optimization.py          # Optimization engine tests
│
├── docs/
│   ├── DATA_DICTIONARY.md            # Data schema documentation
│   ├── MODEL_ARCHITECTURE.md         # Model specifications
│   └── API_REFERENCE.md              # Function documentation
│
├── notebooks/
│   ├── exploration.ipynb             # Data exploration
│   └── evaluation.ipynb              # Model evaluation
│
└── logs/
    └── .gitkeep
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/NLC_Resource_Optimization_System.git
cd NLC_Resource_Optimization_System
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/macOS
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import src; print('Installation successful!')"
```

## 📊 Usage

### Basic Usage

```python
from src.data_preprocessing import load_datasets, preprocess_drivers
from src.fatigue_model import train_fatigue_model

# Load data
drivers, vehicles, deliveries = load_datasets()

# Preprocess
X_train, X_test, y_train, y_test = preprocess_drivers(drivers)

# Train model
model = train_fatigue_model(X_train, X_test, y_train, y_test)
```

### Running the Full Pipeline

```bash
python src/main.py
```

### Using Optimization Engine

```python
from src.optimization_engine import optimize

# Run optimization
results = optimize(drivers_data)
print(results)  # [(driver_id, task_id), ...]
```

## 📈 Data Requirements

### drivers.csv
```
Columns:
- Driver_ID: Unique identifier
- Sleep_Hours_Last_24h: Hours of sleep
- Accident_History_Count: Number of accidents
- Continuous_Driving_Hours: Hours without break
- Age: Driver age
- Experience_Years: Years of experience
```

### vehicles.csv
```
Columns:
- Vehicle_ID: Unique identifier
- Mileage: Total mileage
- Breakdowns_Past: Number of breakdowns
- Age: Vehicle age
- Capacity: Load capacity
- Fuel_Type: Type of fuel
```

### deliveries.csv
```
Columns:
- Delivery_ID: Unique identifier
- Driver_ID: Assigned driver
- Vehicle_ID: Assigned vehicle
- Planned_Delivery_Time: Expected time
- Actual_Delivery_Time: Actual time
- Distance: Delivery distance
- Weather_Condition: Weather conditions
```

## 🤖 Models

### Model Comparison

| Model | Algorithm | Task | Accuracy |
|-------|-----------|------|----------|
| Fatigue Classifier | XGBoost | Driver Risk Detection | ~85% |
| Delivery Model | Random Forest | Delay Prediction | ~78% |
| Vehicle Model | Gradient Boosting | Maintenance Risk | ~82% |

### Model Training

Each model includes:
- Train/test split (80/20)
- Cross-validation
- Confusion matrix analysis
- Classification reports
- Model serialization (joblib)

## ⚙️ Configuration

Edit `config.py` for environment-specific settings:

```python
# Database
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "nlc_optimization"

# Model Parameters
MODEL_RANDOM_STATE = 42
TEST_SIZE = 0.2

# Optimization
NUM_DRIVERS = 10
NUM_TASKS = 10
SOLVER_TIME_LIMIT = 60

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "logs/app.log"
```

## 🧪 Testing

Run tests to verify functionality:

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/

# Verbose output
pytest -v tests/
```

## 📝 Model Architecture

### Driver Fatigue Model
```
XGBoost Classifier
├── Input: Driver features (sleep, accident history, driving hours)
├── Hidden: 5 layers with 200 trees
└── Output: Binary classification (Fatigue Risk: 0 or 1)
```

### Delivery Delay Model
```
Random Forest Classifier
├── Input: Delivery & driver features
├── Ensemble: 200 decision trees
└── Output: Binary classification (Delay Risk: 0 or 1)
```

### Vehicle Health Model
```
Gradient Boosting Classifier
├── Input: Vehicle features (mileage, breakdowns)
├── Stages: Iterative boosting
└── Output: Binary classification (Maintenance Risk: 0 or 1)
```

## 🔍 Logging & Monitoring

Logs are stored in `logs/app.log`:

```
✅ Drivers Loaded Successfully!
✅ Vehicles Loaded Successfully!
✅ Deliveries Loaded Successfully!
📌 Accuracy: 85.45 %
💾 Model Saved Successfully!
```

## 🛠️ Development

### Code Style
- Follow PEP 8
- Use type hints
- Document functions with docstrings

### Adding New Models

1. Create `new_model.py` in `src/`
2. Implement training function
3. Add to `main.py`
4. Create tests in `tests/`

```python
def train_new_model(X_train, X_test, y_train, y_test):
    """
    Train new classification model.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training labels
        y_test: Testing labels
    
    Returns:
        Trained model object
    """
    model = YourModel()
    model.fit(X_train, y_train)
    return model
```

## 📊 Results & Evaluation

After running the pipeline, check:
- `models/` for saved models
- `logs/` for training logs
- Console output for metrics

## 🚨 Troubleshooting

### Missing Data Files
```bash
# Ensure CSV files exist in data/
ls -la data/
```

### Module Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python path
python -c "import sys; print(sys.path)"
```

### Model Accuracy Issues
- Check data quality and preprocessing
- Review feature engineering
- Adjust hyperparameters in model files
- Validate train/test split

## 📚 Documentation

- **DATA_DICTIONARY.md**: Detailed data schema
- **MODEL_ARCHITECTURE.md**: Model specifications
- **API_REFERENCE.md**: Function documentation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Author

**M Talha Farooq**
- Email: talha.farooq@example.com
- LinkedIn: [Your Profile]
- GitHub: [@yourusername]

## 🙏 Acknowledgments

- Google OR-Tools for optimization
- scikit-learn for ML algorithms
- XGBoost for gradient boosting
- pandas for data manipulation

## 📞 Support

For issues and questions:
1. Check [GitHub Issues](https://github.com/yourusername/NLC_Resource_Optimization_System/issues)
2. Submit a new issue with:
   - Detailed description
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and dependencies

## 🗺️ Roadmap

- [ ] REST API endpoint development
- [ ] Real-time predictions
- [ ] Dashboard UI
- [ ] Database integration
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure)
- [ ] Advanced forecasting
- [ ] Mobile application

---

**Last Updated**: 2024
**Version**: 1.0.0
