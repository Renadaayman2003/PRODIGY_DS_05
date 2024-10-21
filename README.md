

```markdown
# Traffic Crashes Data Analysis and Modeling

This project analyzes traffic crash data from Chicago and builds a machine learning model to predict the number of injuries based on various crash-related features. It includes data cleaning, feature engineering, visualizations, and a Random Forest regression model to make predictions.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Dataset](#dataset)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Modeling](#modeling)
- [Visualizations](#visualizations)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to analyze traffic crash data from Chicago, clean and preprocess the data, and create a machine learning model to predict the number of injuries resulting from a crash. We use a **Random Forest Regressor** to predict injuries based on features such as speed limit, weather conditions, lighting, and more. The project also visualizes key insights from the dataset, such as the distribution of crashes by day of the week and speed limits.

## Installation

To run this project, you will need the following dependencies:

- Python 3.x
- Google Colab or Jupyter Notebook
- Kaggle API
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn

### Step-by-Step Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/traffic-crashes-analysis
   cd traffic-crashes-analysis
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Kaggle API credentials:
   - Download your `kaggle.json` file from [Kaggle](https://www.kaggle.com/account).
   - Upload your `kaggle.json` file to your environment:
     ```bash
     mkdir ~/.kaggle
     cp kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

4. Download the dataset from Kaggle using the Kaggle API:
   ```bash
   kaggle datasets download -d anoopjohny/traffic-crashes-crashes
   ```

5. Unzip the dataset:
   ```bash
   unzip traffic-crashes-crashes.zip -d traffic_data
   ```

6. Load the dataset and explore it:
   ```python
   import pandas as pd
   df = pd.read_csv('traffic_data/Traffic_Crashes_-_Crashes.csv')
   df.head()
   ```

## Dataset

The dataset contains traffic crash data from Chicago, including information on crash location, weather conditions, speed limits, and more. The data can be downloaded from Kaggle [here](https://www.kaggle.com/datasets/anoopjohny/traffic-crashes-crashes).

### Key Features:

- `POSTED_SPEED_LIMIT`: Speed limit at the crash location
- `CRASH_HOUR`: Hour of the day when the crash occurred
- `CRASH_DAY_OF_WEEK`: Day of the week when the crash occurred
- `LATITUDE`, `LONGITUDE`: Location coordinates of the crash
- `TRAFFIC_CONTROL_DEVICE`: Traffic control devices present at the crash site
- `WEATHER_CONDITION`: Weather conditions during the crash
- `LIGHTING_CONDITION`: Lighting conditions during the crash
- `INJURIES_TOTAL`: Total number of injuries in the crash

## Data Cleaning and Preparation

1. **Handle Missing Values**: 
   We fill missing values for numeric columns with their mean and categorical columns with the most frequent value.
   ```python
   df.fillna({
       'POSTED_SPEED_LIMIT': df['POSTED_SPEED_LIMIT'].mean(),
       'TRAFFIC_CONTROL_DEVICE': 'NO CONTROLS',
       'WEATHER_CONDITION': 'CLEAR',
   }, inplace=True)
   ```

2. **Feature Engineering**:
   - We create a new feature `IS_WEEKEND` to indicate whether the crash occurred on a weekend.
   - Categorical variables such as `TRAFFIC_CONTROL_DEVICE`, `WEATHER_CONDITION`, and `LIGHTING_CONDITION` are converted into numerical form using one-hot encoding.

3. **Final Features**:
   We select relevant features such as speed limit, crash hour, location, traffic control, weather, and lighting conditions.

4. **Target Variable**:
   We use `INJURIES_TOTAL` as the target variable for predicting the number of injuries.

## Modeling

We use a **Random Forest Regressor** to predict the number of injuries based on the selected features. The steps include:

1. **Train-Test Split**:
   We split the data into training and testing sets:
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
   ```

2. **Training the Model**:
   We train a Random Forest model:
   ```python
   from sklearn.ensemble import RandomForestRegressor
   model = RandomForestRegressor(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   ```

3. **Model Evaluation**:
   We evaluate the model using **Mean Squared Error (MSE)** and **R-squared (R²)**:
   ```python
   from sklearn.metrics import mean_squared_error, r2_score
   mse = mean_squared_error(y_test, y_pred)
   r2 = r2_score(y_test, y_pred)
   print(f'Mean Squared Error: {mse}')
   print(f'R^2 Score: {r2}')
   ```

## Visualizations

We create several visualizations to explore the dataset:

1. **Number of Accidents by Day of the Week**:
   A bar plot to show how crashes vary by day:
   ```python
   sns.countplot(data=df, x='CRASH_DAY_OF_WEEK')
   ```

2. **Number of Accidents by Speed Limit**:
   A count plot to show the distribution of accidents across different speed limits:
   ```python
   sns.countplot(data=df, x='POSTED_SPEED_LIMIT')
   ```

3. **Injury Distribution by Weather Condition**:
   A box plot to show how injuries vary under different weather conditions:
   ```python
   sns.boxplot(data=df, x='WEATHER_CONDITION', y='INJURIES_TOTAL')
   ```

4. **Correlation Heatmap**:
   A heatmap showing the correlation between key numeric features:
   ```python
   sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
   ```

5. **Distribution of First Crash Types**:
   A pie chart showing the distribution of different types of crashes:
   ```python
   accident_types = df['FIRST_CRASH_TYPE'].value_counts()
   plt.pie(accident_types, labels=accident_types.index, autopct='%1.1f%%')
   ```

## Usage

To explore the dataset and run the analysis, follow these steps:

1. Download the dataset and unzip it.
2. Open the Jupyter notebook or run the Python script.
3. Follow the provided code cells or scripts to clean the data, train the model, and generate visualizations.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request with improvements or new features.

### Steps to Contribute:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature"
   ```
4. Push the changes to your branch:
   ```bash
   git push origin feature-branch
   ```
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```
