# 🍽️ Restaurant Data: Quantity Sold Prediction

This is a term project for the DS514/515 Data Science course at SWU, Thailand.

The project aims to explore insights from restaurant sales data and identify the key factors influencing quantity sold. In addition, we developed predictive models for quantity sold using Linear Regression and regularized regression techniques (Ridge, Lasso, and Elastic Net) to support restaurants in improving their performance and developing effective strategies to enhance sales.

🔍 Project Workflow

The modeling workflow includes:

✔️ About dataset

✔️ Data cleaning

✔️ Exploratory Data Analysis (EDA)

✔️ Feature selection

✔️ Data pre-processing

✔️ Data pipeline development and Linear Regression

✔️ Model building and evaluation

---

สวัสดีทุกคน!!! ขอต้อนรับทุกคนเข้าสู่โครงงานของรายวิชา DS514/515 Data Analytics ระดับปริญญาโท มหาวิทยาลัยศรีนครินทรวิโรฒ

โครงงานนี้มีวัตถุประสงค์เพื่อวิเคราะห์ข้อมูลยอดขายของร้านอาหารและระบุปัจจัยสำคัญที่มีผลต่อปริมาณสินค้าที่ขาย นอกจากนี้ยังได้พัฒนาโมเดลทำนายปริมาณสินค้าที่ขายโดยใช้ Linear Regression และ Regularized Linear Regression ได้แก่ Ridge, Lasso, และ Elastic Net เพื่อช่วยให้ร้านอาหารสามารถปรับปรุงประสิทธิภาพการดำเนินงานและวางแผนกลยุทธ์เพื่อกระตุ้นยอดขายได้อย่างมีประสิทธิผล

🔍 Project Workflow

กระบวนการสร้างโมเดฃประกอบด้วย:

✔️ รายละเอียดเกี่ยวกับชุดข้อมูล (About dataset)

✔️ การทำความสะอาดข้อมูล (Data cleaning)

✔️ การวิเคราะห์ข้อมูลเชิงสำรวจ (EDA)

✔️ การเลือกคุณลักษณะสำคัญ (Feature selection)

✔️ การเตรียมข้อมูลก่อนสร้างโมเดล (Pre-processing)

✔️ การสร้าง Data pipeline และทำโมเดล Linear regression

✔️ การสร้างและประเมินโมเดล (Model building & evaluation)

---

## 1️⃣ รายละเอียดเกี่ยวกับชุดข้อมูล (About dataset)

ข้อมูลชุดนี้นำมาจาก Kaggle โพสต์โดย คุณ Alexand Chen (https://www.kaggle.com/datasets/alexandchen/restaurant-sales-report-2024-2025)

Dataset ในรูปแบบ csv ไฟล์ สามารถดาวน์โหลดได้จาก restaurant_dataset.csv (https://github.com/ThitiwutM/Restaurant-data-Quantity-sold-prediction_DS514-515/blob/main/restaurant_dataset.csv)

📊 ชุดข้อมูลนี้ประกอบด้วย:

- 10,000 รายการข้อมูล (records) ข้อมูลระหว่าง 1 ม.ค. 2024 ถึง 1 ม.ค. 2025

- 13 ตัวแปร (attributes)

📁 ลักษณะของข้อมูล

- 9 ข้อมูลทั่วไป ได้แก่ date, restaurant_id, restaurant_type, menu_item_name, meal_type, key_ingredients_tags, has_promotion (T/F), special_event (T/F), weather_condition (Sunny / Cloudy / Rainy)

- 4 ข้อมูลเชิงตัวเลข (numerical data) เกี่ยวกับ ต้นทุน ราคาขาย และ ยอดขาย ได้แก่ typical_ingredient_cost, observed_market_price, actual_selling_price, quantity_sold

📋 DATA DICTIONARY

| ตัวแปร (Attribute)      | คำอธิบาย (Description)               | ประเภทข้อมูล (Data Type) | ช่วงข้อมูล/ตัวอย่าง (Valid Range / Example) |
| ----------------------- | ------------------------------------ | ------------------------ | ------------------------------------------- |
| date                    | วันที่ของการขาย (รูปแบบ dd/MM/yyyy)  | Date/DateTime            | 01/01/2024 – 01/01/2025                     |
| restaurant_id           | รหัสร้านอาหาร                        | Number/Float             | 1 – 50                                      |
| restaurant_type         | ประเภทร้านอาหาร                      | Text                     | Food Stall / Casual / Fine Dining           |
| menu_item_name          | ชื่อเมนูอาหาร/เครื่องดื่ม            | Text                     | Kaya Toast Set / Cendol / Teh Tarik         |
| meal_type               | ประเภทมื้ออาหาร                      | Text                     | Breakfast / Lunch / Dinner                  |
| key_ingredients_tags    | วัตถุดิบหลักของเมนู (คั่นด้วย comma) | Text                     | white bread, kaya, butter, …                |
| typical_ingredient_cost | ต้นทุนวัตถุดิบโดยประมาณ              | Number/Float             | 0.8 – 9                                     |
| observed_market_price   | ราคาตลาดที่สังเกตได้ของเมนู          | Number/Float             | 1.46 – 56.29                                |
| actual_selling_price    | ราคาขายจริงให้ลูกค้า                 | Number/Float             | 1.36 – 83.09                                |
| quantity_sold           | จำนวนที่ขายได้ (หน่วยเป็นจาน/หน่วย)  | Number/Float             | 0 – 1668                                    |
| has_promotion           | มีโปรโมชันหรือไม่                    | Boolean                  | True / False                                |
| special_event           | อยู่ในช่วงวันพิเศษหรือไม่            | Boolean                  | True / False                                |
| weather_condition       | สภาพอากาศในวันขาย                    | Text                     | Sunny / Cloudy / Rainy                      |

---

## 2️⃣ การทำความสะอาดข้อมูล (Data Cleaning)

🧹 Cleansing Summary

นำเข้าข้อมูลและตรวจสอบค่าที่หายไป (Missing Data)
- นำเข้า Library
- นำเข้าข้อมูลจากไฟล์ csv
- ตรวจสอบค่าที่หายไป → ไม่พบค่า NULL
- ข้อมูลมีจำนวน 10,000 แถว และ 13 คอลัมน์

การปรับประเภทข้อมูล (Data Type Conversion)
- แปลงคอลัมน์ date จากชนิดข้อมูล “object” → datetime64[ns]
- แปลงคอลัมน์ restaurant_id จาก “int64” → object

```python
# import numpy, pandas, matplotlib, seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# read csv file
data = pd.read_csv('/content/drive/MyDrive/DS514.csv')
data.head()

# check data.info
data.info()

# change date from 'object' to 'date'
data['date'] = pd.to_datetime(data['date'])
data.info()

# change restaurant_id from 'int' to 'object'
data['restaurant_id'] = data['restaurant_id'].astype(str)
data.info()

# check data is NULL
data.isnull().sum()
```
data  information

![](https://github.com/ThitiwutM/Restaurant-data-Quantity-sold-prediction_DS514-515/blob/main/data_info.jpg)

check NULL value

![](https://github.com/ThitiwutM/Restaurant-data-Quantity-sold-prediction_DS514-515/blob/main/Check_NULL.jpg)

---

## 3️⃣ การวิเคราะห์ข้อมูลเชิงสำรวจ (Exploratory Data Analysis: EDA)

EDA 1 – ภาพรวมของปริมาณการขาย (Quantity Sold Overview)

ผลการวิเคราะห์ค่าเฉลี่ยและผลรวมของ quantity_sold พบว่า
- ร้านประเภท Fine Dining มียอดขายต่ำที่สุดเมื่อเทียบกับร้านทั้ง 50 แห่ง
- กราฟ Box-Whisker แสดงให้เห็นว่า ค่ามัธยฐานของ Fine Dining ต่ำกว่าทุกประเภทร้านอาหาร อย่างชัดเจน

```python
# summary table of average_quantity sold by restaurant_id sorting by descending of average_quantity sold
restaurant = data.groupby('restaurant_id')['quantity_sold'].mean().sort_values(ascending=False)
display(restaurant)

# bar plot the 'restaurant' data sorting by descending of average_quantity sold
plt.figure(figsize=(12,5))
plt.bar(restaurant.index, restaurant.values)
plt.ylabel('quantity_sold')
plt.show()

# summary table of sum_quantity sold by restaurant_id sorting by descending of sum_quantity sold
restaurant2 = data.groupby('restaurant_id')['quantity_sold'].sum().sort_values(ascending=False)
display(restaurant2)

# bar plot the 'restaurant2' data sorting by descending of sum_quantity sold
plt.figure(figsize=(12,5))
plt.bar(restaurant2.index, restaurant2.values)
plt.ylabel('quantity_sold')
plt.show()

# plot box plot of quantity_sold by restaurant type from data
plt.figure(figsize=(6,5))
sns.boxplot(x='restaurant_type', y='quantity_sold', data = data)
```
![](https://github.com/ThitiwutM/Restaurant-data-Quantity-sold-prediction_DS514-515/blob/main/EDA1_bargraph.jpg)

![](https://github.com/ThitiwutM/Restaurant-data-Quantity-sold-prediction_DS514-515/blob/main/EDA1_boxplot.jpg)


EDA 2 – การดูแนวโน้มตลอด 1 ปี และความสัมพันธ์ระหว่างตัวแปร (Trend & Correlation Analysis)
- จากกราฟ Time-series ไม่พบรูปแบบหรือแนวโน้ม (Trend) ที่ชัดเจน
- แต่ค่าเฉลี่ยรายปีของ quantity_sold ในประเภทร้าน Fine Dining ต่ำกว่าร้านประเภทอื่นอย่างมีนัยสำคัญ

ผลการวิเคราะห์ความสัมพันธ์ (Correlation) พบว่า:
- ตัวแปร cost, market_price, และ selling_price มีความสัมพันธ์เชิงบวกที่มีค่าสูงมาก
- ตัวแปร quantity_sold มีความสัมพันธ์เชิงลบในระดับปานกลางกับราคาต่าง ๆ → หมายความว่า เมื่อราคาเพิ่มขึ้น ปริมาณการขายมีแนวโน้มลดลง

```python
# plot the line trend of quantity_sold by date and group by each restaurant_type (5 graph subplot)
plt.figure(figsize=(8,10))
for i, restaurant_type in enumerate(data['restaurant_type'].unique()):
    plt.subplot(5, 1, i+1)
    sns.lineplot(x='date', y='quantity_sold', data=data[data['restaurant_type'] == restaurant_type], label=restaurant_type)
    plt.title(restaurant_type)
    plt.xlabel('date')
    plt.ylabel('quantity_sold')
plt.tight_layout()
plt.show()

# correlation matrix plot of 'typical_ingredient_cost'	'observed_market_price'	'actual_selling_price'	'quantity_sold'
corr = data[['typical_ingredient_cost', 'observed_market_price', 'actual_selling_price', 'quantity_sold']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
```

![](https://github.com/ThitiwutM/Restaurant-data-Quantity-sold-prediction_DS514-515/blob/main/EDA_trend.jpg)

![](https://github.com/ThitiwutM/Restaurant-data-Quantity-sold-prediction_DS514-515/blob/main/EDA2_correlation.jpg)


EDA 3 – ผลของสภาพอากาศ / โปรโมชั่น / อีเวนต์ (Weather, Promotion & Event Effects)

- ปริมาณการขายเฉลี่ยในวันที่ Sunny (251) และ Cloudy (249) สูงกว่าวันที่ Rainy (227) แต่ความแตกต่างค่อนข้างเล็ก ทำให้สังเกตได้ยากในกราฟ Box-Whisker

- ปริมาณการขายเฉลี่ยในวันที่มี โปรโมชั่น (482) สูงกว่าวันที่ ไม่มีโปรโมชั่น (251) อย่างชัดเจน

- วันที่มี กิจกรรมพิเศษ (363) มียอดขายสูงกว่าวันที่ ไม่มีอีเวนต์ (282)

```python
# summary table of average_quantity sold by each weather_condition
season = data.groupby('weather_condition')['quantity_sold'].mean()
display(season)

# summary table of average_quantity sold by each weather_condition
season = data.groupby('weather_condition')['quantity_sold'].mean()
display(season)

# plot box-whisker plot of quantity_sold by each weather_condition
plt.figure(figsize=(3,5))
sns.boxplot(x='weather_condition', y='quantity_sold', data=data)

# plot box-whisker plot of quantity_sold by has_promotion
plt.figure(figsize=(3,5))
sns.boxplot(x='has_promotion', y='quantity_sold', data=data)

# plot box-whisker plot of quantity_sold by special_event
plt.figure(figsize=(3,5))
sns.boxplot(x='special_event', y='quantity_sold', data=data)
```

![](https://github.com/ThitiwutM/Restaurant-data-Quantity-sold-prediction_DS514-515/blob/main/EDA_Weather-Promotion-Event.jpg)

---

## 4️⃣ การเลือกคุณลักษณะสำคัญ (Feature Selection)

โมเดลจะทำการทำนาย Quantity_sold โดยใช้ Features ทั้งแบบ numeric และแบบ categorical ดังนี้:
- numeric: lag1, actual_selling_price
- categorical: weather_condition, has_promotion, special_event

📌 การเลือกข้อมูลที่ใช้สร้างโมเดล

เลือกข้อมูลเฉพาะ 5 ร้านสุดท้าย ที่มียอดขายน้อยที่สุด ได้แก่
restaurant_id = ‘32’, ‘6’, ‘42’, ‘10’, ‘38’
เพื่อใช้ในการสร้างโมเดลคาดการณ์ยอดขาย (Quantity_sold)

จำนวนข้อมูลทั้งหมดหลังการคัดเลือก = 1,027 records

🆕 การสร้างตัวแปรใหม่ (“lag1”)

สร้างตัวแปรใหม่ชื่อ lag1 ซึ่งเป็นค่าของ quantity_sold ของวันก่อนหน้า
เพื่อนำมาใช้เป็น Feature เพิ่มเติมในโมเดล เพื่อช่วยให้โมเดลเรียนรู้รูปแบบของยอดขายในลักษณะ time dependency

```python
# Prepare more data for modeling
# focus on restaurant_id '32' '6' '42' '10 '38'
model_data = data[data['restaurant_id'].isin(['32', '6', '42', '10', '38'])]
model_data.head()

# summary of how many row record in each restaurant_id of model_data
model_data.groupby('restaurant_id').size()

# create lag1 of quantity_sold and diff(quantity_sold and lag1) of model_data
model_data['lag1'] = model_data.groupby(['restaurant_id','menu_item_name'])['quantity_sold'].shift(1)
model_data['diff'] = model_data['quantity_sold'] - model_data['lag1']
model_data.tail()
```

---

## 5️⃣ การเตรียมข้อมูลก่อนสร้างโมเดล (Pre-processing)

ขั้นตอนการเตรียมข้อมูลก่อนสร้างโมเดลประกอบด้วย:
- Import library
- Set “features” and “target”
- Remove “NULL” value of lag1
- Train/test split (80/20)
- Define numerical features / categorical features
- Check size of data

Pre-processing
- Numerical Features → StandardScaler
- Categorical Features → One-Hot Encoder

```python
rom sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# --- 1. Feature and Target Definition ---

# Define the target variable (y) and the predictor variables (X)
X = model_data[['lag1', 'actual_selling_price', 'weather_condition', 'has_promotion', 'special_event']]
y = model_data['quantity_sold']

# Remove NULL value from lag1
X = X.dropna()
y = y[X.index]

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# shuffle=False is crucial for time-series data to maintain chronological order

# Define feature types for the ColumnTransformer
numerical_features = ['lag1', 'actual_selling_price']
categorical_features = ['weather_condition', 'has_promotion', 'special_event']

# summary data_size of X_train, X_test, y_train, y_test
print("X_train size:", X_train.shape)
print("X_test size:", X_test.shape)
print("y_train size:", y_train.shape)
print("y_test size:", y_test.shape)

# --- 2. Data Preprocessing Pipeline (Scaling and Encoding) ---

# Create the preprocessing pipeline using ColumnTransformer
# Apply Standard Scaling to numerical features
# Apply One-Hot Encoding to categorical features (handle_unknown='ignore' prevents error on unseen category)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features)
    ],
    remainder='passthrough' # Keep any columns not specified (if any)
)
```

---

## 6️⃣ การสร้าง Data Pipeline และการสร้างโมเดล Linear Regression

เพื่อให้ขั้นตอนการเตรียมข้อมูลและการสร้างโมเดลเป็นระบบอัตโนมัติ กลุ่มของเราได้สร้าง Data Pipeline ที่ประกอบด้วย:

🔧 Data Pipeline = Preprocessor + Linear Regression

Pipeline จะทำงานครบทุกขั้นตอน ไม่ว่าจะเป็น
- Standard Scaling สำหรับ numeric features
- One-Hot Encoding สำหรับ categorical features
- และการเทรนโมเดล Linear Regression
ทั้งหมดรวมอยู่ในขั้นตอนเดียว ช่วยลดความผิดพลาดและทำให้การทำงานเป็นระบบมากขึ้น

ผลลัพธ์ของโมเดล Linear Regression
|            | ค่า                                       |
| ---------------- | ----------------------------------------- |
| **Coefficients** | 37.02, -62.21, 14.57, 13.10, 86.55, 97.70 |
| **Intercept**    | 163.90                                    |
| **R² Score**     | **0.4315**                                |
| **MAE**          | 96.96                                     |
| **RMSE**         | 127.86                                    |


```python
# --- 3. Model 1: Baseline Linear Regression ---

# Create the full pipeline for Linear Regression
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('regressor', LinearRegression())])

print("--- Baseline Linear Regression ---")
lr_pipeline.fit(X_train, y_train)
lr_predictions = lr_pipeline.predict(X_test)
print('Coefficients:', lr_pipeline.named_steps['regressor'].coef_)
print('Intercept:', lr_pipeline.named_steps['regressor'].intercept_)
print("R2 Score:", r2_score(y_test, lr_predictions))
print("MAE:", mean_absolute_error(y_test, lr_predictions))
print("RMSE:", root_mean_squared_error(y_test, lr_predictions))
```

---

## 7️⃣ การสร้างและประเมินโมเดล (Model Building & Evaluation)

ในขั้นตอนนี้ กลุ่มของเราให้ความสนใจกับการสร้างโมเดล Regularized Linear Regression ร่วมกับ Polynomial Features เพื่อเพิ่มความสามารถในการจับความสัมพันธ์แบบไม่เชิงเส้น (non-linear relationship) ระหว่างตัวแปรต้นและยอดขาย (quantity_sold)

🔧 Model Pipeline

Pipeline ถูกออกแบบให้รวมทุกขั้นตอนการเตรียมข้อมูลและสร้างโมเดลเข้าไว้ด้วยกัน:

Pipeline = Preprocessor + PolynomialFeatures + Ridge / Lasso / ElasticNet

ซึ่งประกอบด้วย:
- Standard Scaling สำหรับ numeric features
- One-Hot Encoding สำหรับ categorical features
- Polynomial Features (degree 1–10)
- Regularized Regression (Ridge / Lasso / Elastic Net)

⚙️ Hyperparameter Tuning

เพื่อค้นหาค่าพารามิเตอร์ที่ดีที่สุดสำหรับแต่ละโมเดล เราใช้เทคนิค GridSearchCV

🔍 Param Grid ที่ใช้
- Polynomial degree: 1 → 10
- Alpha (α): 10⁻⁶ → 10³
- Elastic Net l1_ratio: 0 → 1
- Cross-validation: 5 folds
- Scoring metric: 'r2'

🏆 ผลลัพธ์จากการประเมินโมเดล (Evaluation Metrics)

หลังจากฝึกโมเดลและปรับจูนพารามิเตอร์แล้ว เราประเมินผลด้วยตัวชี้วัดหลัก:
- R² Score (Coefficient of Determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Square Error)

|                      | Ridge  | Lasso        | Elastic Net |
| -------------------------- | ------ | ------------ | ----------- |
| **Best Polynomial Degree** | 3      | 3            | 4           |
| **Best Alpha**             | 100.0  | 1            | 0.01        |
| **Best L1 Ratio**          | –      | –            | 0.33        |
| **R² (Train)**             | 0.5365 | 0.5470       | 0.6192  |
| **R² (Test)**              | 0.5444 | **0.5639** ⭐ | 0.5285      |
| **MAE (Train)**            | 79.17  | 78.91        | 72.14   |
| **MAE (Test)**             | 87.38  | **86.09**  ⭐  | 88.20       |
| **RMSE (Train)**           | 101.38 | 100.23       | 91.89   |
| **RMSE (Test)**            | 114.47 | **111.99**  ⭐ | 116.45      |

![](https://github.com/ThitiwutM/Restaurant-data-Quantity-sold-prediction_DS514-515/blob/main/Hyperparameter_tuning_lasso.jpg)

```python
# --- 4. Model 2 Ridge Regression with polynomial feature ---

from sklearn.preprocessing import PolynomialFeatures

polynomial_ridge_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('polynomialfeatures', PolynomialFeatures()), # Placeholder, degree will be tuned
    ('regressor', Ridge())
])

param_grid = {
    # Alpha values for Ridge regularization (log scale for better searching)
    'regressor__alpha': np.logspace(-6, 3, 10),
    # Polynomial degrees to test (1 to 10)
    'polynomialfeatures__degree': np.arange(1, 11)
}

print("--- Ridge Regression with Polynomial Features Tuning ---")
grid_search = GridSearchCV(
    polynomial_ridge_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2'
)

grid_search.fit(X_train, y_train)

Ridge_best_estimator = grid_search.best_estimator_
Ridge_predictions = Ridge_best_estimator.predict(X_test)

best_params = grid_search.best_params_
best_degree = best_params['polynomialfeatures__degree']
best_alpha = best_params['regressor__alpha']

# Evaluation
r2_train = r2_score(y_train, Ridge_best_estimator.predict(X_train))
r2_test = r2_score(y_test, Ridge_predictions)
mae_train = mean_absolute_error(y_train, Ridge_best_estimator.predict(X_train))
mae_test = mean_absolute_error(y_test, Ridge_predictions)
rmse_train = root_mean_squared_error(y_train, Ridge_best_estimator.predict(X_train))
rmse_test = root_mean_squared_error(y_test, Ridge_predictions)

print("\n--- Optimized Model Results ---")
print(f"Best Polynomial Degree: {best_degree}")
print(f"Best Alpha: {best_alpha:.6f}")
print(f"\nR2 Score on Training Set: {r2_train:.4f}")
print(f"R2 Score on Test Set: {r2_test:.4f}")
print(f"MAE on Training Set: {mae_train:.2f}")
print(f"MAE on Test Set: {mae_test:.2f}")
print(f"RMSE on Training Set: {rmse_train:.2f}")
print(f"RMSE on Test Set: {rmse_test:.2f}")

# --- 5. Model 3 Lasso Regression with polynomial feature ---

from sklearn.preprocessing import PolynomialFeatures

polynomial_lasso_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('polynomialfeatures', PolynomialFeatures()), # Placeholder, degree will be tuned
    ('regressor', Lasso())
])

param_grid = {
    # Alpha values for Lasso regularization
    'regressor__alpha': np.logspace(-6, 3, 10),
    # Polynomial degrees to test (1 to 10)
    'polynomialfeatures__degree': np.arange(1, 11),
    # iteration
    'regressor__max_iter': [100]
}

print("--- Lasso Regression with Polynomial Features Tuning ---")
grid_search = GridSearchCV(
    polynomial_lasso_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2'
)

grid_search.fit(X_train, y_train)

Lasso_best_estimator = grid_search.best_estimator_
Lasso_predictions = Lasso_best_estimator.predict(X_test)

best_params = grid_search.best_params_
best_alpha = best_params['regressor__alpha']
best_degree = best_params['polynomialfeatures__degree']

# Evaluation
r2_train = r2_score(y_train, Lasso_best_estimator.predict(X_train))
r2_test = r2_score(y_test, Lasso_predictions)
mae_train = mean_absolute_error(y_train, Lasso_best_estimator.predict(X_train))
mae_test = mean_absolute_error(y_test, Lasso_predictions)
rmse_train = root_mean_squared_error(y_train, Lasso_best_estimator.predict(X_train))
rmse_test = root_mean_squared_error(y_test, Lasso_predictions)

print("\n--- Optimized Model Results ---")
print(f"Best Polynomial Degree: {best_degree}")
print(f"Best Alpha: {best_alpha:.6f}")
print(f"\nR2 Score on Training Set: {r2_train:.4f}")
print(f"R2 Score on Test Set: {r2_test:.4f}")
print(f"MAE on Training Set: {mae_train:.2f}")
print(f"MAE on Test Set: {mae_test:.2f}")
print(f"RMSE on Training Set: {rmse_train:.2f}")
print(f"RMSE on Test Set: {rmse_test:.2f}")

# --- 6. Model 4 Elastuc Net Regression with polynomial feature ---

from sklearn.preprocessing import PolynomialFeatures

polynomial_elastic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('polynomialfeatures', PolynomialFeatures()), # Placeholder, degree will be tuned
    ('regressor', ElasticNet())
])

param_grid = {
    # Alpha values for Elastic Net
    'regressor__alpha': np.logspace(-6, 3, 10),
    'regressor__l1_ratio': np.linspace(0, 1, 10),
    # Polynomial degrees to test (1 to 10)
    'polynomialfeatures__degree': np.arange(1, 11),
    # iteration
    'regressor__max_iter': [100]
}

print("--- Elastic Net Regression with Polynomial Features Tuning ---")
grid_search = GridSearchCV(
    polynomial_elastic_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='r2'
)

grid_search.fit(X_train, y_train)

Elastic_best_estimator = grid_search.best_estimator_
Elastic_predictions = Elastic_best_estimator.predict(X_test)

best_params = grid_search.best_params_
best_alpha = best_params['regressor__alpha']
best_l1_ratio = best_params['regressor__l1_ratio']
best_degree = best_params['polynomialfeatures__degree']

# evaluation
r2_train = r2_score(y_train, Elastic_best_estimator.predict(X_train))
r2_test = r2_score(y_test, Elastic_predictions)
mae_train = mean_absolute_error(y_train, Elastic_best_estimator.predict(X_train))
mae_test = mean_absolute_error(y_test, Elastic_predictions)
rmse_train = root_mean_squared_error(y_train, Elastic_best_estimator.predict(X_train))
rmse_test = root_mean_squared_error(y_test, Elastic_predictions)

print("\n--- Optimized Model Results ---")
print(f"Best Polynomial Degree: {best_degree}")
print(f"Best Alpha: {best_alpha:.6f}")
print(f"Best L1 Ratio: {best_l1_ratio:.2f}")
print(f"\nR2 Score on Training Set: {r2_train:.4f}")
print(f"R2 Score on Test Set: {r2_test:.4f}")
print(f"MAE on Training Set: {mae_train:.2f}")
print(f"MAE on Test Set: {mae_test:.2f}")
print(f"RMSE on Training Set: {rmse_train:.2f}")
print(f"RMSE on Test Set: {rmse_test:.2f}")

results = pd.DataFrame(grid_search.cv_results_)

heatmap_data = results.pivot_table(
    values='mean_test_score',
    index='param_polynomialfeatures__degree',
    columns='param_regressor__alpha'
)

heatmap_data = heatmap_data.sort_index(axis=1)

plt.figure(figsize=(10, 8))

alpha_labels = [f'{a:.0e}' for a in heatmap_data.columns]

sns.heatmap(
    heatmap_data,
    annot=True,          # Show R2 values on the heatmap
    annot_kws={'size': 7},
    fmt=".3f",           # Format R2 values to 3 decimal places
    cmap="viridis",      # Color map: 'viridis' is a good sequential choice
    cbar_kws={'label': 'Cross-Validated R2 Score'}
)

plt.xlabel(r'Lasso $\alpha$ (Log Scale)', fontsize=12)
plt.ylabel('Polynomial Degree', fontsize=12)
plt.title(r'R2 Score Heatmap: Tuning $\alpha$ and Polynomial Degree', fontsize=14)

# Apply the log scale labels to the actual tick marks
plt.xticks(ticks=np.arange(len(alpha_labels)), labels=alpha_labels, rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
```

---

## 🏁 สรุปและปิดท้าย (Conclusion and Closing Session)

จากการเปรียบเทียบโมเดลทั้งหมด พบว่า Lasso Regression เป็นโมเดลที่ให้ผลการทำนายปริมาณการขายดีที่สุด โดยให้ค่า R² สูงที่สุด และมีค่า MAE และ RMSE ต่ำที่สุด อย่างไรก็ตาม ค่า R² ที่ได้ยังคงอยู่ในระดับต่ำ เพียง 0.5639 ซึ่งสะท้อนว่ายังมีความแปรปรวนของข้อมูลจำนวนมากที่โมเดลไม่สามารถอธิบายได้

![](https://github.com/ThitiwutM/Restaurant-data-Quantity-sold-prediction_DS514-515/blob/main/model%20results.jpg)

ข้อเสนอแนะสำหรับการพัฒนาโมเดลเพิ่มเติม ได้แก่:
- ใช้ numerical features แทน categorical เพื่อให้โมเดลจับความสัมพันธ์ได้ดีขึ้น เช่น อุณหภูมิ ปริมาณฝน หรือเปอร์เซ็นต์ส่วนลดค่าอาหาร
- ใช้การแบ่งข้อมูลแบบ Time-based split แทนการสุ่ม (random split) เนื่องจากข้อมูลอยู่ในรูปแบบอนุกรมเวลา (time-series)
- ทดลองโมเดลอื่นเพิ่มเติม เช่น kNN, SVM หรือโมเดลเชิงไม่เชิงเส้นอื่น ๆ ที่อาจเหมาะกับข้อมูลลักษณะนี้
- วิเคราะห์ความสำคัญของปัจจัยที่มีผลต่อยอดขาย โดยใช้ SHAP เพื่อช่วยตีความโมเดล และระบุว่าตัวแปรใดมีผลมากที่สุดต่อการทำนาย

หากต้องการ code python แบบเต็ม ๆ สามารถดาวน์โหลดได้จาก link นี้ → https://github.com/ThitiwutM/Restaurant-data-Quantity-sold-prediction_DS514-515/blob/main/DS514-515%20Quantity_sold%20prediction.ipynb

---

ขอบคุณทุกคนที่อ่านมาจนถึงตอนสุดท้าย!!! เรายังมีอีกหนึ่งโปรเจคที่ทำคู่ขนานกัน นั่นคือ...

📊 โปรเจคของวิชา DS512/513: การวิเคราะห์ปัจจัยที่ส่งผลต่อยอดขายของร้านอาหาร

โครงการนี้มีวัตถุประสงค์เพื่อ ศึกษาข้อมูลยอดขายของร้านอาหารและระบุปัจจัยสำคัญที่ส่งผลต่อปริมาณการขาย โดยใช้สถิติเชิงพรรณนา (Excel), การวิเคราะห์สหสัมพันธ์ (Excel) และ แดชบอร์ดเชิงโต้ตอบ (Tableau) โปรเจคนี้มุ่งเน้นการสนับสนุน เจ้าของร้านอาหารและทีมการตลาด ให้สามารถวางแผน โปรโมชัน, กิจกรรมพิเศษ, และ การบริหารจัดการร้าน ให้เหมาะสมกับสภาพอากาศและพฤติกรรมผู้บริโภค สุดท้ายนี้ เราคาดหวังให้ร้านสามารถ เพิ่มยอดขาย ผ่านกลยุทธ์ที่ถูกออกแบบอย่างมีข้อมูลรองรับ

🔍 ขั้นตอนการทำงาน (Project Workflow)

กระบวนการวิเคราะห์ประกอบด้วย:

✔️ รายละเอียดเกี่ยวกับชุดข้อมูล (About dataset)
✔️ การทำความสะอาดข้อมูล (Data Cleaning)
✔️ การวิเคราะห์ข้อมูลเชิงสำรวจ (Exploratory Data Analysis, EDA)
✔️ การสร้างภาพข้อมูลและแดชบอร์ด (Data Visualization & Dashboard)

📌 โปรเจคฉบับเต็ม

หากต้องการดูรายละเอียด สามารถเข้าชมได้ที่ลิงก์ด้านล่างนี้ 👇

🔗 GitHub Repository: https://github.com/ThitiwutM/Restaurant-data-insights-and-analysis_DS512-513
