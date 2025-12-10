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

✔️ Data pipeline development

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

✔️ การสร้าง Data pipeline

✔️ การสร้างและประเมินโมเดล (Model building & evaluation)

---

## 1️⃣ รายละเอียดเกี่ยวกับชุดข้อมูล (About dataset)

ข้อมูลชุดนี้นำมาจาก Kaggle โพสต์โดย คุณ Alexand Chen (https://www.kaggle.com/datasets/alexandchen/restaurant-sales-report-2024-2025)

Dataset ในรูปแบบ csv ไฟล์ สามารถดาวน์โหลดได้จาก restaurant_dataset.csv (https://github.com/ThitiwutM/Restaurant-data-insights-and-analysis_DS512-513/blob/2574a13c6e83485b0acc07a17904733ea60ee265/restaurant_dataset.csv)

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


