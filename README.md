## Customer Segmentation in e-commerce

##  Project Overview
This project performs Customer Segmentation using the RFM (Recency, Frequency, Monetary) model and K-Means Clustering algorithm.  
The goal is to group customers based on their purchasing behavior.

## 🛠 Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn

### Data Preprocessing
- Converted InvoiceDate to datetime format
- Created RFM features:
  - Recency → Days since last purchase
  - Frequency → Number of purchases
  - Monetary → Total spending amount

### Feature Scaling
- Applied StandardScaler to normalize data

### 3️ Clustering
- Used K-Means Algorithm
- Number of clusters: 4
- Random state: 42

###  Model Evaluation
- Silhouette Score used to evaluate clustering quality

## 📈 Results
- Customers segmented into 4 clusters
- Silhouette Score: 0.65
- Clear separation between high-value and low-value customers

## 📷 Output Visualization
![Customer Segmentation Output](output.png)

##  Dataset
The dataset contains:
- CustomerID
- InvoiceDate
- Amount

## How to Run
pip install pandas matplotlib scikit-learn
python segmentation.py

