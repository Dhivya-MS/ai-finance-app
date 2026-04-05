import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Finance App", layout="wide")

st.title("💰 AI-Powered Smart Finance Analytics System")

data = {
    "Date": [
        "2026-01-01","2026-01-02","2026-01-03","2026-01-05",
        "2026-01-07","2026-01-10","2026-01-12","2026-01-15",
        "2026-01-18","2026-01-20"
    ],
    "Category": [
        "Food","Travel","Shopping","Bills",
        "Food","Entertainment","Travel","Food",
        "Shopping","Bills"
    ],
    "Amount": [200,500,1500,800,300,600,700,250,1200,900]
}

df = pd.DataFrame(data)

st.subheader("📊 Dataset")
st.write(df)

st.subheader("📊 Category Spending")
category = df.groupby("Category")["Amount"].sum()

fig, ax = plt.subplots()
ax.bar(category.index, category.values)
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("🧠 Spending Clusters")
X_cluster = df[['Amount']]
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(X_cluster)

fig2, ax2 = plt.subplots()
ax2.scatter(df['Amount'], df['Cluster'])
st.pyplot(fig2)

st.subheader("🔮 Prediction")
df['Day'] = np.arange(len(df))
X = df[['Day']]
y = df['Amount']

model = LinearRegression()
model.fit(X, y)

prediction = model.predict([[len(df)]])
st.success(f"Predicted Next Expense: {prediction[0]:.2f}")

st.subheader("🤖 AI Advice")

total = df['Amount'].sum()
avg = df['Amount'].mean()
top_category = df.groupby('Category')['Amount'].sum().idxmax()

st.write(f"Total Spending: {total}")
st.write(f"Average Spending: {avg}")
st.write(f"Top Category: {top_category}")

if top_category == "Shopping":
    st.warning("Reduce shopping expenses 🛍️")
elif top_category == "Food":
    st.info("Control food spending 🍔")
elif top_category == "Travel":
    st.info("Plan travel wisely ✈️")

if avg > 700:
    st.error("High spending ⚠️")
else:
    st.success("Spending under control ✅")