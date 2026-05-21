# 🎓 EduAnalytics AI
## AI-Driven Learning Analytics & Early Warning System

A full-stack Streamlit application with role-based access for Admin, Faculty, and Students.
Uses Machine Learning (Random Forest, Logistic Regression, K-Means) to detect at-risk students.

---

## 📁 Project Structure

```
eduanalytics/
├── app.py                  ← Main Streamlit app (run this)
├── requirements.txt        ← Python dependencies
├── data/
│   ├── students.csv        ← Student profiles (auto-generated)
│   ├── assessment.csv      ← Test marks per subject
│   └── engagement.csv      ← Learning activity logs
└── utils/
    └── data_processor.py   ← ML pipeline, feature engineering, alerts
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app auto-generates 50 student records on first run.

---

## 🔑 Demo Login Credentials

| Role    | ID           | Password |
|---------|--------------|----------|
| Admin   | ADMIN001     | admin123 |
| Faculty | FAC001       | fac123   |
| Faculty | FAC002       | fac123   |
| Student | STU001–STU050| stu123   |

---

## 🧩 Modules

| Module | Description |
|--------|-------------|
| Admin  | System overview, student/faculty management, dataset upload, ML model stats, reports |
| Faculty | Dashboard, all-student table, at-risk detection, analytics (heatmap, scatter, histogram), alert system |
| Student | Personal dashboard, subject-wise scores, alerts, study recommendations |

---

## 🤖 ML Pipeline

- **Random Forest** — Primary risk classifier
- **Logistic Regression** — Secondary classifier for comparison
- **K-Means (k=3)** — Student segmentation/clustering
- **Features used**: avg_score, engagement_score, attendance_pct, assignments_submitted, login_days, quizzes_attempted
- **Risk Labels**: HIGH / MEDIUM / LOW

---

## 📊 Charts & Visualizations (Plotly)

- Risk distribution (donut chart)
- Score distribution (histogram)
- Score vs Engagement (scatter plot with risk color)
- Feature importance (horizontal bar)
- K-Means cluster plot
- Correlation heatmap
- Subject-wise performance (bar)
- Score trend across tests (line)
- Attendance distribution
- Department-wise risk (stacked bar)
