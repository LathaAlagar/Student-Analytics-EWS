import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings("ignore")


def load_data(students_path="data/students.csv",
              assessment_path="data/assessment.csv",
              engagement_path="data/engagement.csv"):
    students = pd.read_csv(students_path)
    assessment = pd.read_csv(assessment_path)
    engagement = pd.read_csv(engagement_path)
    return students, assessment, engagement


def compute_features(students, assessment, engagement):
    # Aggregate assessment scores per student
    avg_scores = (
        assessment.groupby("student_id")["avg_score"]
        .mean()
        .reset_index()
        .rename(columns={"avg_score": "mean_score"})
    )
    avg_scores["mean_score"] = avg_scores["mean_score"].round(1)

    # Merge all data
    df = students.merge(avg_scores, on="student_id", how="left")
    df = df.merge(engagement, on="student_id", how="left")

    # Feature engineering
    df["engagement_score"] = (
        df["videos_watched"] * 1.5 +
        df["assignments_submitted"] * 3 +
        df["forum_posts"] * 2 +
        df["quizzes_attempted"] * 2.5 +
        df["login_days"] * 0.5
    )
    max_eng = df["engagement_score"].max()
    df["engagement_score"] = (df["engagement_score"] / max_eng * 100).round(1)

    # Learning profile
    df["learning_profile"] = pd.cut(
        df["engagement_score"],
        bins=[0, 33, 66, 100],
        labels=["Passive Learner", "Moderate Learner", "Active Learner"],
        include_lowest=True
    )

    # Risk score (0-100, higher = more at risk)
    score_risk = np.where(df["mean_score"] < 50, 40, np.where(df["mean_score"] < 65, 20, 0))
    eng_risk = np.where(df["engagement_score"] < 40, 40, np.where(df["engagement_score"] < 60, 20, 0))
    att_risk = np.where(df["attendance_pct"] < 60, 20, np.where(df["attendance_pct"] < 75, 10, 0))
    df["risk_score"] = score_risk + eng_risk + att_risk

    # Ground truth label for ML training
    df["risk_label"] = pd.cut(
        df["risk_score"],
        bins=[-1, 25, 55, 100],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    return df


def train_models(df):
    features = ["mean_score", "engagement_score", "attendance_pct",
                 "assignments_submitted", "login_days", "quizzes_attempted"]

    X = df[features].fillna(0)
    y = df["risk_label"].astype(str)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = round(accuracy_score(y_test, rf_pred) * 100, 1)

    # Logistic Regression
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = round(accuracy_score(y_test, lr_pred) * 100, 1)

    # K-Means clustering (3 clusters)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    # Final RF predictions on all students
    df["ml_risk"] = rf.predict(X_scaled)
    df["ml_risk_proba"] = rf.predict_proba(X_scaled).max(axis=1).round(2)

    model_info = {
        "rf_accuracy": rf_acc,
        "lr_accuracy": lr_acc,
        "feature_importances": dict(zip(features, rf.feature_importances_.round(3))),
        "features": features
    }

    return df, model_info


def get_student_alerts(row):
    alerts = []
    if row["mean_score"] < 50:
        alerts.append("⚠️ Assessment score is critically low (below 50%)")
    elif row["mean_score"] < 65:
        alerts.append("📉 Assessment score is below class average")
    if row["engagement_score"] < 40:
        alerts.append("⚠️ Learning engagement is very low")
    elif row["engagement_score"] < 60:
        alerts.append("📉 Learning engagement needs improvement")
    if row["attendance_pct"] < 60:
        alerts.append("⚠️ Attendance is critically low (below 60%)")
    elif row["attendance_pct"] < 75:
        alerts.append("📉 Attendance is below the required 75%")
    if row["assignments_submitted"] < 5:
        alerts.append("📝 Very few assignments submitted")
    return alerts


def get_recommendations(row):
    recs = []
    if row["mean_score"] < 65:
        recs.append("📚 Attend remedial/extra coaching classes for weak subjects")
        recs.append("🔁 Revise past test papers and practice regularly")
    if row["engagement_score"] < 60:
        recs.append("💻 Increase daily login and watch pending video lectures")
        recs.append("🙋 Participate in forum discussions and group studies")
    if row["attendance_pct"] < 75:
        recs.append("🏫 Improve class attendance — aim for 90%+")
    if row["assignments_submitted"] < 10:
        recs.append("📋 Submit all pending assignments on time")
    if row["quizzes_attempted"] < 5:
        recs.append("🎯 Attempt more practice quizzes to self-assess")
    if not recs:
        recs.append("✅ Keep up the excellent work! Maintain your current performance.")
        recs.append("🏆 Consider mentoring peers who need help")
    return recs
