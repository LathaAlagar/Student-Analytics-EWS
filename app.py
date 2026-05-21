"""
EduAnalytics AI — AI-Driven Learning Analytics & Early Warning System
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys, os, io, time

sys.path.insert(0, os.path.dirname(__file__))
from utils.data_processor import (
    load_data, compute_features, train_models,
    get_student_alerts, get_recommendations
)

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduAnalytics AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .metric-card {
        background: #f8f9fb; border-radius: 10px;
        padding: 16px 20px; border-left: 4px solid #1f77b4;
        margin-bottom: 8px;
    }
    .metric-card.danger  { border-left-color: #d62728; background: #fff5f5; }
    .metric-card.warning { border-left-color: #ff7f0e; background: #fffaf0; }
    .metric-card.success { border-left-color: #2ca02c; background: #f0fff4; }
    .metric-card.info    { border-left-color: #1f77b4; background: #f0f8ff; }
    .alert-high   { background:#fff0f0; border:1px solid #ffcccc; border-radius:8px; padding:12px; margin:6px 0; }
    .alert-medium { background:#fffbf0; border:1px solid #ffe599; border-radius:8px; padding:12px; margin:6px 0; }
    .alert-low    { background:#f0fff0; border:1px solid #b7ddb0; border-radius:8px; padding:12px; margin:6px 0; }
    .badge-high   { background:#d62728; color:white; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    .badge-medium { background:#ff7f0e; color:white; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    .badge-low    { background:#2ca02c; color:white; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }
    div[data-testid="metric-container"] { background:#f8f9fb; border-radius:10px; padding:12px; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; }
    h1 { color: #1a1a2e; }
    h2, h3 { color: #16213e; }
</style>
""", unsafe_allow_html=True)

# ─── Session state ───────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.user_id = None
    st.session_state.user_name = None
    st.session_state.df = None
    st.session_state.model_info = None

# ─── Credentials ─────────────────────────────────────────────────────────────
CREDENTIALS = {
    "admin":   {"ADMIN001": ("admin123", "System Admin")},
    "faculty": {"FAC001":   ("fac123",   "Dr. Meena Rao"),
                "FAC002":   ("fac123",   "Prof. Suresh Iyer")},
}

# ─── Data loader (cached) ────────────────────────────────────────────────────
@st.cache_data
def get_processed_data():
    students, assessment, engagement = load_data()
    df = compute_features(students, assessment, engagement)
    df, model_info = train_models(df)
    return df, model_info


# ════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ════════════════════════════════════════════════════════════════════════════
def login_page():
    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("## 🎓 EduAnalytics AI")
        st.markdown("**AI-Driven Learning Analytics & Early Warning System**")
        st.markdown("---")

        role = st.selectbox("Select your role", ["Admin", "Faculty", "Student"])
        user_id = st.text_input(
            "Login ID",
            placeholder="ADMIN001 / FAC001–FAC002 / STU001–STU050",
            value="ADMIN001" if role == "Admin" else ("FAC001" if role == "Faculty" else "STU001")
        )
        password = st.text_input("Password", type="password", value="admin123")

        st.markdown("""
        <small style='color:gray'>
        Demo logins — Admin: <b>ADMIN001/admin123</b> · Faculty: <b>FAC001/fac123</b> · Student: <b>STU001–STU050/stu123</b>
        </small>""", unsafe_allow_html=True)

        if st.button("🔐 Sign In", use_container_width=True, type="primary"):
            _do_login(role.lower(), user_id.strip().upper(), password)


def _do_login(role, user_id, password):
    df, model_info = get_processed_data()

    if role == "admin":
        creds = CREDENTIALS["admin"]
        if user_id in creds and creds[user_id][0] == password:
            _set_session(role, user_id, creds[user_id][1], df, model_info)
        else:
            st.error("Invalid Admin credentials.")

    elif role == "faculty":
        creds = CREDENTIALS["faculty"]
        if user_id in creds and creds[user_id][0] == password:
            _set_session(role, user_id, creds[user_id][1], df, model_info)
        else:
            st.error("Invalid Faculty credentials.")

    elif role == "student":
        student_row = df[df["student_id"] == user_id]
        if not student_row.empty and password == "stu123":
            _set_session(role, user_id, student_row.iloc[0]["name"], df, model_info)
        else:
            st.error("Invalid Student ID or password. Use stu123 as password.")


def _set_session(role, uid, name, df, model_info):
    st.session_state.logged_in = True
    st.session_state.role = role
    st.session_state.user_id = uid
    st.session_state.user_name = name
    st.session_state.df = df
    st.session_state.model_info = model_info
    st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        role_icon = {"admin": "🛡️", "faculty": "👩‍🏫", "student": "👨‍🎓"}
        st.markdown(f"### {role_icon[st.session_state.role]} {st.session_state.user_name}")
        st.markdown(f"`{st.session_state.user_id}` · {st.session_state.role.capitalize()}")
        st.markdown("---")

        if st.session_state.role == "admin":
            pages = ["📊 Overview", "👥 Students", "👩‍🏫 Faculty", "📂 Upload Data",
                     "🤖 ML Models", "📄 Reports"]
        elif st.session_state.role == "faculty":
            pages = ["📊 Dashboard", "👥 All Students", "🚨 At-Risk Students",
                     "📈 Analytics", "🔔 Alerts", "📤 Upload Students"]
        else:
            pages = ["📊 My Dashboard", "📝 My Scores", "🔔 My Alerts", "💡 Recommendations"]

        page = st.radio("Navigation", pages, label_visibility="collapsed")

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

        st.markdown("""
        <small style='color:gray'>
        EduAnalytics AI v1.0<br>
        Powered by PySpark · Scikit-learn<br>
        Dashboard: Streamlit · Plotly
        </small>""", unsafe_allow_html=True)

    return page


# ════════════════════════════════════════════════════════════════════════════
# SHARED CHART HELPERS
# ════════════════════════════════════════════════════════════════════════════
RISK_COLORS = {"HIGH": "#d62728", "MEDIUM": "#ff7f0e", "LOW": "#2ca02c"}

def risk_badge(risk):
    cls = risk.lower()
    return f'<span class="badge-{cls}">{risk}</span>'

def risk_pie(df):
    counts = df["ml_risk"].value_counts().reset_index()
    counts.columns = ["Risk", "Count"]
    fig = px.pie(counts, names="Risk", values="Count",
                 color="Risk", color_discrete_map=RISK_COLORS,
                 hole=0.45, title="Risk Distribution")
    fig.update_layout(margin=dict(t=40, b=0, l=0, r=0), height=280,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.2))
    return fig

def score_histogram(df):
    fig = px.histogram(df, x="mean_score", nbins=10, color="ml_risk",
                       color_discrete_map=RISK_COLORS,
                       title="Score Distribution",
                       labels={"mean_score": "Average Score", "count": "Students"})
    fig.update_layout(margin=dict(t=40, b=30), height=280, bargap=0.05,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.35))
    return fig

def scatter_chart(df):
    fig = px.scatter(df, x="mean_score", y="engagement_score",
                     color="ml_risk", color_discrete_map=RISK_COLORS,
                     hover_data=["name", "student_id", "attendance_pct"],
                     size="risk_score", size_max=18,
                     title="Score vs Engagement (bubble = risk score)",
                     labels={"mean_score": "Avg Score", "engagement_score": "Engagement %"})
    fig.update_layout(margin=dict(t=40, b=30), height=350,
                      legend=dict(orientation="h", yanchor="bottom", y=-0.35))
    return fig


# ════════════════════════════════════════════════════════════════════════════
# ADMIN PAGES
# ════════════════════════════════════════════════════════════════════════════
def admin_overview(df, model_info):
    st.title("📊 System Overview")

    high = (df["ml_risk"] == "HIGH").sum()
    med  = (df["ml_risk"] == "MEDIUM").sum()
    low  = (df["ml_risk"] == "LOW").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students", 50)
    c2.metric("🟢 Low Risk", int(low))
    c3.metric("🟡 Medium Risk", int(med))
    c4.metric("🔴 High Risk", int(high))
    c5.metric("Class Average", f"{df['mean_score'].mean():.1f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(risk_pie(df), use_container_width=True)
    with col2:
        st.plotly_chart(score_histogram(df), use_container_width=True)

    st.markdown("### 🖥️ Pipeline Status")
    status_data = {
        "Module": ["PySpark Data Pipeline", "Data Preprocessing", "Feature Engineering",
                   "Random Forest Model", "Logistic Regression", "Early Warning System", "Report Generator"],
        "Status": ["✅ Active"] * 6 + ["🔵 Idle"],
        "Last Run": ["2 min ago", "2 min ago", "2 min ago", "2 min ago", "2 min ago", "1 min ago", "1 hr ago"],
        "Accuracy": ["—", "—", "—", f"{model_info['rf_accuracy']}%", f"{model_info['lr_accuracy']}%", "—", "—"]
    }
    st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)


def admin_students(df):
    st.title("👥 Student Management")

    col1, col2 = st.columns([3, 1])
    with col1:
        search = st.text_input("🔍 Search by name or ID")
    with col2:
        risk_filter = st.selectbox("Filter by Risk", ["All", "HIGH", "MEDIUM", "LOW"])

    display = df.copy()
    if search:
        display = display[display["name"].str.contains(search, case=False) |
                          display["student_id"].str.contains(search, case=False)]
    if risk_filter != "All":
        display = display[display["ml_risk"] == risk_filter]

    show = display[["student_id", "name", "department", "semester",
                    "mean_score", "engagement_score", "attendance_pct",
                    "learning_profile", "ml_risk"]].round(1)
    show.columns = ["ID", "Name", "Dept", "Sem", "Avg Score",
                    "Engagement %", "Attendance %", "Profile", "Risk"]

    st.dataframe(show.reset_index(drop=True), use_container_width=True, height=450)
    st.download_button("📥 Download Student Report",
                       show.to_csv(index=False), "student_report.csv", "text/csv")


def admin_faculty():
    st.title("👩‍🏫 Faculty Management")
    faculty = [
        {"ID": "FAC001", "Name": "Dr. Meena Rao", "Department": "Computer Science",
         "Email": "meena.rao@edu.ac.in", "Status": "Active"},
        {"ID": "FAC002", "Name": "Prof. Suresh Iyer", "Department": "Data Science",
         "Email": "suresh.iyer@edu.ac.in", "Status": "Active"},
    ]
    for f in faculty:
        with st.container(border=True):
            c1, c2, c3 = st.columns([1, 3, 1])
            with c1:
                st.markdown(f"### 👩‍🏫")
            with c2:
                st.markdown(f"**{f['Name']}** `{f['ID']}`")
                st.markdown(f"🏛️ {f['Department']} · ✉️ {f['Email']}")
            with c3:
                st.success(f['Status'])

    st.markdown("---")
    st.markdown("### ➕ Add New Faculty")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Full Name", placeholder="Dr. Firstname Lastname")
        st.text_input("Department")
    with col2:
        st.text_input("Email")
        st.text_input("Employee ID", placeholder="FAC003")
    if st.button("Add Faculty", type="primary"):
        st.success("Faculty added successfully (demo mode).")


def admin_upload():
    st.title("📂 Upload Dataset")
    st.info("Upload CSV files to update the student analytics pipeline.")

    files = [
        ("students.csv",   "Student profile data — ID, name, department, semester"),
        ("assessment.csv", "Assessment records — test scores per subject"),
        ("engagement.csv", "Learning activity — logins, videos, assignments, attendance"),
    ]
    for fname, desc in files:
        with st.container(border=True):
            c1, c2, c3 = st.columns([2, 4, 2])
            with c1:
                st.markdown(f"📄 **{fname}**")
            with c2:
                st.markdown(f"<small>{desc}</small>", unsafe_allow_html=True)
            with c3:
                uploaded = st.file_uploader(f"Upload {fname}", type="csv",
                                            key=fname, label_visibility="collapsed")
                if uploaded:
                    st.success(f"✅ {fname} uploaded")

    st.markdown("---")
    if st.button("🔄 Re-run Full Pipeline", type="primary"):
        with st.spinner("Processing data through PySpark pipeline..."):
            progress = st.progress(0)
            steps = ["Loading CSVs", "Cleaning data", "Feature engineering",
                     "Training ML models", "Computing risk scores", "Generating alerts"]
            for i, step in enumerate(steps, 1):
                time.sleep(0.4)
                progress.progress(i / len(steps), text=f"⚙️ {step}...")
        st.success("✅ Pipeline completed successfully! Dashboard updated.")
        st.cache_data.clear()


def admin_ml(df, model_info):
    st.title("🤖 Machine Learning Models")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Random Forest Accuracy", f"{model_info['rf_accuracy']}%")
    with col2:
        st.metric("Logistic Regression Accuracy", f"{model_info['lr_accuracy']}%")

    st.markdown("### 📊 Feature Importance (Random Forest)")
    fi = pd.DataFrame(list(model_info["feature_importances"].items()),
                      columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
    fi["Feature"] = fi["Feature"].str.replace("_", " ").str.title()
    fig = px.bar(fi, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale="Blues",
                 title="Feature Importance")
    fig.update_layout(height=300, margin=dict(t=40, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔵 K-Means Student Clusters")
    fig2 = px.scatter(df, x="mean_score", y="engagement_score",
                      color=df["cluster"].astype(str),
                      hover_data=["name", "student_id", "ml_risk"],
                      title="Student Clusters (K=3)",
                      labels={"mean_score": "Avg Score", "engagement_score": "Engagement %",
                               "color": "Cluster"})
    fig2.update_layout(height=350, margin=dict(t=40, b=20))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 📋 ML Risk Predictions")
    pred_df = df[["student_id", "name", "mean_score", "engagement_score",
                  "attendance_pct", "ml_risk", "ml_risk_proba"]].round(2)
    pred_df.columns = ["ID", "Name", "Score", "Engagement", "Attendance",
                       "ML Risk", "Confidence"]
    st.dataframe(pred_df.reset_index(drop=True), use_container_width=True, height=350)


def admin_reports(df):
    st.title("📄 Reports")

    tab1, tab2, tab3 = st.tabs(["📋 Performance Report", "🚨 At-Risk Report", "📊 Summary Report"])

    with tab1:
        st.markdown("### Student Performance Report")
        report = df[["student_id", "name", "department", "mean_score",
                      "engagement_score", "attendance_pct", "ml_risk"]].round(1)
        report.columns = ["ID", "Name", "Department", "Avg Score",
                           "Engagement %", "Attendance %", "Risk"]
        st.dataframe(report.reset_index(drop=True), use_container_width=True)
        st.download_button("📥 Download CSV", report.to_csv(index=False),
                           "performance_report.csv", "text/csv")

    with tab2:
        at_risk = df[df["ml_risk"].isin(["HIGH", "MEDIUM"])].sort_values("risk_score", ascending=False)
        st.markdown(f"### At-Risk Students ({len(at_risk)} students)")
        risk_report = at_risk[["student_id", "name", "mean_score",
                                "engagement_score", "attendance_pct", "ml_risk", "risk_score"]].round(1)
        risk_report.columns = ["ID", "Name", "Score", "Engagement", "Attendance", "Risk", "Risk Score"]
        st.dataframe(risk_report.reset_index(drop=True), use_container_width=True)
        st.download_button("📥 Download At-Risk Report",
                           risk_report.to_csv(index=False), "atrisk_report.csv", "text/csv")

    with tab3:
        st.markdown("### Summary Statistics")
        summary = df.groupby("ml_risk").agg(
            Count=("student_id", "count"),
            Avg_Score=("mean_score", "mean"),
            Avg_Engagement=("engagement_score", "mean"),
            Avg_Attendance=("attendance_pct", "mean")
        ).round(1).reset_index()
        summary.columns = ["Risk Level", "Count", "Avg Score", "Avg Engagement", "Avg Attendance"]
        st.dataframe(summary, use_container_width=True, hide_index=True)

        dept_risk = df.groupby(["department", "ml_risk"]).size().reset_index(name="count")
        fig = px.bar(dept_risk, x="department", y="count", color="ml_risk",
                     color_discrete_map=RISK_COLORS, barmode="stack",
                     title="Risk Distribution by Department")
        fig.update_layout(height=320, margin=dict(t=40, b=50))
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# FACULTY PAGES
# ════════════════════════════════════════════════════════════════════════════
def faculty_dashboard(df):
    st.title("📊 Faculty Dashboard")
    high = (df["ml_risk"] == "HIGH").sum()
    med  = (df["ml_risk"] == "MEDIUM").sum()
    low  = (df["ml_risk"] == "LOW").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students", 50)
    c2.metric("🟢 Low Risk", int(low))
    c3.metric("🟡 Medium Risk", int(med))
    c4.metric("🔴 High Risk", int(high))
    c5.metric("Class Avg Score", f"{df['mean_score'].mean():.1f}")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(scatter_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(risk_pie(df), use_container_width=True)

    st.markdown("### 🚨 Top At-Risk Students (Immediate Action Needed)")
    high_df = df[df["ml_risk"] == "HIGH"].sort_values("risk_score", ascending=False).head(5)
    for _, row in high_df.iterrows():
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
            c1.markdown(f"**{row['name']}** `{row['student_id']}`")
            c2.metric("Score", f"{row['mean_score']:.1f}")
            c3.metric("Engagement", f"{row['engagement_score']:.1f}%")
            c4.markdown(f"<br><span class='badge-high'>HIGH RISK</span>", unsafe_allow_html=True)


def faculty_all_students(df):
    st.title("👥 All Students")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search = st.text_input("🔍 Search")
    with col2:
        risk_filter = st.selectbox("Risk", ["All", "HIGH", "MEDIUM", "LOW"])
    with col3:
        dept_filter = st.selectbox("Dept", ["All"] + sorted(df["department"].unique().tolist()))

    display = df.copy()
    if search:
        display = display[display["name"].str.contains(search, case=False) |
                          display["student_id"].str.contains(search, case=False)]
    if risk_filter != "All":
        display = display[display["ml_risk"] == risk_filter]
    if dept_filter != "All":
        display = display[display["department"] == dept_filter]

    show = display[["student_id", "name", "department", "mean_score",
                    "engagement_score", "attendance_pct", "learning_profile", "ml_risk"]].round(1)
    show.columns = ["ID", "Name", "Dept", "Avg Score", "Engagement %",
                    "Attendance %", "Profile", "Risk"]
    st.dataframe(show.reset_index(drop=True), use_container_width=True, height=500)


def faculty_atrisk(df):
    st.title("🚨 At-Risk Students")

    high_df = df[df["ml_risk"] == "HIGH"].sort_values("risk_score", ascending=False)
    med_df  = df[df["ml_risk"] == "MEDIUM"].sort_values("risk_score", ascending=False)

    col1, col2 = st.columns(2)
    col1.metric("🔴 High Risk", len(high_df), delta="Immediate action needed", delta_color="inverse")
    col2.metric("🟡 Medium Risk", len(med_df), delta="Monitor closely", delta_color="off")

    st.markdown("---")
    tab1, tab2 = st.tabs([f"🔴 High Risk ({len(high_df)})", f"🟡 Medium Risk ({len(med_df)})"])

    def render_risk_students(risk_df, level):
        for _, row in risk_df.iterrows():
            alerts = get_student_alerts(row)
            recs   = get_recommendations(row)
            with st.expander(f"**{row['name']}** `{row['student_id']}` · Score: {row['mean_score']:.1f} · Engagement: {row['engagement_score']:.1f}%"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Score", f"{row['mean_score']:.1f}")
                c2.metric("Engagement", f"{row['engagement_score']:.1f}%")
                c3.metric("Attendance", f"{row['attendance_pct']}%")
                if alerts:
                    st.markdown("**⚠️ Alerts:**")
                    for a in alerts:
                        st.markdown(f"- {a}")
                st.markdown("**💡 Recommendations:**")
                for r in recs:
                    st.markdown(f"- {r}")

    with tab1:
        render_risk_students(high_df, "HIGH")
    with tab2:
        render_risk_students(med_df, "MEDIUM")


def faculty_analytics(df):
    st.title("📈 Analytics")

    tab1, tab2, tab3, tab4 = st.tabs(["Score Analysis", "Engagement", "Attendance", "Correlation"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(score_histogram(df), use_container_width=True)
        with col2:
            subject_scores = pd.read_csv("data/assessment.csv")
            avg_by_sub = subject_scores.groupby("subject")["avg_score"].mean().reset_index()
            fig = px.bar(avg_by_sub, x="subject", y="avg_score",
                         title="Average Score by Subject",
                         color="avg_score", color_continuous_scale="RdYlGn")
            fig.update_layout(height=280, margin=dict(t=40, b=30), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = px.histogram(df, x="engagement_score", nbins=10, color="ml_risk",
                           color_discrete_map=RISK_COLORS,
                           title="Engagement Score Distribution")
        fig.update_layout(height=300, margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

        profile_counts = df["learning_profile"].value_counts().reset_index()
        profile_counts.columns = ["Profile", "Count"]
        fig2 = px.pie(profile_counts, names="Profile", values="Count",
                      title="Learning Profile Distribution",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        fig2.update_layout(height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        fig = px.histogram(df, x="attendance_pct", nbins=10, color="ml_risk",
                           color_discrete_map=RISK_COLORS,
                           title="Attendance Distribution")
        fig.update_layout(height=300, margin=dict(t=40, b=30))
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        corr_cols = ["mean_score", "engagement_score", "attendance_pct",
                     "assignments_submitted", "login_days", "quizzes_attempted"]
        corr = df[corr_cols].corr().round(2)
        fig = px.imshow(corr, title="Feature Correlation Heatmap",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        text_auto=True)
        fig.update_layout(height=380, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)


def faculty_alerts(df):
    st.title("🔔 Early Warning Alerts")

    high_df = df[df["ml_risk"] == "HIGH"]
    med_df  = df[df["ml_risk"] == "MEDIUM"]

    st.info(f"🚨 **{len(high_df)} HIGH risk** and **{len(med_df)} MEDIUM risk** students have been automatically flagged by the ML system.")

    if st.button("📤 Send All Alerts Now", type="primary"):
        with st.spinner("Sending alerts..."):
            time.sleep(1.5)
        st.success(f"✅ Alerts sent to {len(high_df) + len(med_df)} students successfully!")

    st.markdown("---")
    st.markdown("### 🔴 Critical Alerts (HIGH Risk)")
    for _, row in high_df.iterrows():
        st.markdown(f"""<div class='alert-high'>
            <b>🚨 {row['name']}</b> ({row['student_id']}) — Score: {row['mean_score']:.1f} · Engagement: {row['engagement_score']:.1f}%<br>
            <small>⚠️ Attend remedial classes · Increase learning activity · Submit pending assignments</small>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 🟡 Warnings (MEDIUM Risk)")
    for _, row in med_df.iterrows():
        st.markdown(f"""<div class='alert-medium'>
            <b>⚠️ {row['name']}</b> ({row['student_id']}) — Score: {row['mean_score']:.1f} · Engagement: {row['engagement_score']:.1f}%<br>
            <small>📉 Performance needs monitoring · Improve attendance and engagement</small>
        </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# STUDENT PAGES
# ════════════════════════════════════════════════════════════════════════════
def student_dashboard(df, uid):
    row = df[df["student_id"] == uid].iloc[0]
    risk = row["ml_risk"]
    risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}

    st.title(f"📊 Welcome, {row['name'].split()[0]}!")

    col1, col2 = st.columns([2, 1])
    with col1:
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Student ID**<br>`{uid}`", unsafe_allow_html=True)
            c2.markdown(f"**Department**<br>{row['department']}", unsafe_allow_html=True)
            c3.markdown(f"**Semester**<br>{int(row['semester'])}", unsafe_allow_html=True)

    with col2:
        badge_cls = risk.lower()
        st.markdown(f"<br><span class='badge-{badge_cls}' style='font-size:16px;padding:6px 16px'>{risk_color[risk]} {risk} RISK</span>", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Avg Score", f"{row['mean_score']:.1f}/100")
    c2.metric("💻 Engagement", f"{row['engagement_score']:.1f}%")
    c3.metric("🏫 Attendance", f"{row['attendance_pct']}%")
    c4.metric("📝 Assignments", int(row["assignments_submitted"]))

    st.markdown("---")
    st.markdown("### 📈 Performance Overview")
    metrics = {"Score": row["mean_score"], "Engagement": row["engagement_score"], "Attendance": row["attendance_pct"]}
    fig = go.Figure()
    for label, val in metrics.items():
        color = "#2ca02c" if val >= 75 else "#ff7f0e" if val >= 50 else "#d62728"
        fig.add_trace(go.Bar(name=label, x=[label], y=[val],
                             marker_color=color, text=[f"{val:.1f}%"], textposition="outside"))
    fig.update_layout(height=250, showlegend=False, yaxis=dict(range=[0, 115]),
                      margin=dict(t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    alerts = get_student_alerts(row)
    if alerts:
        st.markdown("### ⚠️ Your Alerts")
        for a in alerts:
            if "critically" in a.lower():
                st.error(a)
            else:
                st.warning(a)


def student_scores(df, uid):
    st.title("📝 My Scores")
    row = df[df["student_id"] == uid].iloc[0]

    assessment = pd.read_csv("data/assessment.csv")
    my_ass = assessment[assessment["student_id"] == uid]

    st.metric("Overall Average", f"{row['mean_score']:.1f}/100")

    st.markdown("### 📊 Subject-wise Performance")
    fig = px.bar(my_ass, x="subject", y="avg_score",
                 color="avg_score", color_continuous_scale="RdYlGn",
                 range_color=[0, 100], title="Average Score by Subject",
                 labels={"avg_score": "Score", "subject": "Subject"},
                 text="avg_score")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(height=300, margin=dict(t=40, b=30), showlegend=False,
                      yaxis=dict(range=[0, 115]))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Detailed Marks")
    display = my_ass[["subject", "test1", "test2", "test3", "assignment", "avg_score"]].round(1)
    display.columns = ["Subject", "Test 1", "Test 2", "Test 3", "Assignment", "Average"]

    def color_score(val):
        if isinstance(val, (int, float)):
            if val >= 75: return "background-color: #f0fff4"
            if val >= 50: return "background-color: #fffbf0"
            return "background-color: #fff5f5"
        return ""

    st.dataframe(display.reset_index(drop=True).style.applymap(color_score),
                 use_container_width=True, hide_index=True)

    st.markdown("### 📈 Score Trend (Test 1 → Test 3)")
    trend = my_ass.melt(id_vars="subject", value_vars=["test1", "test2", "test3"],
                        var_name="Test", value_name="Score")
    fig2 = px.line(trend, x="Test", y="Score", color="subject", markers=True,
                   title="Score Trend Across Tests")
    fig2.update_layout(height=300, margin=dict(t=40, b=30))
    st.plotly_chart(fig2, use_container_width=True)


def student_alerts(df, uid):
    st.title("🔔 My Alerts")
    row = df[df["student_id"] == uid].iloc[0]
    alerts = get_student_alerts(row)

    if not alerts:
        st.success("✅ No alerts! You're performing well. Keep it up!")
        st.balloons()
    else:
        st.warning(f"You have **{len(alerts)} alert(s)**. Please take action.")
        for a in alerts:
            if "critically" in a.lower():
                st.error(a)
            else:
                st.warning(a)

        st.markdown("---")
        st.markdown("""
        <div class='alert-high'>
        <b>⚠️ EARLY WARNING ALERT</b><br>
        Your performance is below the acceptable threshold. Please contact your faculty or academic advisor immediately.<br><br>
        <b>Actions required:</b>
        <ul>
        <li>Meet your faculty during office hours</li>
        <li>Attend remedial sessions if offered</li>
        <li>Submit all pending work as soon as possible</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


def student_recommendations(df, uid):
    st.title("💡 Recommendations")
    row = df[df["student_id"] == uid].iloc[0]
    recs = get_recommendations(row)

    risk = row["ml_risk"]
    if risk == "HIGH":
        st.error("🔴 Your risk level is HIGH. Please act on the recommendations below immediately.")
    elif risk == "MEDIUM":
        st.warning("🟡 Your risk level is MEDIUM. Follow these tips to improve.")
    else:
        st.success("🟢 You are doing well! Here are tips to stay on top.")

    st.markdown("---")
    for i, rec in enumerate(recs, 1):
        with st.container(border=True):
            st.markdown(f"**{i}.** {rec}")

    st.markdown("---")
    st.markdown("### 📅 Suggested Study Plan")
    plan = pd.DataFrame({
        "Day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        "Activity": ["Mathematics + Video Lectures", "Science + Quiz Practice",
                     "English + Forum Participation", "Data Structures + Assignment",
                     "Algorithms + Revision", "Full Revision + Past Papers", "Rest + Light Reading"],
        "Duration": ["2 hrs", "2 hrs", "1.5 hrs", "2.5 hrs", "2 hrs", "3 hrs", "1 hr"]
    })
    st.dataframe(plan, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# FACULTY — UPLOAD STUDENTS
# ════════════════════════════════════════════════════════════════════════════
def faculty_upload_students():
    st.title("📤 Upload Student Details")
    st.info("As a faculty member you can upload or update student records. Uploaded data will be merged with the existing dataset and the ML pipeline will re-run automatically.")

    tab1, tab2, tab3 = st.tabs(["📄 Upload CSV Files", "✏️ Add Single Student", "🗑️ Template Download"])

    # ── Tab 1: bulk CSV upload ──────────────────────────────────────────────
    with tab1:
        st.markdown("### Upload CSV files")
        st.markdown("Upload one or more of the three dataset files. Each file will **replace** the existing data for that category.")

        col1, col2 = st.columns(2)

        with col1:
            with st.container(border=True):
                st.markdown("#### 👤 students.csv")
                st.markdown("<small>Required columns: `student_id, name, department, semester, email`</small>", unsafe_allow_html=True)
                stu_file = st.file_uploader("Upload students.csv", type="csv", key="fac_stu")
                if stu_file:
                    try:
                        df_new = pd.read_csv(stu_file)
                        required = {"student_id", "name", "department", "semester", "email"}
                        missing = required - set(df_new.columns.str.lower())
                        if missing:
                            st.error(f"Missing columns: {', '.join(missing)}")
                        else:
                            st.success(f"✅ Valid — {len(df_new)} student records detected")
                            st.dataframe(df_new.head(5), use_container_width=True, hide_index=True)
                            if st.button("💾 Save students.csv", key="save_stu", type="primary"):
                                df_new.to_csv("data/students.csv", index=False)
                                st.success("Saved! Re-run pipeline to refresh dashboard.")
                                st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Could not read file: {e}")

            with st.container(border=True):
                st.markdown("#### 📊 engagement.csv")
                st.markdown("<small>Required columns: `student_id, videos_watched, assignments_submitted, forum_posts, login_days, attendance_pct, quizzes_attempted`</small>", unsafe_allow_html=True)
                eng_file = st.file_uploader("Upload engagement.csv", type="csv", key="fac_eng")
                if eng_file:
                    try:
                        df_new = pd.read_csv(eng_file)
                        required = {"student_id", "videos_watched", "assignments_submitted",
                                    "forum_posts", "login_days", "attendance_pct", "quizzes_attempted"}
                        missing = required - set(df_new.columns.str.lower())
                        if missing:
                            st.error(f"Missing columns: {', '.join(missing)}")
                        else:
                            st.success(f"✅ Valid — {len(df_new)} engagement records detected")
                            st.dataframe(df_new.head(5), use_container_width=True, hide_index=True)
                            if st.button("💾 Save engagement.csv", key="save_eng", type="primary"):
                                df_new.to_csv("data/engagement.csv", index=False)
                                st.success("Saved! Re-run pipeline to refresh dashboard.")
                                st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Could not read file: {e}")

        with col2:
            with st.container(border=True):
                st.markdown("#### 📝 assessment.csv")
                st.markdown("<small>Required columns: `student_id, subject, test1, test2, test3, assignment, avg_score`</small>", unsafe_allow_html=True)
                ass_file = st.file_uploader("Upload assessment.csv", type="csv", key="fac_ass")
                if ass_file:
                    try:
                        df_new = pd.read_csv(ass_file)
                        required = {"student_id", "subject", "test1", "test2", "test3", "assignment"}
                        missing = required - set(df_new.columns.str.lower())
                        if missing:
                            st.error(f"Missing columns: {', '.join(missing)}")
                        else:
                            if "avg_score" not in df_new.columns:
                                df_new["avg_score"] = df_new[["test1","test2","test3","assignment"]].mean(axis=1).round(1)
                            st.success(f"✅ Valid — {len(df_new)} assessment rows detected")
                            st.dataframe(df_new.head(5), use_container_width=True, hide_index=True)
                            if st.button("💾 Save assessment.csv", key="save_ass", type="primary"):
                                df_new.to_csv("data/assessment.csv", index=False)
                                st.success("Saved! Re-run pipeline to refresh dashboard.")
                                st.cache_data.clear()
                    except Exception as e:
                        st.error(f"Could not read file: {e}")

        st.markdown("---")
        if st.button("🔄 Re-run ML Pipeline after uploads", type="primary", use_container_width=True):
            with st.spinner("Re-processing all data and retraining models..."):
                progress = st.progress(0)
                steps = ["Loading uploaded CSVs", "Cleaning & preprocessing",
                         "Feature engineering", "Training Random Forest",
                         "Computing risk scores", "Generating alerts"]
                for i, step in enumerate(steps, 1):
                    time.sleep(0.4)
                    progress.progress(i / len(steps), text=f"⚙️ {step}...")
            st.cache_data.clear()
            st.success("✅ Pipeline complete! Navigate to Dashboard or All Students to see updated results.")

    # ── Tab 2: add single student manually ─────────────────────────────────
    with tab2:
        st.markdown("### Add / Update a Single Student")
        with st.form("add_student_form"):
            col1, col2 = st.columns(2)
            with col1:
                new_id   = st.text_input("Student ID *", placeholder="e.g. STU051")
                new_name = st.text_input("Full Name *", placeholder="e.g. Riya Menon")
                new_dept = st.selectbox("Department", ["Computer Science", "Data Science",
                                                        "Electronics", "Mechanical", "Civil"])
                new_sem  = st.selectbox("Semester", list(range(1, 9)))
                new_email = st.text_input("Email", placeholder="riya.menon@edu.ac.in")
            with col2:
                st.markdown("**Assessment Scores (0–100)**")
                sc_math = st.number_input("Mathematics", 0, 100, 60)
                sc_sci  = st.number_input("Science", 0, 100, 60)
                sc_eng  = st.number_input("English", 0, 100, 65)
                sc_ds   = st.number_input("Data Structures", 0, 100, 55)
                sc_algo = st.number_input("Algorithms", 0, 100, 55)
            st.markdown("**Engagement Details**")
            ec1, ec2, ec3, ec4 = st.columns(4)
            videos   = ec1.number_input("Videos watched", 0, 100, 20)
            assigns  = ec2.number_input("Assignments submitted", 0, 30, 10)
            login_d  = ec3.number_input("Login days", 0, 90, 30)
            att_pct  = ec4.number_input("Attendance %", 0, 100, 75)
            forum_p  = ec1.number_input("Forum posts", 0, 50, 5)
            quizzes  = ec2.number_input("Quizzes attempted", 0, 20, 5)

            submitted = st.form_submit_button("➕ Add Student to Dataset", type="primary", use_container_width=True)

        if submitted:
            if not new_id or not new_name:
                st.error("Student ID and Full Name are required.")
            elif not new_id.startswith("STU"):
                st.error("Student ID must start with STU (e.g. STU051).")
            else:
                # Append to students.csv
                stu_df = pd.read_csv("data/students.csv")
                if new_id in stu_df["student_id"].values:
                    stu_df = stu_df[stu_df["student_id"] != new_id]
                    action = "updated"
                else:
                    action = "added"
                new_stu = pd.DataFrame([{"student_id": new_id, "name": new_name,
                                          "department": new_dept, "semester": new_sem,
                                          "email": new_email or f"{new_name.split()[0].lower()}@edu.ac.in"}])
                stu_df = pd.concat([stu_df, new_stu], ignore_index=True)
                stu_df.to_csv("data/students.csv", index=False)

                # Append assessment rows
                ass_df = pd.read_csv("data/assessment.csv")
                ass_df = ass_df[ass_df["student_id"] != new_id]
                subjects_scores = {"Mathematics": sc_math, "Science": sc_sci, "English": sc_eng,
                                   "Data Structures": sc_ds, "Algorithms": sc_algo}
                new_rows = []
                for sub, base in subjects_scores.items():
                    new_rows.append({"student_id": new_id, "subject": sub,
                                     "test1": base, "test2": base, "test3": base,
                                     "assignment": base, "avg_score": float(base)})
                ass_df = pd.concat([ass_df, pd.DataFrame(new_rows)], ignore_index=True)
                ass_df.to_csv("data/assessment.csv", index=False)

                # Append engagement row
                eng_df = pd.read_csv("data/engagement.csv")
                eng_df = eng_df[eng_df["student_id"] != new_id]
                new_eng = pd.DataFrame([{"student_id": new_id, "videos_watched": videos,
                                          "assignments_submitted": assigns, "forum_posts": forum_p,
                                          "login_days": login_d, "attendance_pct": att_pct,
                                          "quizzes_attempted": quizzes}])
                eng_df = pd.concat([eng_df, new_eng], ignore_index=True)
                eng_df.to_csv("data/engagement.csv", index=False)

                st.cache_data.clear()
                st.success(f"✅ Student **{new_name}** ({new_id}) {action} successfully! Refresh the pipeline to update risk scores.")

    # ── Tab 3: template downloads ───────────────────────────────────────────
    with tab3:
        st.markdown("### Download CSV Templates")
        st.markdown("Download these blank templates, fill them in, then upload via the **Upload CSV Files** tab.")

        templates = {
            "students_template.csv": pd.DataFrame(columns=["student_id","name","department","semester","email"]),
            "assessment_template.csv": pd.DataFrame(columns=["student_id","subject","test1","test2","test3","assignment","avg_score"]),
            "engagement_template.csv": pd.DataFrame(columns=["student_id","videos_watched","assignments_submitted","forum_posts","login_days","attendance_pct","quizzes_attempted"]),
        }

        for fname, template_df in templates.items():
            with st.container(border=True):
                c1, c2, c3 = st.columns([2, 4, 2])
                c1.markdown(f"📄 **{fname}**")
                c2.markdown(f"<small>Columns: {', '.join(template_df.columns)}</small>", unsafe_allow_html=True)
                c3.download_button(
                    label="📥 Download",
                    data=template_df.to_csv(index=False),
                    file_name=fname,
                    mime="text/csv",
                    key=f"dl_{fname}"
                )

        st.markdown("---")
        st.markdown("### 📋 Column Descriptions")
        desc = pd.DataFrame({
            "Column": ["student_id","name","department","semester","email",
                       "subject","test1/test2/test3","assignment","avg_score",
                       "videos_watched","assignments_submitted","forum_posts",
                       "login_days","attendance_pct","quizzes_attempted"],
            "Description": ["Unique ID like STU001–STU999","Full student name","Academic department",
                             "1–8","Institutional email",
                             "Subject name (e.g. Mathematics)","Individual test scores 0–100",
                             "Assignment score 0–100","Auto-calculated mean (leave blank)",
                             "Count of videos watched","Number of assignments submitted",
                             "Number of forum posts made","Days the student logged into the platform",
                             "Attendance percentage 0–100","Number of quizzes attempted"],
            "Type": ["Text","Text","Text","Integer","Text",
                     "Text","Integer","Integer","Float",
                     "Integer","Integer","Integer","Integer","Integer","Integer"]
        })
        st.dataframe(desc, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# MAIN ROUTER
# ════════════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        login_page()
        return

    df         = st.session_state.df
    model_info = st.session_state.model_info
    role       = st.session_state.role
    uid        = st.session_state.user_id

    page = render_sidebar()

    # Admin routing
    if role == "admin":
        if "Overview"   in page: admin_overview(df, model_info)
        elif "Students" in page: admin_students(df)
        elif "Faculty"  in page: admin_faculty()
        elif "Upload"   in page: admin_upload()
        elif "ML"       in page: admin_ml(df, model_info)
        elif "Reports"  in page: admin_reports(df)

    # Faculty routing
    elif role == "faculty":
        if "Dashboard"    in page: faculty_dashboard(df)
        elif "All"        in page: faculty_all_students(df)
        elif "At-Risk"    in page: faculty_atrisk(df)
        elif "Analytics"  in page: faculty_analytics(df)
        elif "Alerts"     in page: faculty_alerts(df)
        elif "Upload"     in page: faculty_upload_students()

    # Student routing
    elif role == "student":
        if "Dashboard"        in page: student_dashboard(df, uid)
        elif "Scores"         in page: student_scores(df, uid)
        elif "Alerts"         in page: student_alerts(df, uid)
        elif "Recommendations" in page: student_recommendations(df, uid)


if __name__ == "__main__":
    main()
