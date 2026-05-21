import pandas as pd
import numpy as np
import os

np.random.seed(42)
n = 50

names = [
    "Aarav Sharma","Priya Mehta","Rohan Kumar","Ananya Patel","Vikram Rao",
    "Sneha Tiwari","Arjun Nair","Kavya Lal","Rahul Bose","Deepa Verma",
    "Sanjay Choudhary","Meera Joshi","Karthik Hegde","Pooja Singh","Aditya Reddy",
    "Lakshmi Pillai","Nikhil Gupta","Swati Desai","Suresh Mishra","Nisha Krishnan",
    "Tanvi Agarwal","Manish Pandey","Divya Rao","Ravi Trivedi","Anjali Nambiar",
    "Shreyas Kulkarni","Preethi Varma","Aryan Bajaj","Shalini Chatterjee","Vivek Menon",
    "Keerthi Jain","Ashwin Shetty","Sowmya Prakash","Dhruv Rastogi","Ranjitha Anand",
    "Tarun Goel","Bhavana Hegde","Siddharth Lal","Rashmi Tanna","Naveen Kapoor",
    "Anitha Varghese","Gaurav Puri","Sreedevi Mohan","Mithun Rajan","Lavanya Bhat",
    "Kiran Srinivas","Padma Chandra","Rajesh Naidu","Vennela Devi","Harsha Teja"
]

student_ids = [f"STU{str(i+1).padStart(3,'0') if False else str(i+1).zfill(3)}" for i in range(n)]
departments = np.random.choice(["Computer Science","Data Science","Electronics","Mechanical","Civil"], n)

students_df = pd.DataFrame({
    "student_id": student_ids,
    "name": names,
    "department": departments,
    "semester": np.random.randint(1, 9, n),
    "email": [f"{name.split()[0].lower()}.{name.split()[1].lower()}@edu.ac.in" for name in names]
})

subjects = ["Mathematics","Science","English","Data Structures","Algorithms"]
rows = []
for sid in student_ids:
    base = np.random.randint(28, 92)
    for sub in subjects:
        score = max(0, min(100, base + np.random.randint(-12, 13)))
        rows.append({"student_id": sid, "subject": sub,
                     "test1": max(0,min(100,score+np.random.randint(-8,9))),
                     "test2": max(0,min(100,score+np.random.randint(-8,9))),
                     "test3": max(0,min(100,score+np.random.randint(-8,9))),
                     "assignment": max(0,min(100,score+np.random.randint(-5,6)))})
assessment_df = pd.DataFrame(rows)
assessment_df["avg_score"] = assessment_df[["test1","test2","test3","assignment"]].mean(axis=1).round(1)

engagement_df = pd.DataFrame({
    "student_id": student_ids,
    "videos_watched": np.random.randint(0, 60, n),
    "assignments_submitted": np.random.randint(0, 20, n),
    "forum_posts": np.random.randint(0, 30, n),
    "login_days": np.random.randint(1, 90, n),
    "attendance_pct": np.random.randint(40, 100, n),
    "quizzes_attempted": np.random.randint(0, 15, n)
})

os.makedirs("data", exist_ok=True)
students_df.to_csv("data/students.csv", index=False)
assessment_df.to_csv("data/assessment.csv", index=False)
engagement_df.to_csv("data/engagement.csv", index=False)
print("Data generated successfully.")
print(f"Students: {len(students_df)}, Assessment rows: {len(assessment_df)}, Engagement rows: {len(engagement_df)}")
