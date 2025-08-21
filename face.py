import tkinter as tk
from tkinter import messagebox as mess, simpledialog, ttk
import cv2
import os
import csv
import numpy as np
import pandas as pd
import time
from PIL import Image

def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def validate_student_details(user_id, user_name):
    try:
        df = pd.read_csv("Copy-of-fintech-students-24-26-_1__1_.csv")
        # Convert both to string for comparison
        user_id = str(user_id)
        user_name = str(user_name).strip()
        
        # Check if student exists in the CSV
        student = df[df["UID"] == user_id]
        if not student.empty:
            if student.iloc[0]["Name"].strip() == user_name:
                return True
        return False
    except Exception as e:
        print(f"Error reading student CSV: {e}")
        return False

def check_student_exists(user_id, user_name):
    # First check if student is in the master CSV
    if not validate_student_details(user_id, user_name):
        mess.showerror("Invalid Student", 
                      "Student ID or Name not found in the master list. Please check the details.")
        return True
    
    # Then check if already registered in our system
    if not os.path.exists("StudentDetails/StudentDetails.csv"):
        return False
    
    with open("StudentDetails/StudentDetails.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if row and len(row) >= 2:
                if row[0] == user_id or row[1].lower() == user_name.lower():
                    mess.showerror("Registration Error", 
                                 "This student is already registered in the system.")
                    return True
    return False

def take_images():
    assure_path_exists("TrainingImage/")
    assure_path_exists("StudentDetails/")
    user_id = id_combo.get()  # Get ID from combobox
    user_name = txt2.get()
    
    # Input validation
    if not user_id or not user_name:
        mess.showerror("Invalid Input", "Please select an ID and ensure name is filled.")
        return
    
    # Check for duplicate registration and validate against master CSV
    if check_student_exists(user_id, user_name):
        return
    
    cam = cv2.VideoCapture(0)
    count = 0
    while count < 10:
        ret, img = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"TrainingImage/{user_name}.{user_id}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the video feed
        cv2.imshow("Taking Images", img)
        cv2.waitKey(1)
    
    cam.release()
    cv2.destroyAllWindows()
    
    if count > 0:
        with open("StudentDetails/StudentDetails.csv", "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([user_id, user_name])
        mess.showinfo("Success", f"{count} Images saved for ID: {user_id}, Name: {user_name}")
    else:
        mess.showwarning("No Face Detected", "No face detected. Try again.")

def train_images():
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = [], []
    
    # Create a mapping of student IDs to numeric IDs
    id_mapping = {}
    current_numeric_id = 1
    
    for image_path in os.listdir("TrainingImage/"):
        try:
            img_path = os.path.join("TrainingImage/", image_path)
            img = Image.open(img_path).convert("L")
            img_np = np.array(img, "uint8")
            
            # Extract the student ID from filename (e.g., "Name.24MFT10014.1.jpg")
            student_id = image_path.split(".")[1]
            
            # Create numeric ID mapping if not exists
            if student_id not in id_mapping:
                id_mapping[student_id] = current_numeric_id
                current_numeric_id += 1
            
            # Use the numeric ID for training
            numeric_id = id_mapping[student_id]
            faces.append(img_np)
            ids.append(numeric_id)
            
        except Exception as e:
            print(f"Skipping file {image_path}: {e}")
    
    if faces:
        # Save the ID mapping for later use
        with open("TrainingImageLabel/id_mapping.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["StudentID", "NumericID"])
            for student_id, numeric_id in id_mapping.items():
                writer.writerow([student_id, numeric_id])
        
        recognizer.train(faces, np.array(ids))
        recognizer.save("TrainingImageLabel/Trainer.yml")
        mess.showinfo("Success", "Model trained successfully!")
    else:
        mess.showerror("Error", "No images found for training.")

def save_attendance(user_id, user_name, status="IN"):
    file_path = "Attendance.csv"
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create DataFrame with headers if file doesn't exist
    if not os.path.exists(file_path):
        df = pd.DataFrame(columns=["ID", "Name", "Status", "Timestamp"])
        df.to_csv(file_path, index=False)
    
    # Read existing data
    df = pd.read_csv(file_path)
    
    # Check if student already has an "IN" status without an "OUT" status
    if status == "IN":
        student_entries = df[(df["ID"] == user_id) & (df["Name"] == user_name)]
        if not student_entries.empty and student_entries.iloc[-1]["Status"] == "IN":
            mess.showwarning("Warning", f"{user_name} is already marked as IN!")
            return
    
    # Add new entry
    new_entry = pd.DataFrame([{
        "ID": user_id,
        "Name": user_name,
        "Status": status,
        "Timestamp": current_time
    }])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(file_path, index=False)
    
    # Update attendance list display
    status_text = "entered" if status == "IN" else "exited"
    update_attendance_list(f"{user_name} {status_text} at {current_time}")

def mark_exit(user_id, user_name):
    save_attendance(user_id, user_name, "OUT")

def track_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read("TrainingImageLabel/Trainer.yml")
    except:
        mess.showerror("Error", "No trained model found. Train first.")
        return
    
    # Load the ID mapping
    id_mapping = {}
    try:
        with open("TrainingImageLabel/id_mapping.csv", "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                student_id, numeric_id = row
                id_mapping[int(numeric_id)] = student_id
    except:
        mess.showerror("Error", "ID mapping file not found. Please train the model again.")
        return
    
    names = {}
    with open("StudentDetails/StudentDetails.csv", "r") as file:
        for row in file:
            row = row.strip()
            if not row:
                continue
            values = row.split(",")
            if len(values) < 2:
                continue
            user_id, user_name = values[:2]
            names[user_id] = user_name.strip()
    
    # Read attendance data to check previous status
    attendance_data = pd.DataFrame(columns=["ID", "Name", "Status", "Timestamp"])
    if os.path.exists("Attendance.csv"):
        try:
            attendance_data = pd.read_csv("Attendance.csv")
        except:
            attendance_data = pd.DataFrame(columns=["ID", "Name", "Status", "Timestamp"])
    
    cam = cv2.VideoCapture(0)
    start_time = time.time()
    timeout = 5  # 5 seconds timeout
    recognition_start_time = None
    recognized_face = None
    
    while True:
        ret, img = cam.read()
        if not ret:
            break
            
        elapsed_time = time.time() - start_time
        remaining_time = max(0, timeout - elapsed_time)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            numeric_id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 50 and numeric_id in id_mapping:
                student_id = id_mapping[numeric_id]
                name = names.get(student_id, "Unknown")
                
                # Check student's last status
                student_entries = attendance_data[
                    (attendance_data["ID"] == student_id) & 
                    (attendance_data["Name"] == name)
                ]
                
                if student_entries.empty or student_entries.iloc[-1]["Status"] == "OUT":
                    status = "IN"
                    color = (0, 255, 0)  # Green for entry
                else:
                    status = "OUT"
                    color = (0, 0, 255)  # Red for exit
                
                cv2.putText(img, f"{name} ({status})", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                
                if recognition_start_time is None:
                    recognition_start_time = time.time()
                    recognized_face = (student_id, name, status)
                
                if time.time() - recognition_start_time >= 2:
                    if status == "IN":
                        save_attendance(student_id, name, "IN")
                        mess.showinfo("Attendance", f"Entry marked for {name}")
                    else:
                        mark_exit(student_id, name)
                        mess.showinfo("Attendance", f"Exit marked for {name}")
                    
                    cam.release()
                    cv2.destroyAllWindows()
                    return
            else:
                cv2.putText(img, "Unknown", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        timer_text = f"Time left: {int(remaining_time)}s"
        cv2.putText(img, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if recognition_start_time is not None:
            recognition_time_left = max(0, 2 - (time.time() - recognition_start_time))
            if recognition_time_left > 0:
                cv2.putText(img, f"Marking attendance in: {int(recognition_time_left)}s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Face Recognition", img)
        key = cv2.waitKey(1)
        
        if remaining_time <= 0:
            cv2.putText(img, "Camera closing...", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Face Recognition", img)
            cv2.waitKey(1000)
            break
            
        if key == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()

def update_attendance_list(name):
    attendance_list.insert(tk.END, f"{name}\n")
    attendance_list.see(tk.END)

def delete_attendance():
    file_path = "Attendance.csv"
    if os.path.exists(file_path):
        os.remove(file_path)
        attendance_list.delete(1.0, tk.END)
        mess.showinfo("Success", "Attendance records deleted.")
    else:
        mess.showwarning("Error", "No attendance records found to delete.")

window = tk.Tk()
window.title("Face Recognition System")
window.geometry("800x600")  # Reset window size
window.configure(background="#262523")

# Create a dictionary to store student data
student_data = {}

# Load student data from CSV
def load_student_data():
    try:
        df = pd.read_csv("Copy-of-fintech-students-24-26-_1__1_.csv")
        for _, row in df.iterrows():
            if pd.notna(row["UID"]) and pd.notna(row["Name"]):
                student_data[row["UID"]] = row["Name"].strip()
    except Exception as e:
        print(f"Error loading student data: {e}")

# Function to update name when ID is selected
def on_id_select(event):
    selected_id = id_combo.get()
    if selected_id in student_data:
        txt2.delete(0, tk.END)
        txt2.insert(0, student_data[selected_id])

# Load student data when window starts
load_student_data()

# Create ID dropdown
tk.Label(window, text="Select ID:", bg="#00aeff", font=("times", 12)).place(x=50, y=50)
id_combo = ttk.Combobox(window, width=20, font=("times", 12), values=list(student_data.keys()))
id_combo.place(x=200, y=50)
id_combo.bind('<<ComboboxSelected>>', on_id_select)

tk.Label(window, text="Name:", bg="#00aeff", font=("times", 12)).place(x=50, y=100)
txt2 = tk.Entry(window, width=20, font=("times", 12))
txt2.place(x=200, y=100)

# Remove the old student list and its related code
attendance_list = tk.Text(window, height=10, width=50, font=("times", 12))
attendance_list.place(x=50, y=280)

tk.Button(window, text="Take Images", command=take_images, bg="green", fg="white", font=("times", 12)).place(x=50, y=150)
tk.Button(window, text="Train Model", command=train_images, bg="blue", fg="white", font=("times", 12)).place(x=200, y=150)
tk.Button(window, text="Recognize Face", command=track_images, bg="orange", fg="white", font=("times", 12)).place(x=350, y=150)
tk.Button(window, text="Delete Attendance", command=delete_attendance, bg="red", fg="white", font=("times", 12)).place(x=500, y=150)

tk.Label(window, text="Attendance Log", bg="#00aeff", font=("times", 15)).place(x=50, y=250)

# Add instructions label
tk.Label(window, text="Select ID from dropdown to auto-fill name", 
         bg="#00aeff", font=("times", 12)).place(x=50, y=200)

window.mainloop()



###working