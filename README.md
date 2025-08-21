# 🧑‍🎓 Face Recognition Attendance System

A **Tkinter-based desktop application** for managing student attendance using **face recognition**.
The system captures student images, trains a recognition model, and automatically marks **entry (IN)** and **exit (OUT)** attendance in a CSV file.

---

## 🚀 Features

- 🎭 **Face Recognition with OpenCV**
  - Uses **LBPH (Local Binary Patterns Histogram)** recognizer.
  - Detects and recognizes faces in real-time.

- 📝 **Student Registration**
  - Auto-validates students against a **master student list CSV**.
  - Prevents duplicate registrations.

- 📷 **Image Capture**
  - Captures 10 images per student for model training.

- 🧑‍🏫 **Attendance Management**
  - Marks **IN** and **OUT** timestamps automatically.
  - Saves attendance logs in **Attendance.csv**.
  - Displays attendance log in the GUI.

- 🖥️ **User-Friendly Tkinter GUI**
  - Dropdown for selecting student ID (auto-fills name).
  - Buttons for **Take Images**, **Train Model**, **Recognize Face**, **Delete Attendance**.
  - Real-time attendance log display.

---

## 🖥️ UI Layout

- **Student ID Dropdown** → Fetches IDs from the master CSV.
- **Auto-fill Name** → When an ID is selected, the name auto-populates.
- **Main Buttons**:
  - ✅ Take Images (capture dataset)
  - 🛠 Train Model (train LBPH recognizer)
  - 👤 Recognize Face (mark attendance live)
  - 🗑 Delete Attendance (clear attendance records)
- **Attendance Log** → Displays all entry/exit events in real time.

---

## ⚙️ Technical Implementation

- **Face Detection** → Haar Cascade (`haarcascade_frontalface_default.xml`)
- **Face Recognition** → LBPH algorithm (`cv2.face.LBPHFaceRecognizer_create`)
- **Data Storage**:
  - Student images → `TrainingImage/`
  - Trained model → `TrainingImageLabel/Trainer.yml`
  - ID mapping → `TrainingImageLabel/id_mapping.csv`
  - Student details → `StudentDetails/StudentDetails.csv`
  - Attendance records → `Attendance.csv`
- **Master Student List** → `Copy-of-fintech-students-24-26-_1__1_.csv`

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance
