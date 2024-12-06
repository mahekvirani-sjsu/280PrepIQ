# **Document to VectorDB, Quiz Generation, and S3 Management**

This project includes a backend implemented with FastAPI for managing FAISS indexes, quiz generation, and S3 integration, along with a frontend implemented with Streamlit for testing and interacting with the APIs.

---

## **Table of Contents**
- [Prerequisites](#prerequisites)
- [Setting up the Backend](#setting-up-the-backend)
- [Setting up the Frontend (Streamlit UI)](#setting-up-the-frontend-streamlit-ui)
- [Testing the Application](#testing-the-application)
- [Project Features](#project-features)
- [File Structure](#file-structure)

---

## **Prerequisites**
Ensure the following tools are installed on your machine:
1. Python (>= 3.8)
2. Pip
3. Virtual Environment Tool (e.g., `venv`, `virtualenv`)
4. AWS CLI (for S3 operations)
5. Streamlit

---

## **Setting up the Backend**

### 1. **Clone the Repository**
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. **Set up a Python Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate    # For Linux/Mac
venv\Scripts\activate       # For Windows
```

### 3. **Install Backend Dependencies**
```bash
pip install -r requirements.txt
```

### 4. **Environment Variables**
Create a `.env` file in the root directory to set up the following environment variables:
```env
AWS_ACCESS_KEY_ID=<your_aws_access_key>
AWS_SECRET_ACCESS_KEY=<your_aws_secret_key>
AWS_REGION=<your_region>
S3_BUCKET_NAME=<your_bucket_name>
```

### 5. **Run the FastAPI Backend**
To run the FastAPI server locally:
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
- `--reload`: Allows auto-reload on code changes.
- `--host 0.0.0.0`: Makes the API accessible externally for testing.

The backend will be available at `http://localhost:8000`.

---

## **Setting up the Frontend (Streamlit UI)**
The Streamlit UI allows users to interact with the backend APIs, including VectorDB management, quiz generation, and S3 operations.

### 1. **Install Streamlit**
If Streamlit is not installed, install it using pip:
```bash
pip install streamlit
```

### 2. **Run the Streamlit App**
To launch the Streamlit UI:
```bash
streamlit run streamlit_app.py
```
- The UI will open in your default browser at `http://localhost:8501`.

---

## **Testing the Application**

### **Backend Endpoints**
1. **Upload Documents to Create FAISS Index**
   - Endpoint: `POST /upload-documents`
   - Use this to process a PDF and create a FAISS index.

2. **Upload FAISS Index to S3**
   - Endpoint: `POST /upload-faiss-to-s3`
   - Uploads a previously created FAISS index to a specific S3 folder.

3. **Generate Quiz**
   - Endpoint: `POST /generate-quiz`
   - Generates quiz questions based on a topic and subtopics from the FAISS index.

### **Streamlit UI Features**
- **Upload PDF or FAISS Index**: Allows uploading documents to create or manage FAISS indexes.
- **Delete S3 Folders**: Manage your S3 storage by deleting unwanted folders.
- **Generate Quiz**: Select a topic, add subtopics, and generate quiz questions interactively.

---

## **Project Features**
- **VectorDB Management**: Upload and manage FAISS index files with integration to AWS S3.
- **Quiz Generation**: Generate quizzes based on documents uploaded to VectorDB.
- **Interactive UI**: Streamlit-based UI to interact with the backend and simplify testing.

---

## **File Structure**
```
<repository-folder>/
├── app.py                   # FastAPI backend logic
├── requirements.txt         # Python dependencies
├── streamlit_app.py         # Streamlit UI for interaction
├── .env                     # Environment variables (not committed to repo)
└── README.md                # Project documentation
```

---

## **Notes**
- **AWS Credentials**: Make sure your AWS credentials are correctly configured in `.env` or via the AWS CLI (`aws configure`).
- **API Availability**: Ensure the FastAPI server is running before using the Streamlit UI.

---

## **Contact**
If you face any issues or have questions, please reach out to the maintainer at `<your_email@domain.com>`.

