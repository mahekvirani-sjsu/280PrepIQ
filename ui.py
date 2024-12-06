# streamlit_app.py: Streamlit UI
import streamlit as st
import requests
import os

st.set_page_config(page_title="Document to VectorDB, Quiz Generation, and S3 Management", layout="wide")

def main():
    tabs = st.tabs(["VectorDB Management", "Quiz Generation"])
    # VectorDB Management tab
    with tabs[0]:
        st.subheader("Manage VectorDB")
        st.write("Upload FAISS index files and manage your vector database folders in S3.")

        # Input and button to upload FAISS index to S3
        s3_folder_name = st.text_input("Enter S3 folder name to upload FAISS index")
        uploaded_file = st.file_uploader("Upload PDF file to create a FAISS index or FAISS index file", type=["pdf", "faiss", "zip"])
        if st.button("Upload PDF or FAISS Index", key="upload_faiss_button"):
            if s3_folder_name and uploaded_file:
                with st.spinner("Uploading and processing document..."):
                    try:
                        # Check file type to decide the process
                        if uploaded_file.type == "application/pdf":
                            # Call the FastAPI endpoint to process PDF and create FAISS index
                            response = requests.post("http://localhost:8000/upload-documents", timeout=120)
                            if response.status_code == 200:
                                st.success("PDF files successfully processed and FAISS index created.")
                                # After creating FAISS index, call another endpoint to upload it to S3
                                upload_response = requests.post("http://localhost:8000/upload-faiss-to-s3", json={"s3_folder_name": s3_folder_name}, timeout=60)
                                if upload_response.status_code == 200:
                                    st.success(f"FAISS index successfully uploaded to s3://cmpe-280/{s3_folder_name}")
                                else:
                                    st.error(f"Error during upload to S3: {upload_response.json().get('message', 'Unknown error')}")
                            else:
                                st.error(f"Error: {response.json().get('message', 'Unknown error')}")
                        else:
                            # Upload existing FAISS index to S3
                            files = {'file': (uploaded_file.name, uploaded_file, 'application/octet-stream')}
                            response = requests.post("http://localhost:8000/upload-faiss-to-s3", files=files, data={"s3_folder_name": s3_folder_name}, timeout=60)
                            if response.status_code == 200:
                                st.success(f"FAISS index successfully uploaded to s3://cmpe-280/{s3_folder_name}")
                            else:
                                st.error(f"Error: {response.json().get('message', 'Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error("Failed to connect to the backend. Please ensure the FastAPI server is running.")
                    except requests.exceptions.Timeout:
                        st.error("The request timed out. Please try again later.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
            else:
                st.warning("Please enter a valid S3 folder name and upload a PDF or FAISS index file.")

        # Input and button to delete folder from S3
        st.write("Delete unwanted vector database folders from S3.")
        delete_s3_folder_name = st.text_input("Enter S3 folder name to delete FAISS index from S3", key="delete_s3_folder_input")
        if st.button("Delete S3 Folder", key="delete_s3_folder_button_1"):
            if delete_s3_folder_name:
                with st.spinner("Deleting folder from S3..."):
                    try:
                        response = requests.post("http://localhost:8000/delete-s3-folder", json={"s3_folder_name": delete_s3_folder_name}, timeout=60)
                        if response.status_code == 200:
                            st.success(f"Folder s3://cmpe-280/{delete_s3_folder_name} successfully deleted.")
                        else:
                            st.error(f"Error: {response.json().get('message', 'Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error("Failed to connect to the backend. Please ensure the FastAPI server is running.")
                    except requests.exceptions.Timeout:
                        st.error("The request timed out. Please try again later.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
            else:
                st.warning("Please enter a valid S3 folder name.")

    # Quiz Generation tab
    with tabs[1]:
        st.subheader("Generate Quiz from VectorDB")
        s3_topics = fetch_s3_topics()
        if not s3_topics:
            st.warning("No topics available in S3. Please upload a FAISS index to S3 first.")
        topic_name = st.selectbox("Select a topic from S3", options=s3_topics, disabled=(len(s3_topics) == 0))

        # Use session state to maintain subtopics
        if "subtopics" not in st.session_state:
            st.session_state.subtopics = []

        # Add subtopics section
        subtopic_input = st.text_input("Enter subtopic to focus on")
        if st.button("Add Subtopic"):
            if subtopic_input:
                st.session_state.subtopics.append(subtopic_input)
        
        st.write(f"Current subtopics: {st.session_state.subtopics}")

        # Generate quiz questions
        if st.button("Generate Quiz Questions", key="generate_quiz_button"):
            with st.spinner("Generating quiz questions..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/generate-quiz", 
                        json={"topic_name": topic_name, "subtopics": st.session_state.subtopics}, 
                        timeout=120
                    )
                    if response.status_code == 200:
                        st.write("### Quiz Questions")
                        quiz_questions = response.json().get('quiz_questions', [])
                        for idx, question in enumerate(quiz_questions, 1):
                            st.write(f"{idx}. {question}")
                    else:
                        st.error(f"Error: {response.json().get('message', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend. Please ensure the FastAPI server is running.")
                except requests.exceptions.Timeout:
                    st.error("The request timed out. Please try again later.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

import functools
from datetime import datetime, timedelta

CACHE_EXPIRY = timedelta(minutes=5)
_cache_time = None
_cached_topics = None

@functools.lru_cache(maxsize=1)
def fetch_s3_topics():
    global _cache_time, _cached_topics
    if _cache_time and datetime.now() - _cache_time < CACHE_EXPIRY:
        return _cached_topics
    try:
        response = requests.get("http://localhost:8000/list-s3-topics", timeout=60)
        if response.status_code == 200:
            _cached_topics = response.json().get('folders', [])
            _cache_time = datetime.now()
            return _cached_topics
        else:
            st.error(f"Error fetching topics: {response.json().get('message', 'Unknown error')}")
            return []
    except Exception as e:
        st.error(f"Failed to fetch topics: {str(e)}")
        return []

if __name__ == "__main__":
    main()
