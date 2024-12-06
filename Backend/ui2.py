import streamlit as st
import requests
import json
from datetime import datetime, timedelta
import functools
import os

st.set_page_config(page_title="Document to VectorDB and Quiz Generation", layout="wide")

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

def main():
    tabs = st.tabs(["VectorDB Management", "Quiz Generation", "Quiz Management", "All Quizzes", "Update Quiz Answers"])

    # VectorDB Management tab
    with tabs[0]:
        st.subheader("Manage VectorDB")
        st.write("Upload PDF files and manage your vector database in S3.")

        s3_folder_name = st.text_input("Enter S3 folder name to store the generated FAISS index")

        uploaded_files = st.file_uploader(
            "Upload PDF files to create a FAISS index",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if st.button("Upload PDF and Create FAISS Index", key="upload_pdf_button"):
            if s3_folder_name and uploaded_files:
                with st.spinner("Uploading and processing documents..."):
                    try:
                        files = [
                            ("files", (file.name, file, "application/pdf"))
                            for file in uploaded_files
                        ]

                        response = requests.post(
                            "http://localhost:8000/upload-documents",
                            files=files,
                            timeout=180
                        )

                        if response.status_code == 200:
                            st.success("PDF files successfully processed and FAISS index created.")
                            
                            upload_response = requests.post(
                                "http://localhost:8000/upload-faiss-to-s3",
                                json={"s3_folder_name": s3_folder_name},
                                timeout=60
                            )
                            if upload_response.status_code == 200:
                                st.success(f"FAISS index successfully uploaded to s3://cmpe-280/{s3_folder_name}")
                            else:
                                st.error(f"Error during upload to S3: {upload_response.json().get('message', 'Unknown error')}")
                        else:
                            st.error(f"Error during FAISS creation: {response.json().get('message', 'Unknown error')}")

                    except requests.exceptions.ConnectionError:
                        st.error("Failed to connect to the backend. Please ensure the FastAPI server is running.")
                    except requests.exceptions.Timeout:
                        st.error("The request timed out. Please try again later.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
            else:
                st.warning("Please enter a valid S3 folder name and upload at least one PDF file.")

        st.write("Delete unwanted vector database folders from S3.")
        delete_s3_folder_name = st.text_input("Enter S3 folder name to delete FAISS index from S3", key="delete_s3_folder_input")
        
        if st.button("Delete S3 Folder", key="delete_s3_folder_button"):
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

        if "subtopics" not in st.session_state:
            st.session_state.subtopics = []

        subtopic_input = st.text_input("Enter subtopic to focus on")
        if st.button("Add Subtopic"):
            if subtopic_input:
                st.session_state.subtopics.append(subtopic_input)

        if len(st.session_state.subtopics) > 0:
            st.write("### Current Subtopics")
            for i, subtopic in enumerate(st.session_state.subtopics):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{i + 1}. {subtopic}")
                with col2:
                    if st.button("Remove", key=f"remove_{i}"):
                        st.session_state.subtopics.pop(i)

        # Always show "Store Quiz" button, but disable it when quiz is not yet generated
        generate_button = st.button("Generate Quiz Questions", key="generate_quiz_button")
        
        # Checking if quiz questions are generated
        quiz_generated = False
        if "quiz_questions" in st.session_state:
            quiz_generated = True

        store_button = st.button("Store Quiz", key="store_quiz_button", disabled=not quiz_generated)

        if store_button:
            with st.spinner("Storing quiz data..."):
                try:
                    # Send only the topic_name and subtopics to the backend for storing
                    store_response = requests.post(
                        "http://localhost:8000/store-quiz",
                        json={"topic_name": topic_name, "subtopics": st.session_state.subtopics},
                        timeout=60
                    )

                    if store_response.status_code == 200:
                        st.success("Quiz stored successfully in the database.")
                        # Clear stored quiz after successful store
                        del st.session_state.quiz_questions
                    else:
                        st.error(f"Error storing quiz: {store_response.json().get('message', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend. Please ensure the FastAPI server is running.")
                except requests.exceptions.Timeout:
                    st.error("The request timed out. Please try again later.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

        # If generate quiz is clicked
        if generate_button:
            with st.spinner("Generating quiz questions..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/generate-quiz",
                        json={"topic_name": topic_name, "subtopics": st.session_state.subtopics},
                        timeout=120
                    )
                    if response.status_code == 200:
                        try:
                            quiz_questions = response.json().get('quiz_questions', [])
                        except ValueError:
                            st.error("Invalid response from server. Could not parse JSON.")
                            return

                        if not quiz_questions:
                            st.warning("No quiz questions generated. Please check your inputs.")
                        else:
                            # Save quiz questions in session state
                            st.session_state.quiz_questions = quiz_questions
                            
                            for idx, question in enumerate(quiz_questions, 1):
                                st.write(f"**{idx}. {question['question']}**")
                                for i, choice in enumerate(question['choices']):
                                    st.write(f"{chr(65 + i)}. {choice}")
                                st.write(f"*Correct Answer:* {question['answer']}")
                    else:
                        try:
                            error_message = response.json().get('message', 'Unknown error')
                        except ValueError:
                            error_message = "Invalid response from server."
                        st.error(f"Error: {error_message}")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend. Please ensure the FastAPI server is running.")
                except requests.exceptions.Timeout:
                    st.error("The request timed out. Please try again later.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
    
    # Quiz Management tab
    with tabs[2]:
        st.subheader("Manage Quizzes")
        st.write("Retrieve and delete quizzes from the database.")

        quiz_topics = fetch_s3_topics()
        topic_for_quiz = st.selectbox("Select a topic to manage quizzes", options=quiz_topics)

        if st.button("Retrieve Quizzes"):
            with st.spinner("Fetching quizzes..."):
                try:
                    response = requests.get(f"http://localhost:8000/quizzes/{topic_for_quiz}", timeout=60)
                    if response.status_code == 200:
                        quizzes = response.json()
                        if quizzes:
                            for quiz in quizzes:
                                st.write(f"**Quiz ID:** {quiz['quiz_id']}")
                                st.write(f"**Topic:** {quiz['topic_name']}")
                                st.write(f"**Created On:** {quiz['created_at']}")
                                st.write("---")
                        else:
                            st.warning("No quizzes found for this topic.")
                    else:
                        st.error(f"Error fetching quizzes: {response.json().get('message', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Failed to retrieve quizzes: {str(e)}")

        quiz_id_to_delete = st.text_input("Enter the Quiz ID to delete")
        
        if st.button("Delete Quiz"):
            if quiz_id_to_delete:
                with st.spinner("Deleting quiz..."):
                    try:
                        response = requests.delete(f"http://localhost:8000/delete-quiz/{quiz_id_to_delete}", timeout=60)
                        if response.status_code == 200:
                            st.success(f"Quiz with ID {quiz_id_to_delete} successfully deleted.")
                        else:
                            st.error(f"Error deleting quiz: {response.json().get('message', 'Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error("Failed to connect to the backend. Please ensure the FastAPI server is running.")
                    except requests.exceptions.Timeout:
                        st.error("The request timed out. Please try again later.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
            else:
                st.warning("Please enter a valid Quiz ID to delete.")

    # New "All Quizzes" tab
    with tabs[3]:
        st.subheader("Retrieve All Quizzes")
        st.write("This tab fetches and displays all quizzes from the database.")

        if st.button("Retrieve All Quizzes"):
            with st.spinner("Fetching all quizzes..."):
                try:
                    # Make a GET request to the endpoint that retrieves all quizzes
                    response = requests.get("http://localhost:8000/get-all-quizzes", timeout=60)
                    
                    if response.status_code == 200:
                        quizzes = response.json()
                        if quizzes:
                            # Display all quizzes
                            for quiz in quizzes:
                                st.write(f"**Quiz ID:** {quiz['quiz_id']}")
                                st.write(f"**Topic Name:** {quiz['topic_name']}")
                                st.write(f"**Subtopics:** {', '.join(quiz['subtopics'])}")
                                st.write(f"**Created On:** {quiz['created_at']}")
                                st.write(f"**Quiz Data:** {json.dumps(quiz['quiz_data'], indent=2)}")
                                st.write("---")
                        else:
                            st.warning("No quizzes found.")
                    else:
                        st.error(f"Error retrieving quizzes: {response.json().get('message', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Failed to connect to the backend. Please ensure the FastAPI server is running.")
                except requests.exceptions.Timeout:
                    st.error("The request timed out. Please try again later.")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")

    with tabs[4]:
        st.subheader("Update Quiz Answers")
        st.write("Enter your answers for the following 5 questions:")

        # Define the list to store the answers
        answers = []
        for i in range(1, 6):
            answer = st.text_input(f"Answer for Question {i}", key=f"answer_{i}")
            answers.append(answer)

        # Button to submit the answers
        if st.button("Update Answers"):
            if all(answer for answer in answers):  # Ensure all fields are filled
                with st.spinner("Updating answers..."):
                    try:
                        # Send answers as a list to the backend
                        response = requests.post(
                            "http://localhost:8000/update_answers",  # The endpoint that accepts the answers
                            json={"your_answers": answers},
                            timeout=60
                        )
                        
                        if response.status_code == 200:
                            st.success("Answers updated successfully!")
                        else:
                            st.error(f"Error updating answers: {response.json().get('message', 'Unknown error')}")
                    except requests.exceptions.ConnectionError:
                        st.error("Failed to connect to the backend. Please ensure the FastAPI server is running.")
                    except requests.exceptions.Timeout:
                        st.error("The request timed out. Please try again later.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
            else:
                st.warning("Please fill in all 5 answers before submitting.")

if __name__ == "__main__":
    main()
