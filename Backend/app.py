from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Response
from langchain.document_loaders import PyPDFLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import logging
import traceback
import boto3
from fastapi.responses import JSONResponse
from fastapi import Body
from langchain_groq import ChatGroq
import json
from sqlalchemy import text


from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import shutil
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import create_engine, Column, Integer, String, JSON, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from typing import List
import shutil
import datetime
from sqlalchemy.orm import Session
from quiz_db import *
from pathlib import Path
from alembic import command
from alembic.config import Config
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from io import BytesIO
import requests


# Load environment variables from .env file
load_dotenv()

class QuizRequest(BaseModel):
    topic_name: str
    subtopics: List[str]

# Initialize FastAPI
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://your-frontend-domain.com"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)


DATABASE_URL = "sqlite:///./quiz_database.db"
Base = declarative_base()

# Define the Quiz model
class Quiz(Base):
    __tablename__ = 'quizzes'
    quiz_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_name = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)
    quiz_data = Column(JSON, nullable=False)
    subtopics = Column(JSON, nullable=False)  # Added subtopics as a list of strings

# Create the SQLite engine (in-memory)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create session local for interacting with the DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base.metadata.drop_all(bind=engine)

# Create the table in the in-memory database
# Base.metadata.create_all(bind=engine)


# Pydantic model for the quiz data
class QuizRequest(BaseModel):
    topic_name: str
    subtopics: List[str]

class QuizQuestion(BaseModel):
    question: str
    options: List[str]
    answer: str
    your_answer: str


class QuizResponse(BaseModel):
    quiz_id: int
    topic_name: str
    subtopics: List[str]
    created_at: str
    quiz_data: list

    class Config:
        orm_mode = True
    



# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to store quiz data into the database
def store_quiz_in_db(db, topic_name: str, subtopics: list, quiz_data: dict):
    """
    This function stores the quiz data into the database along with the subtopics.
    
    :param db: Database session
    :param topic_name: The name of the topic for the quiz
    :param subtopics: The list of subtopics for the quiz
    :param quiz_data: The quiz data (questions, choices, answers)
    :return: The newly created quiz record
    """
    # Create a new Quiz record
    new_quiz = Quiz(
        topic_name=topic_name,
        subtopics=subtopics,  # Store the subtopics as well
        quiz_data=quiz_data   # Store the quiz data
    )
    
    # Add the new quiz to the session
    db.add(new_quiz)
    
    # Commit the transaction to save it to the database
    db.commit()
    
    # Refresh to get the latest version of the new_quiz object
    db.refresh(new_quiz)
    
    return new_quiz




pdf_directory = 'course-documents'

@app.post("/upload-documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Upload PDFs and create a FAISS index.
    """
    temp_upload_dir = "temp_uploads"

    try:
        # Ensure the temp_uploads folder exists
        os.makedirs(temp_upload_dir, exist_ok=True)

        documents = []
        # Save uploaded files temporarily and load them
        for file in files:
            file_path = os.path.join(temp_upload_dir, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())  # Load documents from each PDF file

        # Remove temporary files after processing
        shutil.rmtree(temp_upload_dir, ignore_errors=True)

        # Split documents and create FAISS index
        chunked_docs = split_documents(documents)
        create_faiss_index(chunked_docs)

        return {"message": "Files successfully uploaded and vector DB created."}
    except Exception as e:
        # Cleanup temporary directory in case of errors
        shutil.rmtree(temp_upload_dir, ignore_errors=True)
        logging.error(f"Error processing files: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.post("/upload-faiss-to-s3")
async def upload_faiss_to_s3(s3_folder_name: dict):
    """
    This endpoint uploads the FAISS index to the specified S3 folder if it does not already exist.
    """
    """
    This endpoint uploads the FAISS index to the specified S3 folder if it does not already exist. If successful, it deletes the local FAISS index.
    If the folder already exists, notifies the user.
    """
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION")
        )
        s3_bucket = "cmpe-280"
        s3_key = f"{s3_folder_name['s3_folder_name']}/index.faiss"
        s3_key_2 = f"{s3_folder_name['s3_folder_name']}/index.pkl"

        # Check if the folder already exists in S3
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=f"{s3_folder_name['s3_folder_name']}/")
        if 'Contents' in response and len(response['Contents']) > 0:
            logging.info(f"Folder {s3_folder_name['s3_folder_name']} already exists in S3.")
            return JSONResponse(status_code=400, content={"message": f"Folder {s3_folder_name['s3_folder_name']} already exists in S3. Please delete it before proceeding."})

        # Upload FAISS index to S3
        s3.upload_file("faiss_index/index.faiss", s3_bucket, s3_key)
        with open("faiss_index/index.pkl", "rb") as f:
            s3.upload_fileobj(f, s3_bucket, s3_key_2, ExtraArgs={'ContentType': 'application/octet-stream'})

        
        # Delete the faiss_index file after successful upload
        if os.path.exists("faiss_index"):
            try:
                os.chmod("faiss_index", 0o777)  # Change permissions to ensure it can be deleted
                os.remove("faiss_index")
                logging.info("FAISS index file deleted after successful upload.")
            except PermissionError as e:
                logging.error(f"Permission denied while deleting FAISS index: {e}")
            logging.info("FAISS index file deleted after successful upload.")
        
        logging.info(f"FAISS index successfully uploaded to s3://{s3_bucket}/{s3_key}")
        return {"message": f"FAISS index successfully uploaded to s3://{s3_bucket}/{s3_key}"}
    except Exception as e:
        logging.error(f"Error uploading FAISS index to S3: {e}")
        raise HTTPException(status_code=500, detail=f"Error uploading FAISS index to S3: {str(e)}")


def load_pdfs_from_directory(directory):
    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
    documents = []
    for pdf_file in pdf_files:
        file_path = os.path.join(directory, pdf_file)
        loader = PyPDFLoader(file_path)
        docs = loader.load()  # Load the document (each page is a document)
        documents.extend(docs)
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = text_splitter.split_documents(documents)
    logging.info(f"Total number of chunks created: {len(chunked_docs)}")
    return chunked_docs

def create_faiss_index(chunked_docs):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index = FAISS.from_documents(chunked_docs, embedding_model)
    faiss_index.save_local("faiss_index")
    logging.info("FAISS index saved locally as 'faiss_index'.")

# Run FastAPI with: uvicorn fastapi_app:app --reload
    

@app.get("/list-s3-topics")
async def list_s3_topics():
    """
    This endpoint lists all folder names in the S3 bucket.
    """
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION")
        )

        s3_bucket = "cmpe-280"
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix='', Delimiter='/')
        folder_names = set()
        if 'CommonPrefixes' in response:
            for obj in response.get('CommonPrefixes', []):
                folder_name = obj['Prefix'].strip('/')
                folder_names.add(folder_name)
        return {"folders": list(folder_names)}
    except Exception as e:
        logging.error(f"Error listing folders in S3: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing folders in S3: {str(e)}")



@app.post("/delete-s3-folder")
async def delete_s3_folder(s3_folder_name: dict):
    """
    This endpoint deletes the specified folder in S3 if it exists.
    """
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION")
        )

        s3_bucket = "cmpe-280"
        prefix = f"{s3_folder_name['s3_folder_name']}/"

        # Check if the folder exists in S3
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
        if 'Contents' not in response or len(response['Contents']) == 0:
            logging.info(f"Folder {s3_folder_name['s3_folder_name']} does not exist in S3.")
            return JSONResponse(status_code=400, content={"message": f"Folder {s3_folder_name['s3_folder_name']} does not exist in S3."})

        # Delete all objects within the folder
        delete_objects = [{'Key': obj['Key']} for obj in response['Contents']]
        s3.delete_objects(Bucket=s3_bucket, Delete={'Objects': delete_objects})
        logging.info(f"Folder {s3_folder_name['s3_folder_name']} and its contents successfully deleted from S3.")
        return {"message": f"Folder {s3_folder_name['s3_folder_name']} successfully deleted from S3."}
    except Exception as e:
        logging.error(f"Error deleting folder in S3: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting folder in S3: {str(e)}")
    

@app.post("/generate-quiz")
async def generate_quiz(request: QuizRequest):
    """
    This endpoint generates quiz questions based on the given topic and subtopics.
    """
    try:
        # Print request data
        print("Received generate-quiz request.")
        print(f"Request data: topic_name={request.topic_name}, subtopics={request.subtopics}")

        # Ensure correct data types
        topic_name = request.topic_name
        subtopics = request.subtopics

        # Print the types to verify correctness
        print(f"type(topic_name): {type(topic_name)}, type(subtopics): {type(subtopics)}")
        if not isinstance(topic_name, str):
            raise ValueError("The topic_name must be a string.")
        
        if not isinstance(subtopics, list) or not all(isinstance(subtopic, str) for subtopic in subtopics):
            raise ValueError("The subtopics must be a list of strings.")

        # Print the subtopics list to verify each type
        print(f"Subtopics verification passed. Subtopics: {subtopics}")

        # Initialize S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION")
        )
        s3_bucket = "cmpe-280"
        s3_index_prefix = f"{topic_name}/"

        # Define local paths
        local_faiss_index_dir = "faiss_index/"

        # Step 1: Ensure the directory is properly cleaned and set up
        if os.path.exists(local_faiss_index_dir):
            print(f"Directory {local_faiss_index_dir} already exists. Removing it.")
            shutil.rmtree(local_faiss_index_dir)  # Remove existing directory and its content
        
        print(f"Creating directory {local_faiss_index_dir}.")
        os.makedirs(local_faiss_index_dir, exist_ok=True)  # Create a new directory

        # Step 2: List and download all parts of the FAISS index from S3
        print(f"Listing objects in S3 with prefix {s3_index_prefix}.")
        response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=s3_index_prefix)

        if 'Contents' not in response or len(response['Contents']) == 0:
            raise FileNotFoundError(f"No FAISS index found at S3 prefix {s3_index_prefix}")

        for obj in response.get('Contents', []):
            s3_key = obj['Key']
            file_name = s3_key.split('/')[-1]
            local_file_path = os.path.join(local_faiss_index_dir, file_name)
            
            # Download each file in the FAISS index folder
            print(f"Downloading {s3_key} to {local_file_path}.")
            s3.download_file(s3_bucket, s3_key, local_file_path)

        # Step 3: Verify directory contents
        print(f"Contents of {local_faiss_index_dir}: {os.listdir(local_faiss_index_dir)}")

        # Step 4: Load FAISS vector store
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Load the FAISS index by pointing to the directory
        print(f"Loading FAISS index from directory {local_faiss_index_dir}.")
        faiss_index = FAISS.load_local(local_faiss_index_dir, embeddings, allow_dangerous_deserialization=True)

        # Step 5: Generate quiz questions
        retriever = faiss_index.as_retriever()
        subtopics_str = ", ".join(subtopics)
        query = f"Retrieve relevant information specifically focusing on the following subtopics: {subtopics_str} within the context of the main topic '{topic_name}'."
        print(f"Running query: {query}")

        relevant_texts = retriever.get_relevant_documents(query)

        # Print details about the retrieved documents
        if relevant_texts:
            print(f"Retrieved {len(relevant_texts)} documents.")
        else:
            print("No documents retrieved.")

        # Compile context from retrieved documents
        context = "\n".join([text.page_content for text in relevant_texts[:5]])
        print("Context successfully retrieved.")

        # Prepare prompt for quiz generation
        # Prepare prompt for quiz generation
        detailed_prompt = (
            "You are an expert quiz question writer for a professional quiz service ready for production. Your task is to create high-quality, knowledge-testing multiple-choice questions based directly on the provided topic and subtopics.\n\n"
            "Requirements for each question:\n\n"
            "- *Clarity and Conciseness:* Each question should be challenging yet clear and concise.\n"
            "- *Answer Choices:* Generate exactly four plausible answer choices for each question.\n"
            "  - Only one choice should be the correct answer.\n"
            "  - The other three should be incorrect yet believable distractions closely related to the subject.\n"
            "- *Relevance:* Ensure all questions and choices are directly based on the provided topic and subtopics.\n"
            "- *Accuracy:* All content must be accurate and up-to-date as of the knowledge cutoff in 2023-10.\n"
            "- *Ambiguity:* Avoid any ambiguity or confusion in both questions and answer choices.\n"
            "- *Professionalism:* Content should be appropriate for a general audience and adhere to professional standards.\n\n"
            "Output Format:\n\n"
            "- *JSON Structure:* Respond strictly in valid JSON format using the following structure only. ***NO EXTRA FIELDS OR TEXT!!!*** just these:\n\n"
            "[\n"
            "  {\n"
            "    \"question\": \"Your question here\",\n"
            "    \"choices\": [\"Option A\", \"Option B\", \"Option C\", \"Option D\"],\n"
            "    \"answer\": \"Correct option from choices\"\n"
            "  },\n"
            "  ...\n"
            "]\n\n"
            "*Important:*\n"
            "- Do not include any additional text, titles, or explanations outside the JSON structure.\n"
            "- Only output a valid JSON array directly as the response.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Topic: {topic_name}\n"
            "Subtopics: {subtopics_str}\n\n"
            "Generate exactly 5 questions following the above structure. "
            "******* VERY IMPORTANT Ensure the response is concise and strictly adheres to JSON formatting and Remember, the total response must not exceed 450 tokens.********"
        )


        # Prepare messages for ChatGroq model invocation
        messages = [
            ("system", "You are a helpful assistant that creates quiz questions based on the provided content."),
            ("user", detailed_prompt)
        ]

        # Invoke the model
        print("Invoking the ChatGroq model to generate quiz questions.")
        llm = ChatGroq(api_key=os.getenv('GROQ_API_KEY'), max_tokens=500)
        response = llm.invoke(messages)

        # Log raw response for debugging
        # print("Raw model response:", response.content)

        try:
            # Parse the raw response
            parsed_response = json.loads(response.content)

            # Validate the structure of the response
            if not isinstance(parsed_response, dict) or "questions" not in parsed_response:
                raise ValueError("Response does not contain a 'questions' key.")
            
            quiz_questions = parsed_response["questions"]
            
            # Ensure each question has the required structure
            for question in quiz_questions:
                if not {"question", "choices", "answer"} <= question.keys():
                    raise ValueError("One or more questions are missing required keys.")
                question["your_answer"] = ""
            
            output_dir = Path("output")
            output_dir.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist
            file_path = output_dir / "quiz_questions.json"

            # Save the quiz questions to a JSON file
            with open(file_path, "w") as f:
                json.dump({"quiz_questions": quiz_questions}, f, indent=4)
            

            

            
            
            # Return the validated questions
            return {"quiz_questions": quiz_questions}

        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format from model: {response.content}")
            raise HTTPException(status_code=500, detail=f"Invalid JSON format from model: {str(e)}")
    except Exception as e:
        logging.error(f"Error validating model response: {response.content}")
        raise HTTPException(status_code=500, detail=f"Error validating model response: {str(e)}")
        

@app.post("/store-quiz")
async def store_quiz(request: QuizRequest, db: Session = Depends(get_db)):
    """
    This endpoint stores the quiz data into the database.
    It gets the quiz questions from a JSON file in the output folder.
    The input consists only of the topic_name and a list of subtopics.
    """
    try:
        # Define file path in the output folder
        print(f"Request Data: {request}")
        file_path = Path("output/quiz_questions.json")
        
        # Check if the file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Quiz questions file not found.")

        # Read the quiz data from the file
        with open(file_path, "r") as f:
            try:
                quiz_data = json.load(f)
                quiz_questions = quiz_data.get("quiz_questions", [])

                # Validate if the quiz data is present and correct
                if not quiz_questions:
                    raise HTTPException(status_code=400, detail="No quiz questions found in the file.")
                

                # Store the quiz data into the database
                quiz = store_quiz_in_db(db, topic_name=request.topic_name, 
                                        subtopics=request.subtopics,  # Pass subtopics from the request
                                        quiz_data=quiz_questions)
                
                

                return {"message": "Quiz successfully stored.", "quiz_id": quiz.quiz_id}

            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from file: {file_path}. Error: {str(e)}")
                raise HTTPException(status_code=500, detail="Error decoding quiz questions from file.")
            except Exception as e:
                logging.error(f"Unexpected error reading or storing quiz: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error storing quiz: {str(e)}")
    
    except Exception as e:
        logging.error(f"Error in store-quiz endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error storing quiz: {str(e)}")

@app.get("/quizzes/{topic_name}", response_model=List[QuizResponse])
def retrieve_quizzes(topic_name: str, db: Session = Depends(get_db)):
    quizzes = get_quiz_by_topic(db, topic_name)
    if not quizzes:
        raise HTTPException(status_code=404, detail="No quizzes found for this topic")
    return quizzes


# Endpoint to delete a quiz by ID
@app.delete("/delete-quiz/{quiz_id}")
async def delete_quiz(quiz_id: int, db: Session = Depends(get_db)):
    try:
        success = delete_quiz_by_id(db, quiz_id)
        if not success:
            raise HTTPException(status_code=404, detail="Quiz not found.")
        return {"message": "Quiz deleted successfully."}
    except Exception as e:
        logging.error(f"Error deleting quiz: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting quiz: {str(e)}")
    
@app.get("/get-all-quizzes")
async def get_all_quizzes(db: Session = Depends(get_db)):
    """
    Endpoint to retrieve all quizzes from the database.
    """
    try:
        quizzes = db.query(Quiz).all()  # Retrieve all quizzes without filtering by topic
        if not quizzes:
            raise HTTPException(status_code=404, detail="No quizzes found.")
        
        # Return the quizzes in the response format
        return [QuizResponse(
            quiz_id=quiz.quiz_id,
            topic_name=quiz.topic_name,
            subtopics=quiz.subtopics,
            created_at=quiz.created_at.isoformat(),
            quiz_data=quiz.quiz_data
        ) for quiz in quizzes]
    except Exception as e:
        logging.error(f"Error retrieving all quizzes: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving quizzes: {str(e)}")

class UpdateAnswersRequest(BaseModel):
    quiz_id: int
    your_answers: List[str]

def read_json(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    return data

# Define a function to save the updated data to the JSON file
def save_json(file_path: str, data: dict):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


@app.post("/update_answers")
async def update_answers(request: UpdateAnswersRequest, db: Session = Depends(get_db)):
    """
    Updates the answers for a quiz identified by `quiz_id`.
    """
    try:
        # Fetch the quiz by ID
        print("request.quiz_id", request.quiz_id)
        quiz = db.query(Quiz).filter(Quiz.quiz_id == request.quiz_id).first()

        if not quiz:
            raise HTTPException(status_code=404, detail="Quiz not found.")

        # Ensure the number of answers matches the number of questions
        quiz_questions = quiz.quiz_data  # The existing quiz questions
        if len(request.your_answers) != len(quiz_questions):
            raise HTTPException(status_code=400, detail="Number of answers provided does not match the number of questions.")

        # Update the 'your_answer' field for each question
        for i, question in enumerate(quiz_questions):
            question['your_answer'] = request.your_answers[i]

        # Serialize quiz_data to JSON string
        serialized_quiz_data = json.dumps(quiz_questions)

        file_path = './output/quiz_questions.json'
        for i, question in enumerate(quiz_questions):
            question['your_answer'] = request.your_answers[i]
    
    # Save the updated data back to the JSON file
        save_json(file_path, {"quiz_questions": quiz_questions})


        
        # save_json(file_path, json.loads(quiz.quiz_data))

        # Update the quiz data in the database using raw SQL
        query = text("""
            UPDATE quizzes
            SET quiz_data = :quiz_data
            WHERE quiz_id = :quiz_id
        """)
        db.execute(query, {"quiz_data": serialized_quiz_data, "quiz_id": request.quiz_id})
        db.commit()

        print("updated quiz questions", quiz_questions)
        return {"message": "Quiz answers updated successfully", "quiz_id": request.quiz_id, "updated_data": quiz_questions}

    except Exception as e:
        logging.error(f"Error updating quiz answers: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while updating the quiz: {str(e)}")
    


# Path to the quiz JSON file
QUIZ_JSON_PATH = "output/quiz_questions.json"

# Image URL for watermark or header
IMAGE_URL = "https://www.sjsu.edu/communications/pics/identity/043014_Primary_Mark_WEB_01.png"

# Function to download the image and return as a BytesIO object
def get_image_as_bytes(url: str) -> BytesIO:
    response = requests.get(url)
    if response.status_code == 200:
        img_bytes = BytesIO(response.content)
        return img_bytes
    else:
        raise Exception(f"Failed to download image from {url}")

@app.get("/download-quiz-pdf")
async def download_quiz_pdf():
    # Check if the quiz JSON file exists
    if not os.path.exists(QUIZ_JSON_PATH):
        return {"error": "Quiz JSON file not found."}

    # Load quiz data from the JSON file
    with open(QUIZ_JSON_PATH, "r") as file:
        quiz_data = json.load(file)

    # Create a PDF in memory
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Get the image as bytes for watermark
    try:
        img_bytes = get_image_as_bytes(IMAGE_URL)
    except Exception as e:
        return {"error": str(e)}

    # Custom canvas to add watermark with reduced opacity
    def add_watermark(canvas, doc):
        # Set the opacity and watermark text
        canvas.saveState()
        canvas.setFont("Helvetica", 50)
        canvas.setFillColorRGB(0.8, 0.8, 0.8, alpha=0.5)  # Semi-transparent gray
        canvas.rotate(45)
        canvas.drawString(100, 500, "WATERMARK")  # Example watermark text
        canvas.restoreState()

    elements = []
    title = Paragraph("Quiz Questions and Answers", styles["Title"])
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Add questions and answers
    for idx, question in enumerate(quiz_data["quiz_questions"], start=1):
        question_text = f"{idx}. {question['question']}"
        elements.append(Paragraph(question_text, styles["Heading2"]))
        elements.append(Spacer(1, 6))

        # Add choices
        for choice in question['choices']:
            elements.append(Paragraph(f" - {choice}", styles["Normal"]))
        elements.append(Spacer(1, 6))

        # Add correct answer and user's answer
        elements.append(Paragraph(f"Correct Answer: {question['answer']}", styles["Normal"]))
        elements.append(Paragraph(f"Your Answer: {question['your_answer']}", styles["Normal"]))
        elements.append(Spacer(1, 12))

    # Build the PDF using a custom canvas for watermark
    doc.build(elements, onFirstPage=add_watermark, onLaterPages=add_watermark)

    # Serve the PDF as a downloadable file
    buffer.seek(0)
    return Response(buffer.read(), media_type="application/pdf", headers={
        "Content-Disposition": "attachment;filename=quiz_questions.pdf"
    })