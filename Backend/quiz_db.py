from sqlalchemy import create_engine, Column, Integer, String, JSON, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import Session
import datetime
from sqlalchemy import inspect

# Database URL (Make sure it's a persistent SQLite file)
DATABASE_URL = "sqlite:///./quiz_database.db"  # Updated for persistence in a file

# Initialize the base for SQLAlchemy
Base = declarative_base()

# Define the Quiz model with subtopics
class Quiz(Base):
    __tablename__ = 'quizzes'
    quiz_id = Column(Integer, primary_key=True, autoincrement=True)
    topic_name = Column(String(255), nullable=False)
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)
    quiz_data = Column(JSON, nullable=False)
    subtopics = Column(JSON, nullable=False)  # Added subtopics as a list of strings

# Create the SQLite engine (using a persistent database)
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create session local for interacting with the DB
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to create tables (if they don't exist)
def create_tables():
    inspector = inspect(engine)
    # Check if the 'quizzes' table exists
    if not inspector.has_table("quizzes"):
        print("Tables not found, creating them.")
        Base.metadata.create_all(bind=engine)
    else:
        print("Tables already exist. Skipping creation.")

# Create tables
create_tables()

# Function to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Function to store quiz in the database
def store_quiz_in_db(db: Session, topic_name: str, subtopics: list, quiz_data: dict):
    """
    Store the quiz data into the database
    """
    new_quiz = Quiz(
        topic_name=topic_name,
        subtopics=subtopics,  # Store the subtopics list
        quiz_data=quiz_data
    )
    db.add(new_quiz)
    db.commit()
    db.refresh(new_quiz)
    return new_quiz

# Function to retrieve quizzes by topic
def get_quiz_by_topic(db: Session, topic_name: str):
    """
    Retrieve quizzes by topic name from the database and format response
    """
    # Query the quizzes from the database
    quizzes = db.query(Quiz).filter(Quiz.topic_name == topic_name).all()

    # Format the response to match the expected structure
    formatted_quizzes = []
    for quiz in quizzes:
        formatted_quiz = {
            "quiz_id": quiz.quiz_id,
            "topic_name": quiz.topic_name,
            "created_at": quiz.created_at.isoformat(),  # Convert to ISO string
            "quiz_data": quiz.quiz_data,  # Assuming quiz_data is already in the correct format
            "subtopics": quiz.subtopics,
        }
        formatted_quizzes.append(formatted_quiz)
    print(formatted_quizzes)
    return formatted_quizzes

# Function to retrieve quizzes by subtopic
def get_quiz_by_subtopic(db: Session, subtopic: str):
    """
    Retrieve quizzes by a specific subtopic from the database
    """
    return db.query(Quiz).filter(Quiz.subtopics.any(subtopic)).all()

# Function to delete a quiz by ID
def delete_quiz_by_id(db: Session, quiz_id: int):
    """
    Delete a quiz by its ID from the database
    """
    quiz = db.query(Quiz).filter(Quiz.quiz_id == quiz_id).first()
    print(f"deleting quiz {quiz_id} now......")
    if quiz:
        db.delete(quiz)
        db.commit()
        return True
    return False

# New function to retrieve all quizzes
def get_all_quizzes(db: Session):
    """
    Retrieve all quizzes from the database and format the response.
    """
    quizzes = db.query(Quiz).all()

    # Format the response to match the expected structure
    formatted_quizzes = []
    for quiz in quizzes:
        formatted_quiz = {
            "quiz_id": quiz.quiz_id,
            "topic_name": quiz.topic_name,
            "created_at": quiz.created_at.isoformat(),  # Convert to ISO string
            "quiz_data": quiz.quiz_data,  # Assuming quiz_data is already in the correct format
            "subtopics": quiz.subtopics,
        }
        formatted_quizzes.append(formatted_quiz)
        

    return formatted_quizzes
