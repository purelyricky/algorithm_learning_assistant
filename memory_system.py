"""
Memory System for Algorithm Learning Assistant
Handles PostgreSQL conversation storage and student progress tracking
"""

import psycopg2
import json
from datetime import datetime
from collections import defaultdict

# Database connection parameters - modify these for your setup
DB_PARAMS = {
    "host": "localhost",
    "database": "algorithm_tutor",
    "user": "cs_student",
    "password": "demo_user_password"  # Change this to your actual password
}

def connect_db():
    """
    Establishes connection to PostgreSQL database
    Returns: database connection object
    """
    try:
        conn = psycopg2.connect(**DB_PARAMS)
        return conn
    except psycopg2.Error as e:
        print(f"Database connection error: {e}")
        return None

def initialize_database():
    """
    Creates necessary tables if they don't exist
    Now includes document management tables
    """
    conn = connect_db()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cursor:
            # Conversations table - stores all user interactions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_input TEXT NOT NULL,
                    assistant_response TEXT NOT NULL,
                    command_type VARCHAR(50),
                    algorithm_topic VARCHAR(100),
                    difficulty VARCHAR(20)
                )
            """)
            
            # Student progress table - tracks learning progress
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS student_progress (
                    id SERIAL PRIMARY KEY,
                    algorithm VARCHAR(100) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    times_studied INTEGER DEFAULT 1,
                    last_studied TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    difficulty_level VARCHAR(20) DEFAULT 'beginner',
                    mastery_score FLOAT DEFAULT 0.0,
                    UNIQUE(algorithm, category)
                )
            """)
            
            # Practice attempts table - tracks problem solving
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS practice_attempts (
                    id SERIAL PRIMARY KEY,
                    problem_id INTEGER NOT NULL,
                    algorithm VARCHAR(100),
                    difficulty VARCHAR(20),
                    attempted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    success BOOLEAN DEFAULT FALSE
                )
            """)
            
            # EXTENDED: Document management tables
            # Document registry table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS uploaded_documents (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) UNIQUE NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    file_path TEXT,
                    file_hash VARCHAR(64) UNIQUE,
                    file_size BIGINT,
                    mime_type VARCHAR(100),
                    file_extension VARCHAR(10),
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status VARCHAR(50) DEFAULT 'pending',
                    content_summary TEXT,
                    extraction_metadata JSONB,
                    total_chunks INTEGER DEFAULT 0,
                    total_characters INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            """)
            
            # Document chunks table for tracking vector storage
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) REFERENCES uploaded_documents(document_id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    char_count INTEGER,
                    vector_id VARCHAR(100),
                    collection_name VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Document usage tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_usage (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) REFERENCES uploaded_documents(document_id) ON DELETE CASCADE,
                    query_text TEXT,
                    usage_type VARCHAR(50),
                    used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    relevance_score FLOAT
                )
            """)
            
            # Document tags for categorization
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_tags (
                    id SERIAL PRIMARY KEY,
                    document_id VARCHAR(255) REFERENCES uploaded_documents(document_id) ON DELETE CASCADE,
                    tag VARCHAR(100),
                    tag_type VARCHAR(50) DEFAULT 'user',
                    confidence FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            print("Database tables created successfully!")
            print("- Core learning tables: conversations, student_progress, practice_attempts")
            print("- Extended document tables: uploaded_documents, document_chunks, document_usage, document_tags")
            return True
            
    except psycopg2.Error as e:
        print(f"Database initialization error: {e}")
        return False
    finally:
        conn.close()

def store_conversation(user_input, assistant_response, command_type=None, algorithm_topic=None, difficulty=None):
    """
    Stores conversation in database
    Args:
        user_input: User's question/command
        assistant_response: AI assistant's response  
        command_type: Type of command used (/explain, /practice, etc.)
        algorithm_topic: Algorithm discussed (if applicable)
        difficulty: Difficulty level (if applicable)
    """
    conn = connect_db()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO conversations (user_input, assistant_response, command_type, algorithm_topic, difficulty)
                VALUES (%s, %s, %s, %s, %s)
            """, (user_input, assistant_response, command_type, algorithm_topic, difficulty))
            
            conn.commit()
            return True
            
    except psycopg2.Error as e:
        print(f"Error storing conversation: {e}")
        return False
    finally:
        conn.close()

def fetch_conversation_history(limit=10):
    """
    Retrieves recent conversation history
    Args: limit - number of recent conversations to retrieve
    Returns: list of conversation dictionaries
    """
    conn = connect_db()
    if not conn:
        return []
        
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT timestamp, user_input, assistant_response, command_type, algorithm_topic
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            
            conversations = []
            for row in cursor.fetchall():
                conversations.append({
                    'timestamp': row[0],
                    'user_input': row[1],
                    'assistant_response': row[2],
                    'command_type': row[3],
                    'algorithm_topic': row[4]
                })
            
            return conversations
            
    except psycopg2.Error as e:
        print(f"Error fetching conversations: {e}")
        return []
    finally:
        conn.close()

def update_student_progress(algorithm, category, difficulty='beginner'):
    """
    Updates or creates student progress record for an algorithm
    Args:
        algorithm: Name of algorithm studied
        category: Category (sorting, searching, etc.)  
        difficulty: Current difficulty level
    """
    conn = connect_db()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cursor:
            # Check if record exists
            cursor.execute("""
                SELECT id, times_studied, mastery_score FROM student_progress
                WHERE algorithm = %s AND category = %s
            """, (algorithm, category))
            
            result = cursor.fetchone()
            
            if result:
                # Update existing record
                new_times = result[1] + 1
                new_mastery = min(result[2] + 0.1, 1.0)  # Increase mastery score
                
                cursor.execute("""
                    UPDATE student_progress
                    SET times_studied = %s, last_studied = CURRENT_TIMESTAMP, 
                        difficulty_level = %s, mastery_score = %s
                    WHERE algorithm = %s AND category = %s
                """, (new_times, difficulty, new_mastery, algorithm, category))
            else:
                # Create new record
                cursor.execute("""
                    INSERT INTO student_progress (algorithm, category, difficulty_level, mastery_score)
                    VALUES (%s, %s, %s, %s)
                """, (algorithm, category, difficulty, 0.1))
            
            conn.commit()
            return True
            
    except psycopg2.Error as e:
        print(f"Error updating progress: {e}")
        return False
    finally:
        conn.close()

def record_practice_attempt(problem_id, algorithm, difficulty, success=False):
    """
    Records a practice problem attempt
    Args:
        problem_id: ID of practice problem
        algorithm: Algorithm the problem relates to
        difficulty: Problem difficulty level
        success: Whether attempt was successful
    """
    conn = connect_db()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO practice_attempts (problem_id, algorithm, difficulty, success)
                VALUES (%s, %s, %s, %s)
            """, (problem_id, algorithm, difficulty, success))
            
            conn.commit()
            return True
            
    except psycopg2.Error as e:
        print(f"Error recording practice attempt: {e}")
        return False
    finally:
        conn.close()

def get_student_progress_summary():
    """
    Gets summary of student's learning progress
    Returns: dictionary with progress statistics
    """
    conn = connect_db()
    if not conn:
        return {}
        
    try:
        with conn.cursor() as cursor:
            # Get overall progress
            cursor.execute("""
                SELECT 
                    COUNT(*) as algorithms_studied,
                    AVG(mastery_score) as avg_mastery,
                    SUM(times_studied) as total_sessions
                FROM student_progress
            """)
            
            overall = cursor.fetchone()
            
            # Get progress by category
            cursor.execute("""
                SELECT category, COUNT(*) as count, AVG(mastery_score) as avg_mastery
                FROM student_progress
                GROUP BY category
                ORDER BY avg_mastery DESC
            """)
            
            categories = cursor.fetchall()
            
            # Get recent activity
            cursor.execute("""
                SELECT algorithm, last_studied, mastery_score
                FROM student_progress
                ORDER BY last_studied DESC
                LIMIT 5
            """)
            
            recent = cursor.fetchall()
            
            # Get practice statistics
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_attempts,
                    COUNT(CASE WHEN success THEN 1 END) as successful_attempts,
                    COUNT(DISTINCT algorithm) as algorithms_practiced
                FROM practice_attempts
            """)
            
            practice_stats = cursor.fetchone()
            
            return {
                'total_algorithms_studied': overall[0] if overall[0] else 0,
                'average_mastery': round(overall[1], 2) if overall[1] else 0.0,
                'total_study_sessions': overall[2] if overall[2] else 0,
                'categories': [
                    {
                        'category': cat[0],
                        'algorithms_count': cat[1], 
                        'avg_mastery': round(cat[2], 2)
                    } for cat in categories
                ],
                'recent_activity': [
                    {
                        'algorithm': rec[0],
                        'last_studied': rec[1].strftime('%Y-%m-%d %H:%M') if rec[1] else 'Unknown',
                        'mastery': round(rec[2], 2)
                    } for rec in recent
                ],
                'practice_attempts': practice_stats[0] if practice_stats[0] else 0,
                'successful_attempts': practice_stats[1] if practice_stats[1] else 0,
                'algorithms_practiced': practice_stats[2] if practice_stats[2] else 0
            }
            
    except psycopg2.Error as e:
        print(f"Error getting progress summary: {e}")
        return {}
    finally:
        conn.close()

def get_recommended_algorithms():
    """
    Gets algorithm recommendations based on student's progress
    Returns: list of recommended algorithms to study next
    """
    conn = connect_db()
    if not conn:
        return []
        
    try:
        with conn.cursor() as cursor:
            # Get algorithms with low mastery scores (need more practice)
            cursor.execute("""
                SELECT algorithm, category, mastery_score
                FROM student_progress
                WHERE mastery_score < 0.7
                ORDER BY mastery_score ASC, last_studied ASC
                LIMIT 3
            """)
            
            needs_practice = cursor.fetchall()
            
            # Get algorithms not yet studied (from a predefined list)
            known_algorithms = ['quicksort', 'mergesort', 'binary_search', 'linear_search', 
                              'bubble_sort', 'insertion_sort', 'selection_sort', 'heapsort',
                              'binary_tree', 'hash_table', 'linked_list']
            
            cursor.execute("""
                SELECT algorithm FROM student_progress
            """)
            
            studied = [row[0] for row in cursor.fetchall()]
            not_studied = [alg for alg in known_algorithms if alg not in studied]
            
            recommendations = []
            
            # Add algorithms that need more practice
            for alg in needs_practice:
                recommendations.append({
                    'algorithm': alg[0],
                    'category': alg[1],
                    'reason': f'Low mastery score: {alg[2]:.2f}',
                    'priority': 'high'
                })
            
            # Add new algorithms to learn
            for alg in not_studied[:2]:  # Limit to 2 new algorithms
                recommendations.append({
                    'algorithm': alg,
                    'category': 'unknown',  # Would need to look up
                    'reason': 'Not yet studied',
                    'priority': 'medium'
                })
            
            return recommendations
            
    except psycopg2.Error as e:
        print(f"Error getting recommendations: {e}")
        return []
    finally:
        conn.close()

def remove_last_conversation():
    """
    Removes the most recent conversation from database
    Used for /forget command
    """
    conn = connect_db()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                DELETE FROM conversations
                WHERE id = (SELECT MAX(id) FROM conversations)
            """)
            
            conn.commit()
            return True
            
    except psycopg2.Error as e:
        print(f"Error removing conversation: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    # Test database initialization
    if initialize_database():
        print("Database initialized successfully!")
        
        # Test progress tracking
        update_student_progress("quicksort", "sorting", "intermediate")
        update_student_progress("binary_search", "searching", "beginner")
        
        # Test progress retrieval
        progress = get_student_progress_summary()
        print(f"Progress summary: {progress}")
        
        # Test recommendations
        recommendations = get_recommended_algorithms()
        print(f"Recommendations: {recommendations}")
    else:
        print("Database initialization failed!")