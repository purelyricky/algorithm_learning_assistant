"""
Document Manager for Algorithm Learning Assistant
Manages document lifecycle, storage, and integration with vector database
"""

import os
import json
from datetime import datetime
from pathlib import Path
from memory_system import connect_db
from document_processor import process_document, get_supported_file_types

def initialize_document_tables():
    """
    Creates document-related tables in PostgreSQL
    Extends existing database schema without affecting current tables
    """
    conn = connect_db()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cursor:
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
            print("Document management tables created successfully!")
            return True
            
    except Exception as e:
        print(f"Error creating document tables: {e}")
        return False
    finally:
        conn.close()

def check_duplicate_document(file_hash):
    """
    Checks if document with same hash already exists
    Args: file_hash - SHA256 hash of the file
    Returns: existing document_id if duplicate found, None otherwise
    """
    conn = connect_db()
    if not conn:
        return None
        
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT document_id, filename FROM uploaded_documents 
                WHERE file_hash = %s
            """, (file_hash,))
            
            result = cursor.fetchone()
            return result[0] if result else None
            
    except Exception as e:
        print(f"Error checking for duplicates: {e}")
        return None
    finally:
        conn.close()

def store_document_metadata(document_data):
    """
    Stores document metadata in PostgreSQL
    Args: document_data - processed document data from document_processor
    Returns: Boolean indicating success
    """
    conn = connect_db()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cursor:
            # Check for duplicates
            existing_doc = check_duplicate_document(document_data["file_hash"])
            if existing_doc:
                print(f"Document already exists with ID: {existing_doc}")
                return False
            
            # Insert document metadata
            cursor.execute("""
                INSERT INTO uploaded_documents (
                    document_id, filename, file_path, file_hash, file_size,
                    mime_type, file_extension, processing_status, content_summary,
                    extraction_metadata, total_chunks, total_characters
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                document_data["document_id"],
                document_data["filename"],
                document_data["file_path"],
                document_data["file_hash"],
                document_data["file_size"],
                document_data["mime_type"],
                document_data["file_extension"],
                "processed",
                document_data["content_preview"],
                json.dumps(document_data["extraction_metadata"]),
                document_data["total_chunks"],
                document_data["total_characters"]
            ))
            
            # Store chunk metadata
            for chunk in document_data["chunks"]:
                cursor.execute("""
                    INSERT INTO document_chunks (
                        document_id, chunk_index, content, char_count
                    ) VALUES (%s, %s, %s, %s)
                """, (
                    document_data["document_id"],
                    chunk["chunk_index"],
                    chunk["content"],
                    chunk["char_count"]
                ))
            
            # Auto-tag documents based on content
            auto_tags = generate_auto_tags(document_data)
            for tag in auto_tags:
                cursor.execute("""
                    INSERT INTO document_tags (document_id, tag, tag_type, confidence)
                    VALUES (%s, %s, %s, %s)
                """, (document_data["document_id"], tag["tag"], "auto", tag["confidence"]))
            
            conn.commit()
            return True
            
    except Exception as e:
        print(f"Error storing document metadata: {e}")
        return False
    finally:
        conn.close()

def generate_auto_tags(document_data):
    """
    Generates automatic tags based on document content and metadata
    Args: document_data - processed document data
    Returns: list of tag dictionaries
    """
    tags = []
    content = document_data.get("content_preview", "").lower()
    filename = document_data.get("filename", "").lower()
    file_ext = document_data.get("file_extension", "")
    
    # Algorithm-related tags
    algorithm_keywords = {
        "sorting": ["sort", "quicksort", "mergesort", "bubble", "insertion", "selection"],
        "searching": ["search", "binary search", "linear search", "find"],
        "data_structures": ["tree", "graph", "list", "stack", "queue", "hash"],
        "algorithms": ["algorithm", "complexity", "big o", "time complexity"],
        "programming": ["code", "implementation", "function", "class"]
    }
    
    for category, keywords in algorithm_keywords.items():
        for keyword in keywords:
            if keyword in content or keyword in filename:
                tags.append({
                    "tag": category,
                    "confidence": 0.8 if keyword in content else 0.6
                })
                break  # One tag per category
    
    # File type tags
    type_tags = {
        ".pdf": "textbook",
        ".docx": "notes", 
        ".py": "code_example",
        ".java": "code_example",
        ".txt": "documentation",
        ".md": "documentation"
    }
    
    if file_ext in type_tags:
        tags.append({
            "tag": type_tags[file_ext],
            "confidence": 1.0
        })
    
    # Course-related tags (if patterns found)
    course_patterns = {
        "cs": ["computer science", "cs ", "cs-"],
        "algorithms": ["algorithm", "comp sci", "data structure"],
        "programming": ["programming", "coding", "software"]
    }
    
    for course, patterns in course_patterns.items():
        for pattern in patterns:
            if pattern in content or pattern in filename:
                tags.append({
                    "tag": f"course_{course}",
                    "confidence": 0.7
                })
                break
    
    return tags

def get_user_documents(limit=None):
    """
    Retrieves list of uploaded documents
    Args: limit - maximum number of documents to return
    Returns: list of document dictionaries
    """
    conn = connect_db()
    if not conn:
        return []
        
    try:
        with conn.cursor() as cursor:
            query = """
                SELECT document_id, filename, file_extension, upload_date, 
                       total_chunks, total_characters, access_count, 
                       content_summary
                FROM uploaded_documents 
                ORDER BY upload_date DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query)
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    "document_id": row[0],
                    "filename": row[1],
                    "file_extension": row[2],
                    "upload_date": row[3].strftime("%Y-%m-%d %H:%M") if row[3] else "Unknown",
                    "total_chunks": row[4],
                    "total_characters": row[5],
                    "access_count": row[6],
                    "content_summary": row[7]
                })
            
            return documents
            
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []
    finally:
        conn.close()

def get_document_by_id(document_id):
    """
    Retrieves detailed document information by ID
    Args: document_id - unique document identifier
    Returns: document dictionary or None
    """
    conn = connect_db()
    if not conn:
        return None
        
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT document_id, filename, file_path, file_size, mime_type,
                       file_extension, upload_date, content_summary, 
                       extraction_metadata, total_chunks, total_characters
                FROM uploaded_documents 
                WHERE document_id = %s
            """, (document_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Get tags
            cursor.execute("""
                SELECT tag, tag_type, confidence 
                FROM document_tags 
                WHERE document_id = %s
            """, (document_id,))
            
            tags = [{"tag": tag[0], "type": tag[1], "confidence": tag[2]} 
                   for tag in cursor.fetchall()]
            
            return {
                "document_id": row[0],
                "filename": row[1],
                "file_path": row[2],
                "file_size": row[3],
                "mime_type": row[4],
                "file_extension": row[5],
                "upload_date": row[6].strftime("%Y-%m-%d %H:%M") if row[6] else "Unknown",
                "content_summary": row[7],
                "extraction_metadata": json.loads(row[8]) if row[8] else {},
                "total_chunks": row[9],
                "total_characters": row[10],
                "tags": tags
            }
            
    except Exception as e:
        print(f"Error retrieving document: {e}")
        return None
    finally:
        conn.close()

def search_documents(query, limit=10):
    """
    Searches documents by filename, content summary, or tags
    Args: query - search query, limit - max results
    Returns: list of matching document dictionaries
    """
    conn = connect_db()
    if not conn:
        return []
        
    try:
        with conn.cursor() as cursor:
            # Search in filename, content summary, and tags
            cursor.execute("""
                SELECT DISTINCT d.document_id, d.filename, d.file_extension, 
                       d.upload_date, d.content_summary, d.total_chunks
                FROM uploaded_documents d
                LEFT JOIN document_tags t ON d.document_id = t.document_id
                WHERE d.filename ILIKE %s 
                   OR d.content_summary ILIKE %s 
                   OR t.tag ILIKE %s
                ORDER BY d.upload_date DESC
                LIMIT %s
            """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    "document_id": row[0],
                    "filename": row[1],
                    "file_extension": row[2],
                    "upload_date": row[3].strftime("%Y-%m-%d %H:%M") if row[3] else "Unknown",
                    "content_summary": row[4],
                    "total_chunks": row[5]
                })
            
            return documents
            
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []
    finally:
        conn.close()

def record_document_usage(document_id, query_text, usage_type="search", relevance_score=0.0):
    """
    Records when a document is used in queries
    Args: document_id, query_text, usage_type, relevance_score
    """
    conn = connect_db()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cursor:
            # Record usage
            cursor.execute("""
                INSERT INTO document_usage (document_id, query_text, usage_type, relevance_score)
                VALUES (%s, %s, %s, %s)
            """, (document_id, query_text, usage_type, relevance_score))
            
            # Update access count
            cursor.execute("""
                UPDATE uploaded_documents 
                SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                WHERE document_id = %s
            """, (document_id,))
            
            conn.commit()
            return True
            
    except Exception as e:
        print(f"Error recording document usage: {e}")
        return False
    finally:
        conn.close()

def delete_document(document_id):
    """
    Deletes document and all related data
    Args: document_id - unique document identifier
    Returns: Boolean indicating success
    """
    conn = connect_db()
    if not conn:
        return False
        
    try:
        with conn.cursor() as cursor:
            # Get document info for cleanup
            cursor.execute("""
                SELECT filename FROM uploaded_documents WHERE document_id = %s
            """, (document_id,))
            
            result = cursor.fetchone()
            if not result:
                print("Document not found")
                return False
            
            filename = result[0]
            
            # Delete from database (cascading deletes will handle related tables)
            cursor.execute("""
                DELETE FROM uploaded_documents WHERE document_id = %s
            """, (document_id,))
            
            conn.commit()
            print(f"Document '{filename}' deleted successfully")
            return True
            
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False
    finally:
        conn.close()

def get_document_statistics():
    """
    Gets overall document statistics
    Returns: statistics dictionary
    """
    conn = connect_db()
    if not conn:
        return {}
        
    try:
        with conn.cursor() as cursor:
            # Basic stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_documents,
                    SUM(file_size) as total_size,
                    SUM(total_chunks) as total_chunks,
                    SUM(total_characters) as total_characters,
                    AVG(access_count) as avg_access_count
                FROM uploaded_documents
            """)
            
            basic_stats = cursor.fetchone()
            
            # File type distribution
            cursor.execute("""
                SELECT file_extension, COUNT(*) as count
                FROM uploaded_documents
                GROUP BY file_extension
                ORDER BY count DESC
            """)
            
            file_types = dict(cursor.fetchall())
            
            # Most accessed documents
            cursor.execute("""
                SELECT filename, access_count
                FROM uploaded_documents
                ORDER BY access_count DESC
                LIMIT 5
            """)
            
            most_accessed = [{"filename": row[0], "access_count": row[1]} 
                           for row in cursor.fetchall()]
            
            return {
                "total_documents": basic_stats[0] or 0,
                "total_size_bytes": basic_stats[1] or 0,
                "total_chunks": basic_stats[2] or 0,
                "total_characters": basic_stats[3] or 0,
                "avg_access_count": round(basic_stats[4], 2) if basic_stats[4] else 0,
                "file_type_distribution": file_types,
                "most_accessed": most_accessed
            }
            
    except Exception as e:
        print(f"Error getting document statistics: {e}")
        return {}
    finally:
        conn.close()

def cleanup_old_documents(days=30):
    """
    Cleanup documents not accessed in specified days
    Args: days - number of days for cleanup threshold
    Returns: number of documents cleaned up
    """
    conn = connect_db()
    if not conn:
        return 0
        
    try:
        with conn.cursor() as cursor:
            # Find old documents
            cursor.execute("""
                SELECT document_id, filename FROM uploaded_documents
                WHERE last_accessed < NOW() - INTERVAL '%s days'
                AND access_count = 0
            """, (days,))
            
            old_docs = cursor.fetchall()
            cleanup_count = 0
            
            for doc_id, filename in old_docs:
                if delete_document(doc_id):
                    cleanup_count += 1
                    print(f"Cleaned up unused document: {filename}")
            
            return cleanup_count
            
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return 0
    finally:
        conn.close()

if __name__ == "__main__":
    # Test document management
    if initialize_document_tables():
        print("Document management system initialized!")
        
        # Show supported file types
        supported = get_supported_file_types()
        print("\nSupported file types:")
        for ext, desc in supported.items():
            print(f"  .{ext}: {desc}")
        
        # Show document statistics
        stats = get_document_statistics()
        print(f"\nDocument Statistics: {stats}")
    else:
        print("Failed to initialize document management system!")