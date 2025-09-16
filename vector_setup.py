"""
Vector Database Setup for Algorithm Learning Assistant
Handles ChromaDB initialization and document embedding using Ollama
Now supports multi-collection architecture for static knowledge + user documents
"""

import json
import os
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

def load_algorithm_knowledge():
    """
    Loads algorithm knowledge from JSON files
    Returns: tuple (algorithms_dict, problems_dict)
    """
    try:
        with open('data/algorithms.json', 'r') as f:
            algorithms = json.load(f)
        
        with open('data/problems.json', 'r') as f:
            problems = json.load(f)
            
        return algorithms, problems
    except FileNotFoundError:
        print("Algorithm data files not found. Please run algorithm_data.py first.")
        return None, None

def create_algorithm_documents(algorithms_data):
    """
    Converts algorithm JSON data into LangChain Document objects
    Args: algorithms_data - dictionary of algorithm information
    Returns: list of Document objects
    """
    documents = []
    doc_id = 0
    
    # Process each category (sorting, searching, data_structures)
    for category, category_algorithms in algorithms_data.items():
        for algorithm_name, algorithm_info in category_algorithms.items():
            
            # Main algorithm explanation document
            content = f"""
Algorithm: {algorithm_info['name']}
Category: {algorithm_info['category']}
Difficulty: {algorithm_info['difficulty']}

Description: {algorithm_info['description']}

Time Complexity:
- Best Case: {algorithm_info.get('time_complexity', {}).get('best', 'N/A')}
- Average Case: {algorithm_info.get('time_complexity', {}).get('average', 'N/A')}
- Worst Case: {algorithm_info.get('time_complexity', {}).get('worst', 'N/A')}

Space Complexity: {algorithm_info.get('space_complexity', 'N/A')}

How it works:
{chr(10).join(f"- {step}" for step in algorithm_info.get('how_it_works', []))}

Advantages:
{chr(10).join(f"- {adv}" for adv in algorithm_info.get('advantages', []))}

Disadvantages:
{chr(10).join(f"- {dis}" for dis in algorithm_info.get('disadvantages', []))}
            """
            
            # Create main document
            doc = Document(
                page_content=content.strip(),
                metadata={
                    "type": "algorithm_explanation",
                    "algorithm": algorithm_name,
                    "category": category,
                    "difficulty": algorithm_info['difficulty'],
                    "id": str(doc_id)
                }
            )
            documents.append(doc)
            doc_id += 1
            
            # Add pseudocode as separate document if available
            if 'pseudocode' in algorithm_info:
                pseudocode_content = f"""
Algorithm: {algorithm_info['name']} - Pseudocode

{algorithm_info['pseudocode']}
                """
                
                doc = Document(
                    page_content=pseudocode_content.strip(),
                    metadata={
                        "type": "pseudocode",
                        "algorithm": algorithm_name,
                        "category": category,
                        "id": str(doc_id)
                    }
                )
                documents.append(doc)
                doc_id += 1
            
            # Add code implementations as separate documents
            if 'implementations' in algorithm_info:
                for language, code in algorithm_info['implementations'].items():
                    code_content = f"""
Algorithm: {algorithm_info['name']} - {language.title()} Implementation

{code}
                    """
                    
                    doc = Document(
                        page_content=code_content.strip(),
                        metadata={
                            "type": "code_implementation",
                            "algorithm": algorithm_name,
                            "category": category,
                            "language": language,
                            "id": str(doc_id)
                        }
                    )
                    documents.append(doc)
                    doc_id += 1
    
    return documents

def create_problem_documents(problems_data):
    """
    Converts practice problems into Document objects
    Args: problems_data - dictionary of practice problems
    Returns: list of Document objects
    """
    documents = []
    
    for category, problems_list in problems_data.items():
        for problem in problems_list:
            content = f"""
Practice Problem - {problem['topic'].title()}
Difficulty: {problem['difficulty']}
Category: {category}

Problem: {problem['problem']}

Hint: {problem['hint']}
            """
            
            doc = Document(
                page_content=content.strip(),
                metadata={
                    "type": "practice_problem",
                    "category": category,
                    "topic": problem['topic'],
                    "difficulty": problem['difficulty'],
                    "problem_id": problem['id'],
                    "id": str(problem['id'] + 1000)  # Offset to avoid conflicts
                }
            )
            documents.append(doc)
    
    return documents

def setup_vector_database():
    """
    Sets up ChromaDB vector database with algorithm knowledge
    Now supports multi-collection architecture
    Returns: tuple (static_retriever, user_retriever_function)
    """
    # Database location
    db_location = "./chroma_db"
    
    # Initialize embeddings model
    print("Initializing embedding model...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Setup static algorithm knowledge collection
    static_retriever = setup_static_knowledge_collection(embeddings, db_location)
    
    # Setup user documents collection (empty initially)
    user_retriever_func = create_user_documents_collection_handler(embeddings, db_location)
    
    return static_retriever, user_retriever_func

def setup_static_knowledge_collection(embeddings, db_location):
    """
    Sets up the static algorithm knowledge collection (backwards compatible)
    Args: embeddings - embedding model, db_location - database path
    Returns: retriever for static knowledge
    """
    collection_name = "static_algorithms"
    collection_path = os.path.join(db_location, collection_name)
    
    # Check if static collection already exists
    add_documents = not os.path.exists(collection_path)
    
    if add_documents:
        print("Creating static algorithm knowledge collection...")
        
        # Load knowledge base
        algorithms_data, problems_data = load_algorithm_knowledge()
        if algorithms_data is None:
            return None
        
        # Create documents (same as before)
        print("Processing algorithm documents...")
        algorithm_docs = create_algorithm_documents(algorithms_data)
        
        print("Processing practice problem documents...")
        problem_docs = create_problem_documents(problems_data)
        
        # Combine all documents
        all_documents = algorithm_docs + problem_docs
        print(f"Created {len(all_documents)} documents total")
        
        # Create vector store and add documents
        print("Embedding documents (this may take a few minutes)...")
        vector_store = Chroma.from_documents(
            documents=all_documents,
            embedding=embeddings,
            collection_name=collection_name,
            persist_directory=db_location
        )
        print("Static knowledge collection created successfully!")
        
    else:
        print("Loading existing static knowledge collection...")
        vector_store = Chroma(
            collection_name=collection_name, 
            embedding_function=embeddings,
            persist_directory=db_location
        )
    
    # Create retriever with multiple search results
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}  # Retrieve top 5 most relevant documents
    )
    
    return retriever

def create_user_documents_collection_handler(embeddings, db_location):
    """
    Creates a function to handle user document collection operations
    Args: embeddings - embedding model, db_location - database path
    Returns: function to get user document retriever
    """
    def get_user_documents_retriever():
        """
        Gets retriever for user documents collection
        Returns: retriever if documents exist, None otherwise
        """
        collection_name = "user_documents"
        collection_path = os.path.join(db_location, collection_name)
        
        if not os.path.exists(collection_path):
            # No user documents uploaded yet
            return None
        
        try:
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=db_location
            )
            
            # Check if collection has any documents
            collection = vector_store._collection
            if collection.count() == 0:
                return None
            
            return vector_store.as_retriever(search_kwargs={"k": 5})
            
        except Exception as e:
            print(f"Error accessing user documents collection: {e}")
            return None
    
    return get_user_documents_retriever

def add_document_to_user_collection(document_data, embeddings, db_location):
    """
    Adds processed document to user documents collection
    Args: document_data - processed document from document_processor
          embeddings - embedding model, db_location - database path
    Returns: Boolean indicating success
    """
    collection_name = "user_documents"
    
    try:
        # Create documents from chunks
        documents = []
        
        for chunk in document_data["chunks"]:
            # Enhanced metadata for user documents
            metadata = {
                "document_id": document_data["document_id"],
                "filename": document_data["filename"],
                "file_extension": document_data["file_extension"],
                "chunk_index": chunk["chunk_index"],
                "total_chunks": document_data["total_chunks"],
                "upload_date": document_data["processed_at"],
                "type": "user_document",
                "source": "user_upload"
            }
            
            # Add extraction metadata
            metadata.update(chunk["metadata"])
            
            doc = Document(
                page_content=chunk["content"],
                metadata=metadata
            )
            documents.append(doc)
        
        # Check if collection exists
        collection_path = os.path.join(db_location, collection_name)
        collection_exists = os.path.exists(collection_path)
        
        if collection_exists:
            # Add to existing collection
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=db_location
            )
            
            # Add documents to existing collection
            vector_store.add_documents(documents)
            
        else:
            # Create new collection
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=db_location
            )
        
        print(f"Successfully added {len(documents)} chunks from '{document_data['filename']}' to user collection")
        return True
        
    except Exception as e:
        print(f"Error adding document to user collection: {e}")
        return False

def remove_document_from_user_collection(document_id, db_location):
    """
    Removes all chunks of a document from user collection
    Args: document_id - unique document identifier, db_location - database path
    Returns: Boolean indicating success
    """
    collection_name = "user_documents"
    collection_path = os.path.join(db_location, collection_name)
    
    if not os.path.exists(collection_path):
        return True  # Nothing to remove
    
    try:
        # This is a limitation of ChromaDB - we can't easily delete by metadata
        # In a production system, you might want to use a different vector DB
        # For now, we'll print a warning
        print(f"Warning: Cannot remove document {document_id} from vector collection.")
        print("ChromaDB doesn't support selective deletion by metadata.")
        print("Document removed from PostgreSQL but remains in vector storage.")
        print("Consider rebuilding user collection if this becomes a problem.")
        
        return True
        
    except Exception as e:
        print(f"Error removing document from collection: {e}")
        return False

def get_collection_statistics(db_location):
    """
    Gets statistics about vector database collections
    Args: db_location - database path
    Returns: statistics dictionary
    """
    stats = {
        "static_algorithms": {"exists": False, "document_count": 0},
        "user_documents": {"exists": False, "document_count": 0}
    }
    
    try:
        # Check static collection
        static_path = os.path.join(db_location, "static_algorithms")
        if os.path.exists(static_path):
            stats["static_algorithms"]["exists"] = True
            try:
                client = chromadb.PersistentClient(path=db_location)
                collection = client.get_collection("static_algorithms")
                stats["static_algorithms"]["document_count"] = collection.count()
            except:
                pass
        
        # Check user collection
        user_path = os.path.join(db_location, "user_documents")
        if os.path.exists(user_path):
            stats["user_documents"]["exists"] = True
            try:
                client = chromadb.PersistentClient(path=db_location)
                collection = client.get_collection("user_documents")
                stats["user_documents"]["document_count"] = collection.count()
            except:
                pass
                
    except Exception as e:
        print(f"Error getting collection statistics: {e}")
    
    return stats

def test_retrieval(retriever, query="quicksort algorithm"):
    """
    Tests the retrieval system with a sample query
    Args: retriever - ChromaDB retriever object, query - test query
    """
    if not retriever:
        print("No retriever provided for testing")
        return
        
    print(f"\nTesting retrieval with query: '{query}'")
    print("-" * 50)
    
    try:
        results = retriever.invoke(query)
        print(f"Retrieved {len(results)} documents:")
        
        for i, doc in enumerate(results):
            print(f"\n{i+1}. Type: {doc.metadata.get('type', 'unknown')}")
            print(f"   Algorithm: {doc.metadata.get('algorithm', 'N/A')}")
            print(f"   Category: {doc.metadata.get('category', 'N/A')}")
            print(f"   Content preview: {doc.page_content[:100]}...")
            
    except Exception as e:
        print(f"Error during retrieval test: {e}")

if __name__ == "__main__":
    # Setup vector database with multi-collection support
    static_retriever, user_retriever_func = setup_vector_database()
    
    if static_retriever:
        # Test static knowledge retrieval
        test_retrieval(static_retriever, "quicksort implementation python")
        test_retrieval(static_retriever, "binary search time complexity") 
        test_retrieval(static_retriever, "sorting practice problems")
        
        # Test user documents retrieval (if any exist)
        user_retriever = user_retriever_func()
        if user_retriever:
            print("\n" + "="*50)
            print("Testing user documents collection:")
            test_retrieval(user_retriever, "algorithm notes")
        else:
            print("\nNo user documents found. Upload documents using /upload command in main.py")
        
        # Show collection statistics
        stats = get_collection_statistics("./chroma_db")
        print(f"\nCollection Statistics:")
        print(f"Static Knowledge: {stats['static_algorithms']['document_count']} documents")
        print(f"User Documents: {stats['user_documents']['document_count']} documents")
        
    else:
        print("Failed to setup vector database")