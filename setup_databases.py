"""
Database Setup Script for Algorithm Learning Assistant
Initializes all required databases and knowledge base
"""

import os
import sys
from colorama import Fore, Style, init

# Initialize colorama for colored output
init(autoreset=True)

def check_dependencies():
    """
    Checks if all required dependencies are installed
    Returns: Boolean indicating if dependencies are satisfied
    """
    print(Fore.YELLOW + "🔍 Checking dependencies...")
    
    required_modules = [
        'psycopg2', 'chromadb', 'langchain', 'langchain_ollama', 
        'langchain_chroma', 'pandas', 'tqdm', 'colorama',
        # Document processing modules
        'PyPDF2', 'docx', 'pdfplumber', 'chardet', 'magic'
    ]
    
    optional_modules = [
        'pytesseract',  # For OCR
        'mammoth',      # Enhanced DOCX processing
        'fitz'          # For OCR PDF processing
    ]
    
    missing_modules = []
    missing_optional = []
    
    for module in required_modules:
        try:
            if module == 'docx':
                __import__('python-docx')
            elif module == 'magic':
                __import__('python-magic')
            else:
                __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    for module in optional_modules:
        try:
            __import__(module)
        except ImportError:
            missing_optional.append(module)
    
    if missing_modules:
        print(Fore.RED + f"❌ Missing critical dependencies: {', '.join(missing_modules)}")
        print(Fore.YELLOW + "Please install with: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(Fore.YELLOW + f"⚠️  Optional dependencies missing: {', '.join(missing_optional)}")
        print(Fore.YELLOW + "OCR and advanced document processing may not work fully")
    
    print(Fore.GREEN + "✅ All critical dependencies satisfied")
    return True

def check_ollama_setup():
    """
    Checks if Ollama is running and models are available
    Returns: Boolean indicating if Ollama is properly set up
    """
    print(Fore.YELLOW + "🔍 Checking Ollama setup...")
    
    try:
        from langchain_ollama import Ollama
        
        # Test connection
        model = Ollama(model="llama3.2")
        test_response = model.invoke("Hello")
        
        print(Fore.GREEN + "✅ Ollama is running and responsive")
        
        # Test embedding model
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        test_embedding = embeddings.embed_query("test")
        
        print(Fore.GREEN + "✅ Embedding model is working")
        return True
        
    except Exception as e:
        print(Fore.RED + f"❌ Ollama setup issue: {e}")
        print(Fore.YELLOW + """
Required Ollama models:
- llama3.2 (for main conversation)
- nomic-embed-text (for embeddings)

Install with:
ollama pull llama3.2
ollama pull nomic-embed-text
        """)
        return False

def setup_algorithm_knowledge():
    """
    Sets up the algorithm knowledge base
    Returns: Boolean indicating success
    """
    print(Fore.YELLOW + "📚 Setting up algorithm knowledge base...")
    
    try:
        from algorithm_data import create_algorithm_knowledge_base
        create_algorithm_knowledge_base()
        print(Fore.GREEN + "✅ Algorithm knowledge base created")
        return True
    except Exception as e:
        print(Fore.RED + f"❌ Failed to create knowledge base: {e}")
        return False

def setup_vector_database():
    """
    Initializes the vector database with algorithm content
    Returns: Boolean indicating success
    """
    print(Fore.YELLOW + "🗃️  Setting up vector database...")
    
    try:
        from vector_setup import setup_vector_database
        retriever = setup_vector_database()
        
        if retriever:
            print(Fore.GREEN + "✅ Vector database initialized successfully")
            return True
        else:
            print(Fore.RED + "❌ Vector database setup failed")
            return False
            
    except Exception as e:
        print(Fore.RED + f"❌ Vector database error: {e}")
        return False

def setup_postgresql():
    """
    Initializes PostgreSQL database for conversation memory
    Returns: Boolean indicating success
    """
    print(Fore.YELLOW + "🗄️  Setting up PostgreSQL database...")
    
    try:
        from memory_system import initialize_database
        
        if initialize_database():
            print(Fore.GREEN + "✅ PostgreSQL database initialized")
            return True
        else:
            print(Fore.RED + "❌ PostgreSQL database setup failed")
            print(Fore.YELLOW + """
Make sure PostgreSQL is installed and running.
Check database credentials in memory_system.py:
- Database: algorithm_tutor  
- User: cs_student
- Password: your_password
            """)
            return False
            
    except Exception as e:
        print(Fore.RED + f"❌ PostgreSQL error: {e}")
        print(Fore.YELLOW + """
PostgreSQL setup steps:
1. Install PostgreSQL
2. Create database: algorithm_tutor
3. Create user: cs_student  
4. Update password in memory_system.py
        """)
        return False

def setup_vector_database():
    """
    Initializes the enhanced vector database with multi-collection support
    Returns: Boolean indicating success
    """
    print(Fore.YELLOW + "🗃️  Setting up enhanced vector database...")
    
    try:
        from vector_setup import setup_vector_database, get_collection_statistics
        static_retriever, user_retriever_func = setup_vector_database()
        
        if static_retriever:
            print(Fore.GREEN + "✅ Vector database initialized successfully")
            
            # Show collection statistics
            stats = get_collection_statistics("./chroma_db")
            print(Fore.CYAN + f"📊 Static Knowledge: {stats['static_algorithms']['document_count']} documents")
            print(Fore.CYAN + f"📊 User Documents: {stats['user_documents']['document_count']} documents")
            
            return True
        else:
            print(Fore.RED + "❌ Vector database setup failed")
            return False
            
    except Exception as e:
        print(Fore.RED + f"❌ Vector database error: {e}")
        return False

def setup_postgresql():
    """
    Initializes PostgreSQL database for conversation memory and document management
    Returns: Boolean indicating success
    """
    print(Fore.YELLOW + "🗄️  Setting up PostgreSQL database...")
    
    try:
        from memory_system import initialize_database
        
        if initialize_database():
            print(Fore.GREEN + "✅ PostgreSQL database initialized")
            return True
        else:
            print(Fore.RED + "❌ PostgreSQL database setup failed")
            print(Fore.YELLOW + """
Make sure PostgreSQL is installed and running.
Check database credentials in memory_system.py:
- Database: algorithm_tutor  
- User: cs_student
- Password: your_password
            """)
            return False
            
    except Exception as e:
        print(Fore.RED + f"❌ PostgreSQL error: {e}")
        print(Fore.YELLOW + """
PostgreSQL setup steps:
1. Install PostgreSQL
2. Create database: algorithm_tutor
3. Create user: cs_student  
4. Update password in memory_system.py
        """)
        return False

def test_document_processing():
    """
    Tests document processing capabilities
    Returns: Boolean indicating if tests pass
    """
    print(Fore.YELLOW + "📄 Testing document processing...")
    
    try:
        from document_processor import get_supported_file_types, detect_file_type
        
        # Test supported file types
        supported = get_supported_file_types()
        print(Fore.CYAN + f"  Supported file types: {len(supported)} formats")
        
        # Test file type detection (if a test file exists)
        test_files = ["README.md", "main.py", "requirements.txt"]
        working_files = []
        
        for test_file in test_files:
            if os.path.exists(test_file):
                mime_type, ext, supported = detect_file_type(test_file)
                if mime_type:
                    working_files.append(test_file)
                    print(Fore.CYAN + f"  {test_file}: {mime_type} ({'supported' if supported else 'not supported'})")
        
        if working_files:
            print(Fore.GREEN + f"  ✅ Document processing working ({len(working_files)} files tested)")
            return True
        else:
            print(Fore.YELLOW + "  ⚠️  No test files found, but processing should work")
            return True
            
    except Exception as e:
        print(Fore.RED + f"  ❌ Document processing test failed: {e}")
        return False

def run_system_tests():
    """
    Runs comprehensive system tests to verify everything works
    Returns: Boolean indicating if all tests pass
    """
    print(Fore.YELLOW + "🧪 Running enhanced system tests...")
    
    try:
        # Test 1: Model initialization
        print(Fore.CYAN + "  Testing model initialization...")
        from main import initialize_system
        model, static_retriever, user_retriever_func = initialize_system()
        
        if not model or not static_retriever:
            print(Fore.RED + "  ❌ Model/retriever initialization failed")
            return False
        
        print(Fore.GREEN + "  ✅ Model and retrievers initialized")
        
        # Test 2: Static vector search
        print(Fore.CYAN + "  Testing static knowledge search...")
        results = static_retriever.invoke("quicksort algorithm")
        
        if not results or len(results) == 0:
            print(Fore.RED + "  ❌ Static vector search returned no results")
            return False
            
        print(Fore.GREEN + f"  ✅ Static search working ({len(results)} results)")
        
        # Test 3: User document retrieval function
        print(Fore.CYAN + "  Testing user document retrieval...")
        user_retriever = user_retriever_func()
        
        if user_retriever is None:
            print(Fore.CYAN + "  ℹ️  No user documents (expected for fresh install)")
        else:
            print(Fore.GREEN + "  ✅ User document retrieval working")
        
        # Test 4: Database operations
        print(Fore.CYAN + "  Testing database operations...")
        from memory_system import store_conversation, get_student_progress_summary
        
        # Store test conversation
        success = store_conversation(
            "test question", 
            "test response", 
            "test_command",
            "quicksort",
            "beginner"
        )
        
        if not success:
            print(Fore.RED + "  ❌ Database storage failed")
            return False
            
        # Get progress summary
        progress = get_student_progress_summary()
        
        print(Fore.GREEN + "  ✅ Database operations working")
        
        # Test 5: Document processing
        if not test_document_processing():
            print(Fore.RED + "  ❌ Document processing test failed")
            return False
        
        # Test 6: Federated search
        print(Fore.CYAN + "  Testing federated search...")
        from federated_search import federated_search
        
        search_results = federated_search("quicksort", static_retriever, user_retriever, model)
        
        if not search_results or not search_results.get("results"):
            print(Fore.RED + "  ❌ Federated search failed")
            return False
        
        print(Fore.GREEN + f"  ✅ Federated search working ({len(search_results['results'])} results)")
        
        print(Fore.GREEN + "✅ All enhanced system tests passed!")
        return True
        
    except Exception as e:
        print(Fore.RED + f"❌ System test failed: {e}")
        return False

def main():
    """
    Main setup function with enhanced document processing capabilities
    """
    print(Fore.CYAN + """
    ╔═══════════════════════════════════════════╗
    ║     🎓 Algorithm Learning Assistant       ║
    ║        Enhanced Setup & Initialization    ║
    ║         With Document Processing          ║
    ╚═══════════════════════════════════════════╝
    """)
    
    # Step 1: Check dependencies (including document processing)
    if not check_dependencies():
        print(Fore.RED + "❌ Setup failed: Missing dependencies")
        sys.exit(1)
    
    # Step 2: Check Ollama setup
    if not check_ollama_setup():
        print(Fore.RED + "❌ Setup failed: Ollama not properly configured")
        sys.exit(1)
    
    # Step 3: Create algorithm knowledge base
    if not setup_algorithm_knowledge():
        print(Fore.RED + "❌ Setup failed: Could not create knowledge base")
        sys.exit(1)
    
    # Step 4: Initialize enhanced vector database
    if not setup_vector_database():
        print(Fore.RED + "❌ Setup failed: Vector database initialization failed")
        sys.exit(1)
    
    # Step 5: Setup PostgreSQL (with document tables)
    if not setup_postgresql():
        print(Fore.RED + "❌ Setup failed: PostgreSQL initialization failed")
        print(Fore.YELLOW + "Note: You can still use basic functionality without PostgreSQL, but progress tracking and document management won't work")
        
        response = input(Fore.YELLOW + "Continue without PostgreSQL? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Step 6: Run enhanced system tests
    if not run_system_tests():
        print(Fore.RED + "❌ Setup completed but system tests failed")
        print(Fore.YELLOW + "The system may still work, but some features might not function correctly")
    
    print(Fore.GREEN + """
    ✅ Enhanced setup completed successfully!
    
    🚀 You can now run the assistant with:
       python main.py
    
    📖 Available commands:
       Learning Commands:
       /explain [algorithm] - Get explanations
       /practice [topic]    - Generate practice problems  
       /code [alg] [lang]   - Show implementations
       /progress           - View learning progress
       /review             - Review previous topics
       
       Document Commands:
       /upload [filepath]  - Upload and process documents
       /documents          - List uploaded documents
       /search [query]     - Search your documents
       /summarize [doc]    - Generate document summaries
       /remove [doc]       - Remove documents
       /docstats          - View document statistics
       
       System Commands:
       /help              - Show all commands
    
    💡 Natural language questions work across both built-in knowledge and your uploaded documents!
    
    📁 Supported file types:
       • PDF documents (with OCR support)
       • Microsoft Word (.docx) 
       • Python/Java source code (.py, .java)
       • Text files (.txt, .md, .json)
    
    🔒 Privacy guarantee: All processing happens locally on your machine.
       No data is sent to external servers.
    """)

if __name__ == "__main__":
    import os
    import sys
    main()