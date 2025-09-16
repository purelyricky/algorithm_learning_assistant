"""
Main Interface for Algorithm Learning Assistant
Privacy-preserving AI tutor for Computer Science students
"""

import json
import ast
from langchain_ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from colorama import Fore, Style, init
from tqdm import tqdm

# Import our custom modules
from vector_setup import setup_vector_database, add_document_to_user_collection, get_collection_statistics
from memory_system import (
    store_conversation, fetch_conversation_history, 
    update_student_progress, get_student_progress_summary,
    get_recommended_algorithms, record_practice_attempt,
    remove_last_conversation
)
# Import document processing modules  
from document_processor import process_document, get_supported_file_types
from document_manager import (
    store_document_metadata, get_user_documents, get_document_by_id,
    search_documents, get_document_statistics, delete_document
)
from federated_search import (
    federated_search, format_search_results_for_prompt, 
    get_enhanced_response_template
)

# Initialize colorama for cross-platform colored output
init(autoreset=True)

def initialize_system():
    """
    Initializes the learning assistant system components
    Now supports multi-collection vector database
    Returns: tuple (model, static_retriever, user_retriever_func) or (None, None, None) if failed
    """
    print(Fore.YELLOW + "🔧 Initializing Algorithm Learning Assistant...")
    
    try:
        # Initialize Ollama model
        print(Fore.YELLOW + "📡 Connecting to Ollama...")
        model = Ollama(model="llama3.2")
        
        # Setup vector database with multi-collection support
        print(Fore.YELLOW + "🗃️  Setting up knowledge databases...")
        static_retriever, user_retriever_func = setup_vector_database()
        
        if static_retriever is None:
            print(Fore.RED + "❌ Failed to initialize vector database")
            return None, None, None
            
        print(Fore.GREEN + "✅ System initialized successfully!")
        
        # Show collection statistics
        stats = get_collection_statistics("./chroma_db")
        print(Fore.CYAN + f"📊 Static Knowledge: {stats['static_algorithms']['document_count']} documents")
        print(Fore.CYAN + f"📊 User Documents: {stats['user_documents']['document_count']} documents")
        
        return model, static_retriever, user_retriever_func
        
    except Exception as e:
        print(Fore.RED + f"❌ Initialization failed: {e}")
        return None, None, None

def create_multi_queries(prompt, model):
    """
    Creates multiple search queries for better retrieval (inspired by video 2)
    Args: prompt - user's question, model - LLM instance
    Returns: list of query strings
    """
    query_system_message = """You are a query generation assistant. Given a student's question about algorithms or data structures, generate 3-5 specific search queries that will help find relevant information.

Generate queries as a Python list of strings. Focus on:
1. Conceptual understanding
2. Implementation details  
3. Practice problems
4. Related algorithms
5. Common use cases

Return ONLY a valid Python list, nothing else."""

    # Multi-shot learning examples to teach the model the correct format
    query_conversation = [
        {"role": "system", "content": query_system_message},
        {"role": "user", "content": "How does quicksort work?"},
        {"role": "assistant", "content": '["quicksort algorithm explanation", "quicksort pseudocode", "quicksort implementation", "quicksort time complexity", "quicksort practice problems"]'},
        {"role": "user", "content": "What is the time complexity of binary search?"},
        {"role": "assistant", "content": '["binary search time complexity", "binary search algorithm", "binary search vs linear search", "binary search implementation", "logarithmic time complexity"]'},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = model.invoke(query_conversation)
        print(Fore.YELLOW + f"🔍 Generated search queries for: {prompt[:50]}...")
        
        # Parse the response as a Python list
        try:
            queries = ast.literal_eval(response.content if hasattr(response, 'content') else str(response))
            if isinstance(queries, list):
                return queries
        except:
            pass
        
        # Fallback: return the original prompt
        return [prompt]
        
    except Exception as e:
        print(Fore.RED + f"Error generating queries: {e}")
        return [prompt]

def classify_relevance(query, context, model):
    """
    Classifies if retrieved context is relevant to the query
    Args: query, context, model
    Returns: boolean indicating relevance
    """
    classify_system_message = """You are a relevance classifier. Determine if the given context directly answers or relates to the student's query about algorithms/data structures.

Respond with ONLY 'yes' if relevant, 'no' if not relevant. No explanations."""

    classify_conversation = [
        {"role": "system", "content": classify_system_message},
        {"role": "user", "content": "Query: quicksort time complexity\nContext: Quicksort has average O(n log n) time complexity"},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content": "Query: binary search implementation\nContext: Bubble sort is a simple sorting algorithm"},
        {"role": "assistant", "content": "no"},
        {"role": "user", "content": f"Query: {query}\nContext: {context}"}
    ]
    
    try:
        response = model.invoke(classify_conversation)
        result = (response.content if hasattr(response, 'content') else str(response)).strip().lower()
        return result == 'yes'
    except:
        return True  # Default to including context if classification fails

def retrieve_relevant_context(prompt, static_retriever, user_retriever_func, model):
    """
    Enhanced retrieval using federated search across static and user sources
    Args: prompt, static_retriever, user_retriever_func, model
    Returns: formatted context string
    """
    # Get user retriever if available
    user_retriever = user_retriever_func() if user_retriever_func else None
    
    # Use federated search for comprehensive results
    search_results = federated_search(prompt, static_retriever, user_retriever, model)
    
    # Format results for LLM prompt
    return format_search_results_for_prompt(search_results)

def extract_algorithm_info(prompt):
    """
    Extracts algorithm name and category from prompt for progress tracking
    Args: prompt - user input
    Returns: tuple (algorithm, category)
    """
    # Simple keyword matching - could be improved with NLP
    prompt_lower = prompt.lower()
    
    algorithms = {
        'quicksort': 'sorting',
        'mergesort': 'sorting', 
        'merge sort': 'sorting',
        'binary search': 'searching',
        'linear search': 'searching',
        'bubble sort': 'sorting',
        'insertion sort': 'sorting',
        'selection sort': 'sorting',
        'binary tree': 'data_structures',
        'hash table': 'data_structures',
        'linked list': 'data_structures'
    }
    
    for algorithm, category in algorithms.items():
        if algorithm in prompt_lower:
            return algorithm.replace(' ', '_'), category
    
    return None, None

def handle_upload_command(file_path, static_retriever, user_retriever_func, model):
    """
    Handles /upload command for document ingestion
    """
    print(Fore.YELLOW + f"📄 Processing document: {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        return f"❌ File not found: {file_path}"
    
    # Process document
    document_data = process_document(file_path)
    
    if not document_data["success"]:
        return f"❌ Failed to process document: {document_data['error']}"
    
    # Store metadata in PostgreSQL
    if not store_document_metadata(document_data):
        return f"❌ Failed to store document metadata"
    
    # Add to vector database
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    if not add_document_to_user_collection(document_data, embeddings, "./chroma_db"):
        return f"❌ Failed to add document to vector database"
    
    return f"""✅ Document processed successfully!

📊 **Document Summary:**
• **Filename:** {document_data['filename']}
• **Type:** {document_data['file_extension']}
• **Size:** {document_data['file_size']:,} bytes
• **Chunks Created:** {document_data['total_chunks']}
• **Characters:** {document_data['total_characters']:,}

💡 You can now ask questions about this document or use /search to find specific content."""

def handle_documents_command():
    """
    Handles /documents command to list uploaded documents
    """
    print(Fore.YELLOW + "📚 Retrieving your uploaded documents...")
    
    documents = get_user_documents(limit=20)
    
    if not documents:
        return "📭 No documents uploaded yet. Use '/upload [filepath]' to add documents."
    
    doc_list = f"📚 **Your Uploaded Documents** ({len(documents)} total):\n\n"
    
    for i, doc in enumerate(documents, 1):
        doc_list += f"**{i}. {doc['filename']}**\n"
        doc_list += f"   • Type: {doc['file_extension']}\n"
        doc_list += f"   • Uploaded: {doc['upload_date']}\n"
        doc_list += f"   • Chunks: {doc['total_chunks']}\n"
        doc_list += f"   • Accessed: {doc['access_count']} times\n"
        doc_list += f"   • Summary: {doc['content_summary'][:100]}...\n\n"
    
    return doc_list

def handle_search_in_command(query, document_name=None):
    """
    Handles /search command for searching in documents
    """
    print(Fore.YELLOW + f"🔍 Searching documents for: {query}")
    
    if document_name:
        # Search in specific document
        documents = search_documents(document_name, limit=1)
        if not documents:
            return f"❌ Document '{document_name}' not found"
        
        return f"🔍 Search functionality for specific documents will use federated search in natural language queries.\n\nTry asking: 'What does {document_name} say about {query}?'"
    
    else:
        # Search across all documents
        documents = search_documents(query, limit=10)
        
        if not documents:
            return f"❌ No documents found matching: {query}"
        
        results = f"🔍 **Search Results for '{query}'** ({len(documents)} found):\n\n"
        
        for i, doc in enumerate(documents, 1):
            results += f"**{i}. {doc['filename']}**\n"
            results += f"   • Type: {doc['file_extension']}\n"
            results += f"   • Summary: {doc['content_summary'][:150]}...\n\n"
        
        return results

def handle_summarize_command(document_identifier, static_retriever, user_retriever_func, model):
    """
    Handles /summarize command for document summarization
    """
    print(Fore.YELLOW + f"📋 Generating summary for: {document_identifier}")
    
    # Try to find document
    documents = search_documents(document_identifier, limit=1)
    
    if not documents:
        return f"❌ Document not found: {document_identifier}"
    
    document = documents[0]
    doc_details = get_document_by_id(document['document_id'])
    
    if not doc_details:
        return f"❌ Could not retrieve document details"
    
    # Use federated search to get document content
    user_retriever = user_retriever_func() if user_retriever_func else None
    
    if not user_retriever:
        return f"❌ No user documents available for summarization"
    
    # Search for content related to this document
    query = f"content from {doc_details['filename']}"
    search_results = federated_search(query, static_retriever, user_retriever, model)
    
    if not search_results["results"]:
        return f"❌ Could not retrieve content for summarization"
    
    # Create summarization template
    template = """You are creating a summary of a student's uploaded document.

Document Information:
- Filename: {filename}
- Type: {file_type}
- Upload Date: {upload_date}
- Size: {total_chars} characters

Document Content:
{context}

Create a comprehensive summary that includes:
1. **Main Topics Covered**
2. **Key Concepts and Definitions**
3. **Important Algorithms/Code Examples** (if any)
4. **Practical Applications**
5. **Study Recommendations**

Make it useful for review and study purposes."""
    
    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | model
    
    # Get document content from search results
    context = format_search_results_for_prompt(search_results)
    
    response = chain.invoke({
        "filename": doc_details["filename"],
        "file_type": doc_details["file_extension"], 
        "upload_date": doc_details["upload_date"],
        "total_chars": doc_details["total_characters"],
        "context": context
    })
    
    return response.content if hasattr(response, 'content') else str(response)

def handle_remove_document_command(document_identifier):
    """
    Handles /remove command for document deletion
    """
    print(Fore.YELLOW + f"🗑️ Removing document: {document_identifier}")
    
    # Find document
    documents = search_documents(document_identifier, limit=1)
    
    if not documents:
        return f"❌ Document not found: {document_identifier}"
    
    document = documents[0]
    
    # Confirm deletion (in a real app, you'd want user confirmation)
    if delete_document(document['document_id']):
        return f"✅ Document '{document['filename']}' removed successfully.\n\n⚠️  Note: Document removed from database but may remain in vector storage due to ChromaDB limitations."
    else:
        return f"❌ Failed to remove document: {document['filename']}"

def handle_doc_stats_command():
    """
    Handles /docstats command for document statistics
    """
    print(Fore.YELLOW + "📊 Analyzing document statistics...")
    
    stats = get_document_statistics()
    
    if not stats or stats.get('total_documents', 0) == 0:
        return "📊 No document statistics available. Upload some documents first!"
    
    stats_report = f"""📊 **Document Statistics Report**

📈 **Overall Stats:**
   • Total Documents: {stats['total_documents']}
   • Total Storage: {stats['total_size_bytes']:,} bytes
   • Total Chunks: {stats['total_chunks']:,}
   • Total Characters: {stats['total_characters']:,}
   • Average Access Count: {stats['avg_access_count']}

📁 **File Types:**"""
    
    for file_type, count in stats['file_type_distribution'].items():
        stats_report += f"\n   • {file_type}: {count} files"
    
    if stats['most_accessed']:
        stats_report += f"\n\n🔥 **Most Accessed Documents:**"
        for doc in stats['most_accessed']:
            stats_report += f"\n   • {doc['filename']}: {doc['access_count']} times"
    
    return stats_report
    """
    Handles /explain command for algorithm explanations
    """
    print(Fore.YELLOW + f"📚 Explaining {algorithm}...")
    
    # Create specific query for explanation
    query = f"{algorithm} algorithm explanation how it works"
    
    # Retrieve relevant context
    relevant_contexts = retrieve_relevant_context(query, retriever, model)
    
    if not relevant_contexts:
        return f"I don't have information about {algorithm} in my knowledge base."
    
    # Create explanation prompt
    template = """You are an expert computer science tutor. Explain the {algorithm} algorithm to a CS student.

Use this information from my knowledge base:
{context}

Provide a clear, educational explanation covering:
1. What the algorithm does
2. How it works (step by step)
3. Time and space complexity
4. When to use it
5. A simple example

Make it engaging and easy to understand."""

    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | model
    
    response = chain.invoke({
        "algorithm": algorithm,
        "context": "\n\n".join(relevant_contexts[:3])  # Limit context size
    })
    
    # Update progress
    alg_clean, category = extract_algorithm_info(algorithm)
    if alg_clean and category:
        update_student_progress(alg_clean, category)
    
    return response.content if hasattr(response, 'content') else str(response)

def handle_practice_command(topic, model, retriever):
    """
    Handles /practice command for practice problems
    """
    print(Fore.YELLOW + f"💪 Finding practice problems for {topic}...")
    
    # Query for practice problems
    query = f"{topic} practice problems exercises"
    relevant_contexts = retrieve_relevant_context(query, retriever, model)
    
    template = """You are a CS tutor creating practice problems. Based on this information:

{context}

Generate 2-3 practice problems for {topic}. For each problem:
1. State the problem clearly
2. Provide difficulty level (Easy/Medium/Hard)
3. Give a helpful hint
4. Mention what concepts it tests

Make problems progressive in difficulty."""

    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | model
    
    response = chain.invoke({
        "topic": topic,
        "context": "\n\n".join(relevant_contexts[:2])
    })
    
    # Record practice attempt (simplified)
    alg_clean, category = extract_algorithm_info(topic)
    if alg_clean:
        record_practice_attempt(0, alg_clean, "medium", False)  # Placeholder values
    
    return response.content if hasattr(response, 'content') else str(response)

def handle_code_command(algorithm, language, model, retriever):
    """
    Handles /code command for implementation examples
    """
    print(Fore.YELLOW + f"💻 Finding {language} code for {algorithm}...")
    
    query = f"{algorithm} implementation {language} code example"
    relevant_contexts = retrieve_relevant_context(query, retriever, model)
    
    template = """You are a coding tutor. Show a {language} implementation of {algorithm}.

Based on this information:
{context}

Provide:
1. Clean, well-commented {language} code
2. Explanation of key parts
3. Usage example
4. Common pitfalls to avoid

Focus on readability and educational value."""

    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | model
    
    response = chain.invoke({
        "algorithm": algorithm,
        "language": language,
        "context": "\n\n".join(relevant_contexts[:2])
    })
    
    return response.content if hasattr(response, 'content') else str(response)

def handle_progress_command():
    """
    Handles /progress command to show learning progress
    """
    print(Fore.YELLOW + "📊 Analyzing your learning progress...")
    
    progress = get_student_progress_summary()
    
    if not progress or progress['total_algorithms_studied'] == 0:
        return "You haven't studied any algorithms yet! Try '/explain quicksort' to get started."
    
    progress_report = f"""
{Fore.CYAN}📈 Your Learning Progress Report

🎯 Overall Stats:
   • Algorithms Studied: {progress['total_algorithms_studied']}
   • Study Sessions: {progress['total_study_sessions']}
   • Average Mastery: {progress['average_mastery']}/1.00
   • Practice Attempts: {progress['practice_attempts']}

📚 Progress by Category:"""
    
    for cat in progress['categories']:
        progress_report += f"\n   • {cat['category'].title()}: {cat['algorithms_count']} algorithms (avg: {cat['avg_mastery']}/1.00)"
    
    progress_report += f"\n\n🕒 Recent Activity:"
    for activity in progress['recent_activity']:
        progress_report += f"\n   • {activity['algorithm']} - {activity['last_studied']} (mastery: {activity['mastery']})"
    
    # Add recommendations
    recommendations = get_recommended_algorithms()
    if recommendations:
        progress_report += f"\n\n💡 Recommendations:"
        for rec in recommendations[:3]:
            progress_report += f"\n   • {rec['algorithm']} - {rec['reason']}"
    
    return progress_report

def handle_review_command(model, retriever):
    """
    Handles /review command to review previous concepts
    """
    print(Fore.YELLOW + "🔄 Reviewing your previous learning...")
    
    # Get recent conversation history
    history = fetch_conversation_history(5)
    
    if not history:
        return "No previous conversations to review."
    
    # Extract topics from recent conversations
    topics = set()
    for conv in history:
        if conv['algorithm_topic']:
            topics.add(conv['algorithm_topic'])
    
    if not topics:
        return "No specific algorithms found in recent conversations to review."
    
    # Create review query
    topics_list = list(topics)[:3]  # Review up to 3 recent topics
    query = f"review summary {' '.join(topics_list)}"
    
    relevant_contexts = retrieve_relevant_context(query, retriever, model)
    
    template = """Create a study review for these algorithms: {topics}

Based on this information:
{context}

Provide a concise review covering:
1. Key concepts for each algorithm
2. Important differences between them
3. When to use each one
4. Common mistakes to avoid

Format as a quick reference guide."""

    prompt_template = ChatPromptTemplate.from_template(template)
    chain = prompt_template | model
    
    response = chain.invoke({
        "topics": ", ".join(topics_list),
        "context": "\n\n".join(relevant_contexts[:3])
    })
    
    return response.content if hasattr(response, 'content') else str(response)

def handle_review_command(model, static_retriever, user_retriever_func):
    """
    Handles /review command to review previous concepts
    Now uses federated search across all sources
    """
    print(Fore.YELLOW + "🔄 Reviewing your previous learning...")
    
    # Get recent conversation history
    history = fetch_conversation_history(5)
    
    if not history:
        return "No previous conversations to review."
    
    # Extract topics from recent conversations
    topics = set()
    for conv in history:
        if conv['algorithm_topic']:
            topics.add(conv['algorithm_topic'])
    
    if not topics:
        return "No specific algorithms found in recent conversations to review."
    
    # Create review query
    topics_list = list(topics)[:3]  # Review up to 3 recent topics
    query = f"review summary {' '.join(topics_list)}"
    
    context = retrieve_relevant_context(query, static_retriever, user_retriever_func, model)
    
    template = get_enhanced_response_template("comparison")
    chain = template | model
    
    response = chain.invoke({
        "question": f"Create a study review for these algorithms: {', '.join(topics_list)}",
        "context": context
    })
    
    return response.content if hasattr(response, 'content') else str(response)

def show_help():
    """Shows available commands including new document processing commands"""
    return f"""{Fore.CYAN}🎓 Algorithm Learning Assistant - Available Commands:

{Fore.GREEN}📚 Learning Commands:{Fore.WHITE}
/explain [algorithm]     - Get detailed explanation of an algorithm
   Example: /explain quicksort

/practice [topic]        - Generate practice problems for a topic  
   Example: /practice sorting

/code [algorithm] [lang] - Show implementation in specific language
   Example: /code binary_search python

/progress               - View your learning progress and statistics

/review                 - Review recently studied algorithms

{Fore.GREEN}📄 Document Commands:{Fore.WHITE}
/upload [filepath]      - Upload and process a document (PDF, DOCX, code files)
   Example: /upload ./CS_Notes.pdf

/documents              - List all your uploaded documents

/search [query]         - Search across your uploaded documents
   Example: /search "binary trees"

/summarize [document]   - Generate summary of a specific document
   Example: /summarize "algorithm_notes.pdf"

/remove [document]      - Remove a document from the system
   Example: /remove "old_notes.pdf"

/docstats              - View document storage statistics

{Fore.GREEN}🔧 System Commands:{Fore.WHITE}
/help                  - Show this help message

/forget                - Remove last conversation from memory

q                      - Quit the assistant

{Fore.YELLOW}💡 Natural Language Queries:
You can also ask questions naturally! The system will search both the built-in 
algorithm knowledge and your uploaded documents.

Examples:
• "What's the difference between quicksort and mergesort?"
• "What did my lecture notes say about binary trees?"
• "Show me a Python implementation of depth-first search"

{Fore.GREEN}📁 Supported File Types:{Fore.WHITE}
• PDF documents (with OCR support)
• Microsoft Word (.docx)
• Python source code (.py)
• Java source code (.java)
• Plain text (.txt)
• Markdown (.md)
• JSON files (.json)
"""

def main():
    """Main interaction loop with enhanced document processing capabilities"""
    print(Fore.CYAN + """
    ╔═══════════════════════════════════════════╗
    ║     🎓 Algorithm Learning Assistant        ║
    ║    Privacy-Preserving CS Education AI     ║
    ║         Enhanced with Document Support    ║
    ╚═══════════════════════════════════════════╝
    """)
    
    # Initialize system with multi-collection support
    model, static_retriever, user_retriever_func = initialize_system()
    
    if not model or not static_retriever:
        print(Fore.RED + "Failed to initialize system. Please check your setup.")
        return
    
    print(Fore.GREEN + "\n✅ Ready to help you learn algorithms!")
    print(Fore.YELLOW + "Type '/help' for commands or ask any algorithm question.")
    print(Fore.CYAN + "📄 New: Upload documents with '/upload [filepath]' for personalized learning!")
    print(Fore.WHITE + "Type 'q' to quit.\n")
    
    while True:
        try:
            # Get user input
            user_input = input(Fore.WHITE + "You: ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print(Fore.YELLOW + "👋 Happy learning! See you next time.")
                break
            
            if not user_input:
                continue
            
            print()  # Add spacing
            
            # Handle commands
            if user_input.startswith('/'):
                parts = user_input.split()
                command = parts[0].lower()
                
                if command == '/help':
                    response = show_help()
                
                elif command == '/explain' and len(parts) > 1:
                    algorithm = ' '.join(parts[1:])
                    response = handle_explain_command(algorithm, model, static_retriever, user_retriever_func)
                
                elif command == '/practice' and len(parts) > 1:
                    topic = ' '.join(parts[1:])
                    response = handle_practice_command(topic, model, static_retriever, user_retriever_func)
                
                elif command == '/code' and len(parts) > 2:
                    algorithm = parts[1]
                    language = parts[2]
                    response = handle_code_command(algorithm, language, model, static_retriever, user_retriever_func)
                
                elif command == '/progress':
                    response = handle_progress_command()
                
                elif command == '/review':
                    response = handle_review_command(model, static_retriever, user_retriever_func)
                
                # Document processing commands
                elif command == '/upload' and len(parts) > 1:
                    file_path = ' '.join(parts[1:])  # Handle file paths with spaces
                    response = handle_upload_command(file_path, static_retriever, user_retriever_func, model)
                
                elif command == '/documents':
                    response = handle_documents_command()
                
                elif command == '/search' and len(parts) > 1:
                    search_query = ' '.join(parts[1:])
                    response = handle_search_in_command(search_query)
                
                elif command == '/summarize' and len(parts) > 1:
                    document_name = ' '.join(parts[1:])
                    response = handle_summarize_command(document_name, static_retriever, user_retriever_func, model)
                
                elif command == '/remove' and len(parts) > 1:
                    document_name = ' '.join(parts[1:])
                    response = handle_remove_document_command(document_name)
                
                elif command == '/docstats':
                    response = handle_doc_stats_command()
                
                elif command == '/forget':
                    if remove_last_conversation():
                        response = "🗑️  Last conversation removed from memory."
                    else:
                        response = "❌ Failed to remove conversation."
                
                else:
                    response = f"❓ Unknown command. Type '/help' for available commands."
            
            else:
                # Handle natural language questions with federated search
                print(Fore.YELLOW + "🤔 Processing your question...")
                
                # Use federated search for comprehensive context
                context = retrieve_relevant_context(user_input, static_retriever, user_retriever_func, model)
                
                # Create conversational template based on available sources
                if context and "No relevant information found" not in context:
                    # Check if user documents are referenced in context
                    has_user_content = "Your Study Materials" in context
                    
                    if has_user_content:
                        # Use document-aware template
                        template = get_enhanced_response_template("document_query")
                    else:
                        # Use general explanation template
                        template = get_enhanced_response_template("explanation")
                    
                    chain = template | model
                    
                    response = chain.invoke({
                        "question": user_input,
                        "context": context
                    })
                    response = response.content if hasattr(response, 'content') else str(response)
                else:
                    response = "I don't have specific information about that in my knowledge base. Try using /explain [algorithm] for detailed explanations, or /upload [filepath] to add your own study materials."
            
            # Display response
            print(Fore.GREEN + "🤖 Assistant: " + response)
            print()
            
            # Store conversation (extract algorithm info for better tracking)
            algorithm, category = extract_algorithm_info(user_input)
            command_type = user_input.split()[0] if user_input.startswith('/') else 'question'
            
            store_conversation(
                user_input, 
                response,
                command_type=command_type,
                algorithm_topic=algorithm,
                difficulty='intermediate'  # Could be made dynamic
            )
            
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\n\n👋 Goodbye! Keep practicing those algorithms.")
            break
        except Exception as e:
            print(Fore.RED + f"❌ An error occurred: {e}")
            print(Fore.YELLOW + "Please try again or restart the assistant.")

if __name__ == "__main__":
    # Add import for os (needed for file path handling)
    import os
    from langchain_ollama import OllamaEmbeddings
    
    main()