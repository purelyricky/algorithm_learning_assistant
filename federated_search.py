"""
Federated Search System for Algorithm Learning Assistant
Handles multi-source RAG across static knowledge and user documents
"""

import ast
from langchain_ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from document_manager import record_document_usage, get_document_by_id

def create_enhanced_queries(prompt, model):
    """
    Creates multiple specialized search queries for different knowledge sources
    Args: prompt - user's question, model - LLM instance  
    Returns: dictionary with categorized queries
    """
    query_system_message = """You are a query generation assistant for an educational AI system. Given a student's question, generate 5 specific search queries categorized for different knowledge sources.

Generate queries as a Python dictionary with these categories:
- "conceptual": queries for theoretical explanations
- "implementation": queries for code examples and implementations  
- "practice": queries for problems and exercises
- "user_content": queries for student's uploaded documents
- "comparison": queries for comparing algorithms/concepts

Return ONLY a valid Python dictionary, nothing else."""

    # Enhanced multi-shot learning with categorized examples
    query_conversation = [
        {"role": "system", "content": query_system_message},
        {"role": "user", "content": "How does quicksort work and can you show me Python code?"},
        {"role": "assistant", "content": '''{"conceptual": ["quicksort algorithm explanation", "divide and conquer sorting"], "implementation": ["quicksort python code", "quicksort implementation example"], "practice": ["quicksort practice problems", "sorting algorithm exercises"], "user_content": ["quicksort notes", "sorting lecture materials"], "comparison": ["quicksort vs mergesort", "sorting algorithm comparison"]}'''},
        {"role": "user", "content": "What's the time complexity of binary search?"},
        {"role": "assistant", "content": '''{"conceptual": ["binary search time complexity", "logarithmic complexity analysis"], "implementation": ["binary search algorithm", "binary search code"], "practice": ["binary search problems", "complexity analysis exercises"], "user_content": ["search algorithm notes", "complexity study materials"], "comparison": ["binary vs linear search complexity", "search algorithm performance"]}'''},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = model.invoke(query_conversation)
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse the response as a Python dictionary
        try:
            queries = ast.literal_eval(response_content)
            if isinstance(queries, dict):
                return queries
        except:
            pass
        
        # Fallback: create simple categorized queries
        return {
            "conceptual": [prompt, f"{prompt} explanation"],
            "implementation": [f"{prompt} code", f"{prompt} implementation"],
            "practice": [f"{prompt} problems", f"{prompt} exercises"],
            "user_content": [prompt],
            "comparison": [f"{prompt} comparison", f"{prompt} vs alternatives"]
        }
        
    except Exception as e:
        print(f"Error generating enhanced queries: {e}")
        return {
            "conceptual": [prompt],
            "implementation": [prompt],
            "practice": [prompt],
            "user_content": [prompt],
            "comparison": [prompt]
        }

def classify_query_intent(prompt, model):
    """
    Classifies the user's query intent to determine search strategy
    Args: prompt - user's question, model - LLM instance
    Returns: intent classification
    """
    intent_system_message = """Classify the user's query intent for an educational AI system. Respond with ONLY one of these categories:

- "explanation": User wants to understand a concept or algorithm
- "implementation": User wants to see code examples or implementations
- "practice": User wants practice problems or exercises
- "comparison": User wants to compare different approaches/algorithms
- "help": User needs general help or guidance
- "document_query": User is asking about their uploaded materials

Respond with only the category name, nothing else."""

    intent_conversation = [
        {"role": "system", "content": intent_system_message},
        {"role": "user", "content": "How does quicksort work?"},
        {"role": "assistant", "content": "explanation"},
        {"role": "user", "content": "Show me Python code for merge sort"},
        {"role": "assistant", "content": "implementation"},
        {"role": "user", "content": "Give me some sorting practice problems"},
        {"role": "assistant", "content": "practice"},
        {"role": "user", "content": "What did my lecture notes say about binary trees?"},
        {"role": "assistant", "content": "document_query"},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = model.invoke(intent_conversation)
        intent = (response.content if hasattr(response, 'content') else str(response)).strip().lower()
        
        valid_intents = ["explanation", "implementation", "practice", "comparison", "help", "document_query"]
        return intent if intent in valid_intents else "explanation"
        
    except Exception as e:
        print(f"Error classifying intent: {e}")
        return "explanation"

def search_static_knowledge(queries, static_retriever, max_results=3):
    """
    Searches the static algorithm knowledge base
    Args: queries - list of search queries, static_retriever - ChromaDB retriever
    Returns: list of relevant documents with metadata
    """
    results = []
    
    for query in queries:
        try:
            docs = static_retriever.invoke(query)
            for doc in docs[:max_results]:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": "static_knowledge",
                    "query": query,
                    "relevance_score": 1.0  # Could implement proper scoring
                })
        except Exception as e:
            print(f"Error searching static knowledge for '{query}': {e}")
            continue
    
    return results

def search_user_documents(queries, user_retriever, max_results=3):
    """
    Searches user-uploaded documents
    Args: queries - list of search queries, user_retriever - user document retriever
    Returns: list of relevant documents with metadata
    """
    results = []
    
    if not user_retriever:
        return results
    
    for query in queries:
        try:
            docs = user_retriever.invoke(query)
            for doc in docs[:max_results]:
                # Record usage for analytics
                doc_id = doc.metadata.get("document_id")
                if doc_id:
                    record_document_usage(doc_id, query, "search")
                
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "source": "user_documents", 
                    "query": query,
                    "relevance_score": 1.0
                })
        except Exception as e:
            print(f"Error searching user documents for '{query}': {e}")
            continue
    
    return results

def rank_and_filter_results(results, model, original_query, max_final_results=5):
    """
    Ranks and filters search results for relevance
    Args: results - search results, model - LLM, original_query - user's query
    Returns: filtered and ranked results
    """
    if not results:
        return []
    
    # Simple relevance classification for each result
    relevant_results = []
    
    for result in results:
        try:
            # Classify relevance using LLM
            is_relevant = classify_result_relevance(
                original_query, 
                result["content"][:300],  # Use first 300 chars for efficiency
                model
            )
            
            if is_relevant:
                relevant_results.append(result)
                
        except Exception as e:
            print(f"Error classifying relevance: {e}")
            # Include result if classification fails
            relevant_results.append(result)
    
    # Sort by source priority (user documents first, then static knowledge)
    relevant_results.sort(key=lambda x: (
        0 if x["source"] == "user_documents" else 1,
        -x["relevance_score"]
    ))
    
    return relevant_results[:max_final_results]

def classify_result_relevance(query, content, model):
    """
    Classifies if a search result is relevant to the user's query
    Args: query - original user query, content - result content, model - LLM
    Returns: boolean indicating relevance
    """
    classify_system_message = """Determine if the given content directly answers or relates to the student's query about algorithms/computer science.

Respond with ONLY 'yes' if highly relevant, 'no' if not relevant. No explanations."""

    classify_conversation = [
        {"role": "system", "content": classify_system_message},
        {"role": "user", "content": "Query: quicksort algorithm\nContent: Quicksort is a divide-and-conquer sorting algorithm that works by selecting a pivot..."},
        {"role": "assistant", "content": "yes"},
        {"role": "user", "content": "Query: binary search implementation\nContent: Bubble sort is a simple sorting algorithm that repeatedly steps through the list..."},
        {"role": "assistant", "content": "no"},
        {"role": "user", "content": f"Query: {query}\nContent: {content}"}
    ]
    
    try:
        response = model.invoke(classify_conversation)
        result = (response.content if hasattr(response, 'content') else str(response)).strip().lower()
        return result == 'yes'
    except:
        return True  # Default to including content if classification fails

def federated_search(prompt, static_retriever, user_retriever, model):
    """
    Main federated search function that coordinates multi-source RAG
    Args: prompt - user query, static_retriever - static knowledge retriever, 
          user_retriever - user document retriever, model - LLM
    Returns: comprehensive search results from all sources
    """
    print("🔍 Starting federated search across knowledge sources...")
    
    # Step 1: Classify query intent
    intent = classify_query_intent(prompt, model)
    print(f"📊 Query intent classified as: {intent}")
    
    # Step 2: Generate enhanced queries based on intent
    enhanced_queries = create_enhanced_queries(prompt, model)
    print(f"🎯 Generated {sum(len(v) for v in enhanced_queries.values())} specialized queries")
    
    # Step 3: Search strategy based on intent
    all_results = []
    
    if intent == "document_query":
        # Prioritize user documents for document-specific queries
        if user_retriever:
            user_results = search_user_documents(
                enhanced_queries.get("user_content", [prompt]), 
                user_retriever, 
                max_results=4
            )
            all_results.extend(user_results)
            
        # Add some static knowledge as context
        static_results = search_static_knowledge(
            enhanced_queries.get("conceptual", [prompt])[:2], 
            static_retriever, 
            max_results=2
        )
        all_results.extend(static_results)
        
    else:
        # For other intents, search both sources with appropriate weighting
        
        # Search static knowledge (always available)
        for category, queries in enhanced_queries.items():
            if category != "user_content":
                static_results = search_static_knowledge(
                    queries[:2],  # Limit queries per category
                    static_retriever,
                    max_results=2
                )
                all_results.extend(static_results)
        
        # Search user documents if available
        if user_retriever:
            user_results = search_user_documents(
                enhanced_queries.get("user_content", [prompt]), 
                user_retriever,
                max_results=3
            )
            all_results.extend(user_results)
    
    # Step 4: Rank and filter results
    final_results = rank_and_filter_results(all_results, model, prompt)
    
    print(f"✅ Federated search completed: {len(final_results)} relevant results")
    
    return {
        "results": final_results,
        "intent": intent,
        "query_count": sum(len(v) for v in enhanced_queries.values()),
        "sources_searched": ["static_knowledge"] + (["user_documents"] if user_retriever else [])
    }

def format_search_results_for_prompt(search_results):
    """
    Formats federated search results for LLM prompt
    Args: search_results - results from federated_search
    Returns: formatted context string
    """
    if not search_results["results"]:
        return "No relevant information found in knowledge base."
    
    formatted_context = []
    
    # Group results by source
    static_results = [r for r in search_results["results"] if r["source"] == "static_knowledge"]
    user_results = [r for r in search_results["results"] if r["source"] == "user_documents"]
    
    # Add static knowledge
    if static_results:
        formatted_context.append("=== Algorithm Knowledge Base ===")
        for i, result in enumerate(static_results, 1):
            algorithm = result["metadata"].get("algorithm", "Unknown")
            content_type = result["metadata"].get("type", "information")
            formatted_context.append(f"\n{i}. {algorithm.title()} ({content_type}):")
            formatted_context.append(result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"])
    
    # Add user document results
    if user_results:
        formatted_context.append("\n\n=== Your Study Materials ===")
        for i, result in enumerate(user_results, 1):
            doc_id = result["metadata"].get("document_id", "unknown")
            
            # Get document info for better attribution
            doc_info = get_document_by_id(doc_id)
            source_name = doc_info["filename"] if doc_info else "Unknown Document"
            
            formatted_context.append(f"\n{i}. From '{source_name}':")
            formatted_context.append(result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"])
    
    return "\n".join(formatted_context)

def get_enhanced_response_template(intent):
    """
    Returns response template based on query intent
    Args: intent - classified query intent
    Returns: appropriate ChatPromptTemplate
    """
    templates = {
        "explanation": """You are an expert computer science tutor. Explain the concept clearly and educationally.

Context from knowledge sources:
{context}

Student question: {question}

Provide a comprehensive explanation covering:
1. Core concept and how it works
2. Key characteristics and properties  
3. Real-world applications
4. Common misconceptions to avoid

Make it engaging and easy to understand for a CS student.""",

        "implementation": """You are a coding tutor helping with algorithm implementations.

Context from knowledge sources:
{context}

Student question: {question}

Provide:
1. Clean, well-commented code implementation
2. Explanation of key parts and logic
3. Time and space complexity analysis
4. Usage examples and test cases
5. Common implementation pitfalls to avoid""",

        "practice": """You are creating practice exercises for CS students.

Context from knowledge sources:
{context}

Student request: {question}

Generate 2-3 practice problems that:
1. Progress in difficulty (Easy → Medium → Hard)
2. Include clear problem statements
3. Provide helpful hints for each problem
4. Mention what concepts each problem tests
5. Suggest approaches without giving full solutions""",

        "comparison": """You are helping students understand differences between algorithms/concepts.

Context from knowledge sources:
{context}

Student question: {question}

Provide a clear comparison covering:
1. Key similarities and differences
2. When to use each approach
3. Performance trade-offs
4. Practical examples showing the differences
5. Decision criteria for choosing between them""",

        "document_query": """You are helping a student understand their own study materials.

Context from study materials and knowledge base:
{context}

Student question: {question}

Based on the student's materials and supplementary knowledge:
1. Answer their specific question about the content
2. Provide additional context from the knowledge base if helpful
3. Highlight key points from their materials
4. Suggest related concepts to explore further""",

        "help": """You are a helpful CS tutor providing guidance and support.

Context from knowledge sources:
{context}

Student question: {question}

Provide helpful guidance by:
1. Understanding what the student needs help with
2. Offering clear, actionable advice
3. Suggesting specific resources or commands to try
4. Encouraging continued learning"""
    }
    
    template_text = templates.get(intent, templates["explanation"])
    return ChatPromptTemplate.from_template(template_text)

if __name__ == "__main__":
    # Test federated search functionality
    from langchain_ollama import Ollama
    
    model = Ollama(model="llama3.2")
    
    # Test query intent classification
    test_queries = [
        "How does quicksort work?",
        "Show me Python code for binary search",
        "Give me practice problems for sorting",
        "What did my notes say about trees?",
        "Compare quicksort and mergesort"
    ]
    
    print("Testing query intent classification:")
    for query in test_queries:
        intent = classify_query_intent(query, model)
        print(f"'{query}' → {intent}")
    
    print("\nTesting enhanced query generation:")
    enhanced = create_enhanced_queries("How does quicksort work?", model)
    print(f"Enhanced queries: {enhanced}")