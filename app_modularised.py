from flask import Flask, render_template, request, jsonify
import io, time
from io import BytesIO

from config import get_cerebras_client, call_cerebras_with_retry
from stores import raptor_store, excel_store, chart_store
from embeddings import (build_raptor_tree, create_raptor_index, raptor_retrieve,
                        chunk_text, format_raptor_context)
from excel_utils import parse_excel_file, generate_excel_context_text
from search import search_web
from agent import DataAnalysisAgent
from utils import remove_thinking_tags, format_markdown_response

app = Flask(__name__)

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF upload and RAPTOR tree building"""
    try:
        if 'pdf' not in request.files:
            return jsonify({'success': False, 'error': 'No PDF file provided'})
        
        pdf_file = request.files['pdf']
        
        if pdf_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Extract text from PDF
        text = extract_text_from_pdf(io.BytesIO(pdf_file.read()))
        
        # Chunk the text
        chunks = chunk_text(text)
        
        if not chunks:
            return jsonify({'success': False, 'error': 'No text extracted from PDF'})
        
        # Get Cerebras client for summarization
        try:
            client = get_cerebras_client()
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)})
        
        # Build RAPTOR tree
        indexing_start = time.time()
        tree_nodes, num_levels = build_raptor_tree(chunks, client, max_levels=3)
        
        # Create FAISS index from all nodes
        index = create_raptor_index(tree_nodes)
        indexing_time = round(time.time() - indexing_start, 2)
        
        # Store in memory
        raptor_store['tree'] = tree_nodes
        raptor_store['index'] = index
        raptor_store['filename'] = pdf_file.filename
        raptor_store['levels'] = num_levels
        
        return jsonify({
            'success': True,
            'chunks': len(chunks),
            'total_nodes': len(tree_nodes),
            'levels': num_levels,
            'filename': pdf_file.filename,
            'indexing_time': indexing_time,
            'file_type': 'pdf'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload_excel', methods=['POST'])
def upload_excel():
    """Handle Excel file upload and parsing"""
    try:
        if 'excel' not in request.files:
            return jsonify({'success': False, 'error': 'No Excel file provided'})
        
        excel_file = request.files['excel']
        
        if excel_file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Parse Excel file
        indexing_start = time.time()
        result = parse_excel_file(BytesIO(excel_file.read()))
        
        if not result['success']:
            return jsonify({'success': False, 'error': result['error']})
        
        # Store in memory
        excel_store['dataframes'] = result['dataframes']
        excel_store['metadata'] = result['metadata']
        excel_store['filename'] = excel_file.filename
        
        # Generate text representation for RAPTOR indexing (optional)
        context_text = generate_excel_context_text(result['metadata'])
        excel_store['context_text'] = context_text
        
        # Optionally build RAPTOR tree for Excel metadata
        try:
            client = get_cerebras_client()
            chunks = chunk_text(context_text, chunk_size=500, overlap=50)
            
            if chunks:
                tree_nodes, num_levels = build_raptor_tree(chunks, client, max_levels=2)
                index = create_raptor_index(tree_nodes)
                
                # Merge with existing RAPTOR tree if exists
                if raptor_store['tree']:
                    raptor_store['tree'].extend(tree_nodes)
                    # Rebuild combined index
                    raptor_store['index'] = create_raptor_index(raptor_store['tree'])
                else:
                    raptor_store['tree'] = tree_nodes
                    raptor_store['index'] = index
                
                raptor_store['filename'] = f"{raptor_store.get('filename', '')} + {excel_file.filename}"
        except Exception as e:
            print(f"Warning: Could not build RAPTOR tree for Excel: {e}")
        
        indexing_time = round(time.time() - indexing_start, 2)
        
        return jsonify({
            'success': True,
            'filename': excel_file.filename,
            'sheets': result['metadata']['num_sheets'],
            'sheet_names': result['metadata']['sheet_names'],
            'total_rows': result['metadata']['total_rows'],
            'total_columns': result['metadata']['total_columns'],
            'indexing_time': indexing_time,
            'file_type': 'excel'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handle chat requests with autonomous agent support.
    
    Modes:
    - use_agent=False: Standard RAG with web search
    - use_agent=True: Autonomous multi-step reasoning agent with charting
    """
    try:
        data = request.json
        user_message = data.get('message', '')
        use_web_search = data.get('use_web_search', True)
        use_pdf_context = data.get('use_pdf_context', True)
        use_raptor = data.get('use_raptor', True)
        use_agent = data.get('use_agent', False)
        max_agent_steps = data.get('max_agent_steps', 5)
        
        if not user_message:
            return jsonify({'success': False, 'error': 'No message provided'})
        
        # Get Cerebras client
        try:
            client = get_cerebras_client()
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)})
        
        retrieval_start = time.time()
        pdf_context_used = False
        web_context_used = False
        excel_context_used = bool(excel_store.get('dataframes'))
        
        # Get PDF context using RAPTOR
        pdf_nodes = []
        if use_pdf_context and raptor_store['tree']:
            pdf_nodes = raptor_retrieve(user_message, top_k=16, tree_traverse=use_raptor)
            if pdf_nodes:
                pdf_context_used = True
        
        # AUTONOMOUS AGENT MODE
        if use_agent:
            print(f"\n🤖 Using autonomous data analysis agent mode...")
            
            # Format initial context
            context_parts = []
            
            if pdf_nodes:
                context_parts.append(format_raptor_context(pdf_nodes))
            
            if excel_store.get('dataframes'):
                context_parts.append(f"Excel data loaded: {excel_store.get('filename', 'Unknown')}")
                metadata = excel_store.get('metadata', {})
                context_parts.append(f"Sheets: {', '.join(metadata.get('sheet_names', []))}")
            
            combined_context = "\n".join(context_parts) if context_parts else ""
            
            # Create and run agent
            agent = DataAnalysisAgent(
                client_getter=get_cerebras_client,
                goal=user_message,
                context=combined_context,
                pdf_nodes=pdf_nodes,
                max_steps=max_agent_steps
            )
            
            raw_response, charts = agent.run()
            cleaned_response = remove_thinking_tags(raw_response)
            formatted_response = format_markdown_response(cleaned_response)
            
            retrieval_time = time.time() - retrieval_start
            
            return jsonify({
                'success': True,
                'response': formatted_response,
                'response_markdown': cleaned_response,
                'pdf_context_used': pdf_context_used,
                'web_context_used': False,
                'excel_context_used': excel_context_used,
                'retrieval_time': f"{retrieval_time:.4f}",
                'raptor_enabled': use_raptor,
                'nodes_retrieved': len(pdf_nodes),
                'agent_mode': True,
                'agent_steps': agent.step_count,
                'charts': charts,
                # NEW: Include execution details
                'execution_log': agent.execution_log,
                'thinking_process': agent.thinking_process,
                'raw_responses': agent.raw_responses,
                'memory': agent.memory
            })
        
        # STANDARD RAG MODE
        else:
            # Get web context if enabled
            web_results = []
            if use_web_search:
                web_results = search_web(user_message, num_results=10)
                if web_results:
                    web_context_used = True
            
            # Format context
            context_parts = []
            
            if pdf_nodes:
                context_parts.append(format_raptor_context(pdf_nodes))
            
            if excel_store.get('dataframes'):
                context_parts.append("=== EXCEL DATA ===\n")
                context_parts.append(excel_store.get('context_text', ''))
            
            if web_results:
                web_context = "=== WEB SEARCH RESULTS ===\n\n"
                for i, result in enumerate(web_results, 1):
                    web_context += f"[{i}] {result.get('title', 'No title')}\n"
                    web_context += f"    {result.get('snippet', '')}\n"
                    if result.get('url'):
                        web_context += f"    Source: {result['url']}\n"
                    web_context += "\n"
                context_parts.append(web_context)
            
            combined_context = "\n".join(context_parts) if context_parts else ""
            
            retrieval_time = time.time() - retrieval_start
            
            # Build prompt
            if combined_context:
                system_message = f"""You are a helpful assistant. The user asked: "{user_message}"

Information has been retrieved from multiple sources (including hierarchical document summaries, Excel data, and web search):

{combined_context}

Instructions:
- Answer based on the retrieved information above
- RAPTOR nodes show both SUMMARY (high-level) and DETAIL (specific) information
- Higher level summaries provide context; lower level details provide specifics
- For Excel data, reference the structure and sample data shown
- Cite sources using numbers (e.g., "According to [1]...")
- Be direct and confident - you have current information
"""
            else:
                system_message = "You are a helpful assistant. No specific context was retrieved for this query."
            
            # Call Cerebras API with retry logic
            def make_chat_call(cerebras_client):
                return cerebras_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    model="gpt-oss-120b",
                    max_tokens=4000,
                    temperature=0.7
                )
            
            chat_completion = call_cerebras_with_retry(make_chat_call)
            raw_response = chat_completion.choices[0].message.content

            # Remove thinking tags from the response
            cleaned_response = remove_thinking_tags(raw_response)

            # Convert Markdown to HTML for proper formatting
            formatted_response = format_markdown_response(cleaned_response)

            return jsonify({
                'success': True,
                'response': formatted_response,
                'response_markdown': cleaned_response,
                'pdf_context_used': pdf_context_used,
                'web_context_used': web_context_used,
                'excel_context_used': excel_context_used,
                'retrieval_time': f"{retrieval_time:.4f}",
                'raptor_enabled': use_raptor,
                'nodes_retrieved': len(pdf_nodes),
                'agent_mode': False,
                'charts': []  # No charts in standard mode
            })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
