from config import call_cerebras_with_retry
from stores import raptor_store, excel_store
from embeddings import raptor_retrieve
from charts import create_matplotlib_chart, create_plotly_chart
from utils import remove_thinking_tags
from models import RAPTORNode

import os
from flask import Flask, render_template, request, jsonify
from cerebras.cloud.sdk import Cerebras
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import io
from concurrent.futures import ThreadPoolExecutor
import time
import requests
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass, field
import re
import markdown
import json
import contextlib
import pandas as pd
import openpyxl
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64

# ============================================================================
# AUTONOMOUS AGENT CLASSES AND FUNCTIONS
# ============================================================================

class DataAnalysisAgent:
    """
    Enhanced autonomous agent with Excel data analysis and charting capabilities.
    
    Can:
    - Retrieve RAPTOR nodes dynamically
    - Execute safe Python code on in-memory data
    - Analyze Excel data with pandas operations
    - Generate visualizations (matplotlib and plotly)
    - Perform multi-step reasoning
    """
    
    def __init__(self, client_getter, goal: str, context: str, pdf_nodes: List[RAPTORNode], max_steps=5):
        """
        Args:
            client_getter: Function that returns a Cerebras client (for retry logic)
            goal: User's question/goal
            context: Initial RAPTOR context
            pdf_nodes: Initial set of retrieved RAPTOR nodes
            max_steps: Maximum reasoning steps
        """
        self.client_getter = client_getter
        self.goal = goal
        self.context = context
        self.max_steps = max_steps
        self.memory = []
        self.step_count = 0
        self.charts_created = []  # Track charts created in this session

        # ADD THESE THREE LINES:
        self.execution_log = []  # List of {step, type, content, result, timestamp}
        self.thinking_process = []  # Track raw LLM thinking
        self.raw_responses = []  # Track raw LLM outputs
        
        # Local environment for code execution - includes Excel data
        self.local_env = {
            "raptor_nodes": pdf_nodes,
            "excel_data": excel_store.get('dataframes', {}),
            "excel_metadata": excel_store.get('metadata', {}),
            "np": np,
            "pd": pd,
            "plt": plt,
            "go": go,
            "px": px,
            # Safe built-in functions
            "len": len,
            "sum": sum,
            "max": max,
            "min": min,
            "sorted": sorted,
            "list": list,
            "dict": dict,
            "set": set,
            "str": str,
            "int": int,
            "float": float,
            "round": round,
            "abs": abs,
            "enumerate": enumerate,
            "zip": zip,
            "range": range,
            "type": type,
            "isinstance": isinstance,
        }
    
    def retrieve_nodes_tool(self, query: str, top_k: int = 5) -> str:
        """Tool: Retrieve RAPTOR nodes dynamically"""
        try:
            exec_start = time.time()
            nodes = raptor_retrieve(query, top_k=top_k, tree_traverse=True)
            exec_time = time.time() - exec_start
            
            if not nodes:
                result = "No relevant RAPTOR nodes found."
            else:
                result = f"Retrieved {len(nodes)} nodes:\n"
                for i, n in enumerate(nodes, 1):
                    result += f"[{i}] Level {n.level}: {n.text[:200]}...\n"
                
                # Update local environment with new nodes
                self.local_env['latest_retrieved_nodes'] = nodes
            
            # NEW: Log retrieval
            self.execution_log.append({
                'step': self.step_count,
                'type': 'retrieve_nodes',
                'query': query,
                'top_k': top_k,
                'result': result,
                'nodes_found': len(nodes),
                'success': len(nodes) > 0,
                'execution_time': exec_time,
                'timestamp': time.time()
            })
            
            return result
        except Exception as e:
            return f"Error retrieving nodes: {str(e)}"
    
    def run_code_tool(self, code_snippet: str) -> str:
        """Tool: Execute Python code safely on in-memory data including Excel"""
        try:
            # Clean up the code snippet
            code_snippet = code_snippet.strip()
            
            # Remove any markdown code block markers if present
            if code_snippet.startswith('```python'):
                code_snippet = code_snippet.replace('```python', '').replace('```', '').strip()
            elif code_snippet.startswith('```'):
                code_snippet = code_snippet.replace('```', '').strip()
            
            # Validate f-strings don't have unmatched brackets
            if 'f"' in code_snippet or "f'" in code_snippet:
                # Count brackets in f-strings
                open_count = code_snippet.count('{')
                close_count = code_snippet.count('}')
                # If unmatched, this will likely fail - provide guidance
                if open_count != close_count:
                    return f"F-string syntax error: Unmatched brackets ({{ vs }}). Found {open_count} '{{' and {close_count} '}}'. Please balance your brackets."
            
            # Create safe execution environment with more built-ins
            allowed_builtins = {
                "len": len, "sum": sum, "max": max, "min": min, 
                "sorted": sorted, "list": list, "dict": dict,
                "set": set, "str": str, "int": int, "float": float,
                "range": range, "enumerate": enumerate, "zip": zip,
                "abs": abs, "round": round, "print": print, "type": type,
                "isinstance": isinstance, "bool": bool, "tuple": tuple,
                "any": any, "all": all, "map": map, "filter": filter,
            }
            
            exec_globals = {
                "__builtins__": allowed_builtins,
                "np": np,
                "pd": pd,
                "plt": plt,
                "go": go,
                "px": px,
            }
            exec_locals = dict(self.local_env)
            
            # Capture print output
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(code_snippet, exec_globals, exec_locals)
            
            output = buf.getvalue()
            
            # Update local environment with new variables (exclude internal variables)
            for k, v in exec_locals.items():
                if k not in exec_globals and not k.startswith('_') and k not in self.local_env:
                    # Only add if it's a simple type or pandas/numpy object
                    if isinstance(v, (int, float, str, list, dict, set, tuple, pd.DataFrame, pd.Series, np.ndarray)):
                        self.local_env[k] = v
            
            # If no output, try to show the last variable created
            if not output:
                # Find newly created variables
                new_vars = [k for k in exec_locals.keys() 
                           if k not in self.local_env and not k.startswith('_') and k not in exec_globals]
                if new_vars:
                    last_var = new_vars[-1]
                    value = exec_locals[last_var]
                    output = f"{last_var} = {value}\n"
                else:
                    output = "Code executed successfully (no output)."
            
            return output.strip()
        
        except SyntaxError as e:
            return f"Syntax Error: {str(e)}\nPlease check your code syntax. Make sure quotes are balanced and statements are complete."
        except NameError as e:
            return f"Name Error: {str(e)}\nAvailable variables: {', '.join([k for k in self.local_env.keys() if not k.startswith('_')])}"
        except KeyError as e:
            # Provide helpful info about available sheets/columns
            if 'excel_data' in self.local_env:
                sheets = list(self.local_env['excel_data'].keys())
                return f"Key Error: {str(e)}\nAvailable Excel sheets: {sheets}"
            return f"Key Error: {str(e)}"
        except Exception as e:
            return f"Error: {type(e).__name__}: {str(e)}"
    
    def create_chart_tool(self, chart_type: str, code: str) -> str:
        """
        Tool: Create a chart (matplotlib or plotly).
        
        Args:
            chart_type: 'matplotlib' or 'plotly'
            code: Python code to generate the chart
        
        Returns:
            Status message with chart_id
        """
        try:
            if chart_type == 'matplotlib':
                result = create_matplotlib_chart(code)
                if result['success']:
                    chart_id = result['chart_id']
                    self.charts_created.append({
                        'id': chart_id,
                        'type': 'matplotlib',
                        'image_b64': result['image_b64']
                    })
                    return f"✅ Chart created successfully! Chart ID: {chart_id}"
                else:
                    return f"❌ Chart creation failed: {result['error']}"
            
            elif chart_type == 'plotly':
                result = create_plotly_chart(code)
                if result['success']:
                    chart_id = result['chart_id']
                    self.charts_created.append({
                        'id': chart_id,
                        'type': 'plotly',
                        'figure_json': result['figure_json']
                    })
                    return f"✅ Chart created successfully! Chart ID: {chart_id}"
                else:
                    return f"❌ Chart creation failed: {result['error']}"
            
            else:
                return f"❌ Unknown chart type: {chart_type}. Use 'matplotlib' or 'plotly'."
        
        except Exception as e:
            return f"❌ Error creating chart: {str(e)}"
    
    def analyze_excel_tool(self, analysis_type: str = "summary") -> str:
        """Tool: Quick Excel data analysis"""
        try:
            if not excel_store.get('dataframes'):
                return "No Excel data loaded."
            
            results = []
            
            if analysis_type == "summary":
                results.append("=== EXCEL DATA SUMMARY ===")
                results.append(f"File: {excel_store.get('filename', 'Unknown')}")
                metadata = excel_store.get('metadata', {})
                results.append(f"Sheets: {metadata.get('num_sheets', 0)}")
                results.append(f"Total Rows: {metadata.get('total_rows', 0)}")
                results.append("")
                
                for sheet_name, df in excel_store['dataframes'].items():
                    results.append(f"Sheet: {sheet_name}")
                    results.append(f"  Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                    results.append(f"  Columns: {', '.join(df.columns.tolist()[:10])}")
                    
                    # Show data types
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        results.append(f"  Numeric columns: {', '.join(numeric_cols[:5])}")
                    
                    # Show sample
                    if len(df) > 0:
                        results.append(f"  Sample (first row):")
                        sample = df.iloc[0].to_dict()
                        for k, v in list(sample.items())[:5]:
                            results.append(f"    {k}: {v}")
                    results.append("")
            
            elif analysis_type == "statistics":
                results.append("=== STATISTICAL ANALYSIS ===")
                for sheet_name, df in excel_store['dataframes'].items():
                    results.append(f"\nSheet: {sheet_name}")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_cols) > 0:
                        stats = df[numeric_cols].describe()
                        # Format nicely
                        results.append(f"Numeric columns: {', '.join(numeric_cols.tolist())}")
                        results.append("\nStatistics (mean, std, min, max):")
                        for col in numeric_cols[:3]:  # Limit to first 3 columns
                            results.append(f"  {col}:")
                            results.append(f"    Mean: {df[col].mean():.2f}")
                            results.append(f"    Std: {df[col].std():.2f}")
                            results.append(f"    Min: {df[col].min():.2f}")
                            results.append(f"    Max: {df[col].max():.2f}")
                    else:
                        results.append("  No numeric columns found")
            
            elif analysis_type == "columns":
                results.append("=== COLUMN INFORMATION ===")
                for sheet_name, df in excel_store['dataframes'].items():
                    results.append(f"\nSheet: {sheet_name}")
                    results.append(f"Columns ({len(df.columns)}):")
                    for col in df.columns:
                        dtype = df[col].dtype
                        non_null = df[col].notna().sum()
                        results.append(f"  - {col} ({dtype}): {non_null} non-null values")
            
            result_text = "\n".join(results)
            exec_time = time.time() - exec_start
            
            # NEW: Log analysis
            self.execution_log.append({
                'step': self.step_count,
                'type': 'analyze_excel',
                'analysis_type': analysis_type,
                'result': result_text[:500] + '...' if len(result_text) > 500 else result_text,
                'success': True,
                'execution_time': exec_time,
                'timestamp': time.time()
            })
            
            return result_text
        
        except Exception as e:
            return f"Error analyzing Excel: {str(e)}"
    
    def step(self) -> Dict:
        """Execute one reasoning step"""
        self.step_count += 1
        
        # Build memory context
        memory_str = "\n".join([f"Step {i+1}: {m}" for i, m in enumerate(self.memory)])
        
        # Check if Excel data is available
        has_excel = bool(excel_store.get('dataframes'))
        excel_context = ""
        if has_excel:
            metadata = excel_store.get('metadata', {})
            excel_context = f"""
EXCEL DATA AVAILABLE:
- File: {excel_store.get('filename', 'Unknown')}
- Sheets: {', '.join(metadata.get('sheet_names', []))}
- Access via: excel_data[sheet_name] (returns pandas DataFrame)
- Metadata: excel_metadata (contains column info, statistics, etc.)
"""
        
        prompt = f"""You are an autonomous data analysis agent with access to tools.

GOAL: {self.goal}

CURRENT CONTEXT:
{self.context}

{excel_context}

MEMORY (Previous Steps):
{memory_str if memory_str else "No previous steps yet."}

AVAILABLE TOOLS:
1. retrieve_nodes(query, top_k) - Retrieve relevant document nodes from RAPTOR tree
   Example: {{"tool": "retrieve_nodes", "query": "machine learning algorithms", "top_k": 5}}

2. run_code(code) - Execute Python code on in-memory data
   Available: excel_data (dict of DataFrames), pd (pandas), np (numpy), plt (matplotlib), go/px (plotly)
   IMPORTANT CODE RULES:
   - Use simple, single-line or short multi-line statements
   - Always use print() to display results
   - Access Excel sheets: df = excel_data['Sheet1']
   - No imports allowed (pd, np, plt, go, px already available)
   - Avoid line continuation characters (\\)
   Example: {{"tool": "run_code", "code": "df = excel_data['Sheet1']; print(df.columns.tolist())"}}
   Example: {{"tool": "run_code", "code": "df = excel_data['Sheet1']; result = df['Revenue'].sum(); print(f'Total: {{result}}')"}}

3. create_chart(chart_type, code) - Create visualizations
   chart_type: "matplotlib" or "plotly"
   
   MATPLOTLIB EXAMPLE:
   {{"tool": "create_chart", "chart_type": "matplotlib", "code": "df = excel_data['Sheet1']; plt.figure(figsize=(10,6)); plt.bar(df['Category'], df['Sales']); plt.title('Sales by Category'); plt.xlabel('Category'); plt.ylabel('Sales')"}}
   
   PLOTLY EXAMPLE:
   {{"tool": "create_chart", "chart_type": "plotly", "code": "df = excel_data['Sheet1']; fig = px.bar(df, x='Category', y='Sales', title='Sales by Category')"}}
   
   IMPORTANT: Charts will be displayed to the user automatically. Create charts when visualization helps answer the question.

4. analyze_excel(analysis_type) - Quick Excel analysis ("summary" or "statistics")
   Example: {{"tool": "analyze_excel", "analysis_type": "summary"}}

INSTRUCTIONS:
- Start with analyze_excel to understand the data structure
- Use run_code for specific calculations - keep code simple
- Create charts when visualization helps answer the question
- For complex analysis, break into multiple small run_code steps
- Always use print() to see results
- When ready to answer, use: {{"action": "final_answer", "content": "your answer here"}}
- Always respond with valid JSON only

Current step: {self.step_count}/{self.max_steps}

Respond with JSON:"""

        # Call LLM
        def make_api_call(client):
            return client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an autonomous data analysis agent. Always respond with valid JSON.

🚨 CRITICAL RULE: Output ONLY ONE JSON object per response. Execute ONE action at a time.

IMPORTANT CODING GUIDELINES:
1. Keep code simple - use semicolons to separate statements on one line
2. Always use print() to display results
3. No imports - pd (pandas), np (numpy), plt (matplotlib), go/px (plotly) are already available
4. Access Excel data: excel_data['SheetName']
5. Check available sheets first: print(list(excel_data.keys()))
6. For f-strings, be careful with brackets - use double braces {{}} to escape

CHART CREATION GUIDELINES:
1. Use create_chart tool for visualizations
2. For matplotlib: Create figure, plot, and set labels in one code string
3. For plotly: Create fig variable with go or px
4. Charts are automatically shown to user - mention this in your final answer
5. Create charts when they help answer the question

WORKFLOW - ONE STEP AT A TIME:
Step 1: {{"tool": "analyze_excel", "analysis_type": "summary"}}
Step 2: {{"tool": "run_code", "code": "..."}}
Step 3: {{"tool": "create_chart", "chart_type": "matplotlib", "code": "..."}}
Step 4: {{"action": "final_answer", "content": "..."}}

GOOD CODE EXAMPLES:
{{"tool": "run_code", "code": "sheets = list(excel_data.keys()); print(sheets)"}}
{{"tool": "run_code", "code": "df = excel_data['Sheet1']; print(df.shape)"}}
{{"tool": "run_code", "code": "df = excel_data['Sheet1']; total = df['Revenue'].sum(); print('Total:', total)"}}

GOOD CHART EXAMPLES:
{{"tool": "create_chart", "chart_type": "matplotlib", "code": "df = excel_data['Sheet1']; plt.figure(figsize=(8,5)); plt.plot(df['Date'], df['Value']); plt.title('Trend Over Time'); plt.xlabel('Date'); plt.ylabel('Value')"}}
{{"tool": "create_chart", "chart_type": "plotly", "code": "df = excel_data['Sheet1']; fig = px.line(df, x='Date', y='Value', title='Interactive Trend')"}}

❌ WRONG - Multiple JSON objects:
{{"tool": "run_code", ...}}
{{"tool": "create_chart", ...}}
{{"action": "final_answer", ...}}

✅ CORRECT - One JSON object:
{{"tool": "create_chart", "chart_type": "matplotlib", "code": "..."}}

AVOID:
- Multiple JSON objects in one response
- Multi-line code with line continuations (\\)
- Code without print() statements
- Trying to import modules
- Complex f-strings with unmatched brackets

For final answers, use simple text and mention any charts created."""
                    },
                    {"role": "user", "content": prompt}
                ],
                model="gpt-oss-120b",
                max_tokens=2000,
                temperature=0.3
            )
        
        try:
            api_resp = call_cerebras_with_retry(make_api_call)
            raw_text = api_resp.choices[0].message.content
            
            # NEW: Capture raw thinking (before cleaning)
            thinking_content = ""
            if "<think>" in raw_text or "<Think>" in raw_text:
                # Extract thinking tags content
                think_matches = re.findall(r'<think>(.*?)</think>', raw_text, re.DOTALL | re.IGNORECASE)
                if think_matches:
                    thinking_content = "\n\n".join(think_matches)
                    self.thinking_process.append({
                        'step': self.step_count,
                        'thinking': thinking_content,
                        'timestamp': time.time()
                    })
            
            # Store raw response
            self.raw_responses.append({
                'step': self.step_count,
                'raw_text': raw_text[:1000],  # Limit to first 1000 chars
                'has_thinking': bool(thinking_content)
            })
            
            cleaned_text = remove_thinking_tags(raw_text)
            
            print(f"\n📥 Agent raw response:\n{cleaned_text[:500]}\n")
            
            # Extract JSON (handle markdown code blocks and multiple JSON objects)
            json_text = cleaned_text
            if "```json" in json_text:
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif "```" in json_text:
                json_text = json_text.split("```")[1].split("```")[0].strip()
            
            # Handle multiple JSON objects on separate lines (common LLM output)
            # Try to find the FIRST valid JSON object
            lines = json_text.strip().split('\n')
            action_data = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        action_data = json.loads(line)
                        print(f"✅ Parsed JSON: {action_data}")
                        break  # Use the first valid JSON we find
                    except json.JSONDecodeError:
                        continue
            
            # If no valid JSON found in lines, try the whole text
            if action_data is None:
                action_data = json.loads(json_text)
                print(f"✅ Parsed JSON (whole text): {action_data}")
            
            # Execute tool if requested
            if "tool" in action_data:
                tool_name = action_data["tool"]
                
                if tool_name == "retrieve_nodes":
                    query = action_data.get("query", "")
                    top_k = action_data.get("top_k", 5)
                    result = self.retrieve_nodes_tool(query, top_k)
                    self.memory.append(f"Tool: retrieve_nodes('{query}') → {result[:300]}...")
                    return {"type": "tool_use", "tool": tool_name, "result": result}
                
                elif tool_name == "run_code":
                    code = action_data.get("code", "")
                    
                    # NEW: Log code execution attempt
                    exec_start = time.time()
                    result = self.run_code_tool(code)
                    exec_time = time.time() - exec_start
                    
                    # NEW: Record execution log
                    self.execution_log.append({
                        'step': self.step_count,
                        'type': 'code_execution',
                        'code': code,
                        'result': result,
                        'success': 'Error' not in result and 'error' not in result.lower() and 'Syntax' not in result,
                        'execution_time': exec_time,
                        'timestamp': time.time()
                    })
                    
                    self.memory.append(f"Tool: run_code(...) → {result[:300]}...")
                    return {"type": "tool_use", "tool": tool_name, "result": result}
                
                elif tool_name == "create_chart":
                    chart_type = action_data.get("chart_type", "matplotlib")
                    code = action_data.get("code", "")
                    
                    # NEW: Log chart creation attempt
                    exec_start = time.time()
                    result = self.create_chart_tool(chart_type, code)
                    exec_time = time.time() - exec_start
                    
                    # NEW: Record execution log
                    self.execution_log.append({
                        'step': self.step_count,
                        'type': 'chart_creation',
                        'chart_type': chart_type,
                        'code': code,
                        'result': result,
                        'success': '✅' in result,
                        'execution_time': exec_time,
                        'timestamp': time.time()
                    })
                    
                    self.memory.append(f"Tool: create_chart({chart_type}) → {result}")
                    return {"type": "tool_use", "tool": tool_name, "result": result}
                
                elif tool_name == "analyze_excel":
                    analysis_type = action_data.get("analysis_type", "summary")
                    # Support multiple analysis types
                    if analysis_type not in ["summary", "statistics", "columns"]:
                        analysis_type = "summary"
                    result = self.analyze_excel_tool(analysis_type)
                    self.memory.append(f"Tool: analyze_excel('{analysis_type}') → {result[:300]}...")
                    return {"type": "tool_use", "tool": tool_name, "result": result}
                
                else:
                    error_msg = f"Unknown tool: {tool_name}"
                    self.memory.append(error_msg)
                    return {"type": "error", "message": error_msg}
            
            # Handle different actions
            elif "action" in action_data:
                action_type = action_data["action"]
                content = action_data.get("content", "")
                
                if action_type == "final_answer":
                    return {"type": "final_answer", "content": content}
                else:
                    self.memory.append(f"Action: {action_type} - {content[:200]}...")
                    return {"type": "intermediate", "action": action_type, "content": content}
            
            else:
                # Fallback: treat as intermediate reasoning
                self.memory.append(f"Reasoning: {cleaned_text[:200]}...")
                return {"type": "intermediate", "content": cleaned_text}
        
        except json.JSONDecodeError as e:
            error_msg = f"JSON parsing error: {str(e)}"
            self.memory.append(error_msg)
            
            # Try to extract any useful information from the text
            if "final" in cleaned_text.lower() or "answer" in cleaned_text.lower():
                # Treat as final answer attempt
                return {"type": "final_answer", "content": cleaned_text}
            
            return {"type": "error", "message": error_msg, "raw": cleaned_text[:500]}
        
        except Exception as e:
            error_msg = f"Step error: {str(e)}"
            self.memory.append(error_msg)
            return {"type": "error", "message": error_msg}
    
    def run(self) -> Tuple[str, List[Dict]]:
        """
        Run the agent until completion or max steps.
        
        Returns:
            Tuple of (answer_text, list_of_charts)
        """
        print(f"\n🤖 Starting autonomous agent (max {self.max_steps} steps)...")
        
        for step_num in range(self.max_steps):
            print(f"   Step {step_num + 1}/{self.max_steps}...")
            step_result = self.step()
            
            if step_result["type"] == "final_answer":
                print(f"✅ Agent completed in {step_num + 1} steps")
                return step_result["content"], self.charts_created
            
            elif step_result["type"] == "error":
                print(f"⚠️  Error at step {step_num + 1}: {step_result.get('message', 'Unknown error')}")
                # Continue to next step rather than failing completely
        
        # Max steps reached - generate answer from what we learned
        print(f"⚠️  Agent reached max steps ({self.max_steps}), generating summary...")
        
        # Try to generate a coherent answer from memory
        summary_prompt = f"""Based on the analysis performed, provide a final answer to: {self.goal}

Analysis performed:
{chr(10).join(self.memory[-5:])}

Provide a clear, direct answer based on the analysis results above. If specific numbers were calculated, include them. If charts were created, mention them. Be concise."""

        def make_summary_call(client):
            return client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a data analyst. Provide clear, concise answers based on analysis results."},
                    {"role": "user", "content": summary_prompt}
                ],
                model="gpt-oss-120b",
                max_tokens=1000,
                temperature=0.3
            )
        
        try:
            summary_response = call_cerebras_with_retry(make_summary_call)
            summary_text = summary_response.choices[0].message.content
            summary_clean = remove_thinking_tags(summary_text)
            return summary_clean, self.charts_created
        except Exception as e:
            print(f"❌ Could not generate summary: {e}")
            # Fallback: return raw analysis
            return f"Analysis Results:\n\n" + "\n\n".join(self.memory[-3:]), self.charts_created
