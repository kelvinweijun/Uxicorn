import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64, numpy as np, pandas as pd
from io import BytesIO
from stores import excel_store, chart_store
from typing import Dict, Any

# ============================================================================
# CHART GENERATION FUNCTIONS
# ============================================================================

def create_matplotlib_chart(code: str) -> Dict[str, Any]:
    """
    Execute matplotlib plotting code and return base64 encoded image.
    
    Returns:
        Dict with 'success', 'image_b64', 'chart_id', 'error'
    """
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Execute the plotting code
        exec_globals = {
            'plt': plt,
            'np': np,
            'pd': pd,
            'excel_data': excel_store.get('dataframes', {}),
        }
        exec(code, exec_globals)
        
        # Save to buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Encode to base64
        image_b64 = base64.b64encode(buf.read()).decode('utf-8')
        
        # Clear plot
        plt.clf()
        plt.close('all')
        
        # Store chart
        chart_store['counter'] += 1
        chart_id = f"chart_{chart_store['counter']}"
        chart_store['charts'].append({
            'id': chart_id,
            'type': 'matplotlib',
            'data': image_b64
        })
        
        return {
            'success': True,
            'image_b64': image_b64,
            'chart_id': chart_id
        }
    
    except Exception as e:
        plt.clf()
        plt.close('all')
        return {
            'success': False,
            'error': str(e)
        }

def create_plotly_chart(code: str) -> Dict[str, Any]:
    """
    Execute plotly plotting code and return JSON figure.
    
    Returns:
        Dict with 'success', 'figure_json', 'chart_id', 'error'
    """
    try:
        # Execute the plotting code
        exec_globals = {
            'go': go,
            'px': px,
            'make_subplots': make_subplots,
            'pd': pd,
            'np': np,
            'excel_data': excel_store.get('dataframes', {}),
        }
        exec_locals = {}
        exec(code, exec_globals, exec_locals)
        
        # Find the figure object (usually named 'fig')
        fig = exec_locals.get('fig')
        if fig is None:
            # Try to find any plotly figure object
            for var_name, var_value in exec_locals.items():
                if isinstance(var_value, (go.Figure, go.FigureWidget)):
                    fig = var_value
                    break
        
        if fig is None:
            return {
                'success': False,
                'error': 'No plotly figure found. Make sure to create a figure named "fig".'
            }
        
        # Convert to JSON
        figure_json = fig.to_json()
        
        # Store chart
        chart_store['counter'] += 1
        chart_id = f"chart_{chart_store['counter']}"
        chart_store['charts'].append({
            'id': chart_id,
            'type': 'plotly',
            'data': figure_json
        })
        
        return {
            'success': True,
            'figure_json': figure_json,
            'chart_id': chart_id
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }
