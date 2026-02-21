import pandas as pd
import numpy as np
from typing import Dict, Any
from stores import excel_store

def parse_excel_file(excel_file) -> Dict[str, Any]:
    """
    Parse Excel file and extract all sheets as DataFrames.
    Returns dict with metadata and dataframes.
    """
    try:
        # Read all sheets
        excel_data = pd.read_excel(excel_file, sheet_name=None, engine='openpyxl')
        
        metadata = {
            'num_sheets': len(excel_data),
            'sheet_names': list(excel_data.keys()),
            'total_rows': sum(df.shape[0] for df in excel_data.values()),
            'total_columns': sum(df.shape[1] for df in excel_data.values()),
        }
        
        # Generate summary for each sheet
        sheet_summaries = {}
        for sheet_name, df in excel_data.items():
            summary = {
                'rows': df.shape[0],
                'columns': df.shape[1],
                'column_names': df.columns.tolist(),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'sample_data': df.head(3).to_dict('records') if len(df) > 0 else [],
                'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
                'text_columns': df.select_dtypes(include=['object']).columns.tolist(),
            }
            
            # Add basic statistics for numeric columns
            if summary['numeric_columns']:
                summary['statistics'] = df[summary['numeric_columns']].describe().to_dict()
            
            sheet_summaries[sheet_name] = summary
        
        metadata['sheet_summaries'] = sheet_summaries
        
        return {
            'dataframes': excel_data,
            'metadata': metadata,
            'success': True
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def generate_excel_context_text(metadata: Dict) -> str:
    """Generate text description of Excel file for embedding and retrieval"""
    context_parts = []
    
    context_parts.append(f"Excel file with {metadata['num_sheets']} sheet(s)")
    context_parts.append(f"Total: {metadata['total_rows']} rows, {metadata['total_columns']} columns")
    
    for sheet_name, summary in metadata['sheet_summaries'].items():
        context_parts.append(f"\nSheet: {sheet_name}")
        context_parts.append(f"  Dimensions: {summary['rows']} rows × {summary['columns']} columns")
        context_parts.append(f"  Columns: {', '.join(summary['column_names'])}")
        
        if summary['numeric_columns']:
            context_parts.append(f"  Numeric columns: {', '.join(summary['numeric_columns'])}")
        
        if summary['text_columns']:
            context_parts.append(f"  Text columns: {', '.join(summary['text_columns'])}")
        
        # Add sample data
        if summary['sample_data']:
            context_parts.append("  Sample rows:")
            for i, row in enumerate(summary['sample_data'][:2], 1):
                context_parts.append(f"    Row {i}: {str(row)[:200]}")
    
    return "\n".join(context_parts)
