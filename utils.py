import re, markdown

def remove_thinking_tags(text: str) -> str:
    """
    Remove <think>...</think> tags and their content from Qwen's output.
    Handles multiple thinking blocks and nested content.
    """
    # Remove <think>...</think> blocks (non-greedy, case-insensitive, multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Also handle self-closing or malformed tags
    cleaned = re.sub(r'<think\s*/>', '', cleaned, flags=re.IGNORECASE)
    
    # Clean up any extra whitespace left behind
    cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Multiple newlines to double
    cleaned = cleaned.strip()
    
    return cleaned

def format_markdown_response(text: str) -> str:
    """
    Convert Markdown formatting to HTML for proper display.
    Handles bold, italic, code, headers, lists, etc.
    Preserves LaTeX math expressions by protecting them before markdown processing.
    """
    import uuid
    
    # Dictionary to store protected content
    protected = {}
    
    def protect_content(match):
        """Replace content with a unique placeholder"""
        key = f"PROTECTED_{uuid.uuid4().hex}"
        protected[key] = match.group(0)
        return key
    
    # Protect display math $$...$$ (must come before inline math)
    text = re.sub(r'\$\$(.+?)\$\$', protect_content, text, flags=re.DOTALL)
    
    # Protect inline math $...$
    # Use negative lookbehind/lookahead to avoid matching $$ 
    text = re.sub(r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)', protect_content, text, flags=re.DOTALL)
    
    # Protect code blocks ```...```
    text = re.sub(r'```[\s\S]+?```', protect_content, text, flags=re.DOTALL)
    
    # Protect inline code `...`
    text = re.sub(r'`[^`\n]+?`', protect_content, text)
    
    # Now process markdown
    html = markdown.markdown(
        text,
        extensions=[
            'fenced_code',
            'nl2br',
            'tables',
            'codehilite'
        ]
    )
    
    # Restore all protected content
    for key, value in protected.items():
        html = html.replace(key, value)
    
    return html
