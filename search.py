import os, requests
from typing import List, Dict

def get_langsearch_api_key():
    """Get LangSearch API key"""
    api_key = os.environ.get("LANGSEARCH_API_KEY")
    if not api_key:
        try:
            with open('langsearch_config.txt', 'r') as f:
                api_key = f.read().strip()
        except:
            pass
    return api_key

def search_web(query: str, num_results: int = 10) -> List[Dict]:
    """Search the web using available APIs with fallback"""
    
    langsearch_key = get_langsearch_api_key()
    if langsearch_key:
        try:
            url = "https://api.langsearch.com/v1/web-search"
            payload = {
                "query": query,
                "freshness": "noLimit",
                "summary": True,
                "count": num_results
            }
            headers = {
                'Authorization': f'Bearer {langsearch_key}',
                'Content-Type': 'application/json'
            }
            
            print(f"🔍 Searching LangSearch for: {query}")
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                results = []
                api_data = data.get('data', {})
                web_pages = api_data.get('webPages', {})
                search_results = web_pages.get('value', [])
                
                for r in search_results[:num_results]:
                    if isinstance(r, dict):
                        results.append({
                            'title': r.get('name', r.get('title', '')),
                            'snippet': r.get('snippet', r.get('content', r.get('description', ''))),
                            'url': r.get('url', r.get('link', '')),
                            'summary': r.get('summary', '')
                        })
                
                if results:
                    print(f"✅ Web search via LangSearch: {len(results)} results")
                    return results
        except Exception as e:
            print(f"❌ LangSearch search failed: {str(e)}")
    
    # DuckDuckGo fallback
    try:
        print("🔍 Trying DuckDuckGo fallback...")
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=num_results):
                results.append({
                    'title': r.get('title', ''),
                    'snippet': r.get('body', ''),
                    'url': r.get('href', ''),
                    'summary': ''
                })
        
        if results:
            print(f"✅ Web search via DuckDuckGo: {len(results)} results")
            return results
    except Exception as e:
        print(f"❌ DuckDuckGo search failed: {str(e)}")
    
    print("❌ No web search API available")
    return []
