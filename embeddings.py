from sentence_transformers import SentenceTransformer
import numpy as np, faiss
from sklearn.cluster import KMeans
from typing import List, Tuple
from models import RAPTORNode
from stores import raptor_store
from config import call_cerebras_with_retry
from utils import remove_thinking_tags
from cerebras.cloud.sdk import Cerebras

embedding_model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
embedding_dim = 768

def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks (leaf nodes)"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def summarize_cluster(texts: List[str], client: Cerebras = None) -> str:
    """Use Cerebras LLM to generate abstractive summary of clustered texts"""
    combined = "\n\n---\n\n".join(texts[:5])
    
    prompt = f"""Summarize the following text chunks into a coherent, comprehensive summary. 
Capture the main ideas, key details, and important information. Be concise but thorough.

TEXT CHUNKS:
{combined}

SUMMARY:"""
    
    def make_api_call(cerebras_client):
        response = cerebras_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert at creating concise, informative summaries."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-oss-120b",
            max_tokens=1000,
            temperature=0.3
        )
        return response.choices[0].message.content
    
    try:
        raw_content = call_cerebras_with_retry(make_api_call)
        cleaned_content = remove_thinking_tags(raw_content)
        return cleaned_content
    except Exception as e:
        print(f"❌ Summarization error: {e}")
        return " ".join([t.split('.')[0] + '.' for t in texts[:3]])

def build_raptor_tree(chunks: List[str], client: Cerebras, max_levels: int = 3) -> Tuple[List[RAPTORNode], int]:
    """
    Build RAPTOR tree using recursive clustering and summarization.
    
    Algorithm:
    1. Start with leaf nodes (original chunks)
    2. Embed all nodes
    3. Cluster similar nodes using K-means
    4. Summarize each cluster to create parent nodes
    5. Repeat until we have < threshold nodes or reach max_levels
    """
    print(f"\n🌳 Building RAPTOR tree from {len(chunks)} chunks...")
    
    all_nodes = []
    node_counter = 0
    
    # Level 0: Create leaf nodes (original chunks)
    print(f"   Level 0: Creating {len(chunks)} leaf nodes...")
    current_level_nodes = []
    for chunk in chunks:
        embedding = embedding_model.encode([chunk])[0]
        node = RAPTORNode(
            text=chunk,
            embedding=embedding,
            level=0,
            children=None,
            node_id=node_counter
        )
        current_level_nodes.append(node)
        all_nodes.append(node)
        node_counter += 1
    
    # Store leaf nodes separately
    raptor_store['leaf_nodes'] = current_level_nodes
    
    # Recursively build higher levels
    current_level = 0
    while current_level < max_levels and len(current_level_nodes) > 5:
        current_level += 1
        print(f"   Level {current_level}: Clustering {len(current_level_nodes)} nodes...")
        
        # Extract embeddings
        embeddings = np.array([node.embedding for node in current_level_nodes])
        
        # Determine number of clusters (reduce by ~3x each level)
        n_clusters = max(3, len(current_level_nodes) // 3)
        n_clusters = min(n_clusters, len(current_level_nodes))
        
        # Cluster nodes using K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group nodes by cluster
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(current_level_nodes[idx])
        
        print(f"      Created {len(clusters)} clusters, generating summaries...")
        
        # Create parent nodes from clusters
        next_level_nodes = []
        for cluster_id, cluster_nodes in clusters.items():
            # Get texts from cluster
            cluster_texts = [node.text for node in cluster_nodes]
            
            # Generate summary using LLM
            summary = summarize_cluster(cluster_texts, client)
            
            # Create parent node
            embedding = embedding_model.encode([summary])[0]
            parent_node = RAPTORNode(
                text=summary,
                embedding=embedding,
                level=current_level,
                children=cluster_nodes,
                node_id=node_counter
            )
            next_level_nodes.append(parent_node)
            all_nodes.append(parent_node)
            node_counter += 1
        
        print(f"      Level {current_level} complete: {len(next_level_nodes)} summary nodes")
        current_level_nodes = next_level_nodes
    
    print(f"✅ RAPTOR tree built: {len(all_nodes)} total nodes across {current_level + 1} levels")
    return all_nodes, current_level + 1

def create_raptor_index(nodes: List[RAPTORNode]):
    """Create FAISS index from all RAPTOR nodes"""
    embeddings = np.array([node.embedding for node in nodes]).astype('float32')
    faiss.normalize_L2(embeddings)
    
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(embeddings)
    
    return index

def raptor_retrieve(query: str, top_k: int = 16, tree_traverse: bool = True) -> List[RAPTORNode]:
    """
    Retrieve relevant nodes using RAPTOR strategy.
    
    Two modes:
    1. tree_traverse=True: Retrieve from all levels (captures both details and summaries)
    2. tree_traverse=False: Retrieve only from leaf nodes (original chunks)
    """
    if not raptor_store['tree'] or raptor_store['index'] is None:
        return []
    
    # Encode query
    query_embedding = embedding_model.encode([query])
    query_embedding_np = np.array(query_embedding).astype('float32')
    faiss.normalize_L2(query_embedding_np)
    
    if tree_traverse:
        # RAPTOR approach: Search across all levels
        # This allows retrieving both high-level summaries and specific details
        distances, indices = raptor_store['index'].search(query_embedding_np, top_k * 2)
        
        retrieved_nodes = []
        seen_texts = set()
        
        for idx in indices[0]:
            node = raptor_store['tree'][idx]
            # Avoid duplicates
            if node.text not in seen_texts:
                retrieved_nodes.append(node)
                seen_texts.add(node.text)
                
                if len(retrieved_nodes) >= top_k:
                    break
        
        return retrieved_nodes[:top_k]
    else:
        # Traditional RAG: Only search leaf nodes
        leaf_embeddings = np.array([node.embedding for node in raptor_store['leaf_nodes']]).astype('float32')
        faiss.normalize_L2(leaf_embeddings)
        
        leaf_index = faiss.IndexFlatIP(embedding_dim)
        leaf_index.add(leaf_embeddings)
        
        distances, indices = leaf_index.search(query_embedding_np, top_k)
        return [raptor_store['leaf_nodes'][i] for i in indices[0]]

def format_raptor_context(nodes: List[RAPTORNode]) -> str:
    """Format RAPTOR nodes into context for LLM"""
    if not nodes:
        return ""
    
    formatted = "=== RETRIEVED INFORMATION (RAPTOR Multi-Level Retrieval) ===\n\n"
    
    for i, node in enumerate(nodes, 1):
        level_label = "SUMMARY" if node.level > 0 else "DETAIL"
        formatted += f"[{i}] {level_label} (Level {node.level})\n"
        formatted += f"    {node.text}\n\n"
    
    return formatted                     # unchanged
