from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    Language)
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

# Sample documents for testing
SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

### Supervised Learning
Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs.

Common algorithms include:
- Linear Regression
- Decision Trees
- Neural Networks

### Unsupervised Learning
Unsupervised learning finds hidden patterns in unlabeled data. The algorithm discovers structure without predefined labels.

Common algorithms include:
- K-Means Clustering
- Principal Component Analysis
- Autoencoders

## Applications

Machine learning is used in many fields:
1. Image recognition
2. Natural language processing
3. Recommendation systems
4. Fraud detection
5. Autonomous vehicles
""".strip()

SAMPLE_CODE = '''
def quicksort(arr):
    """
    Quicksort implementation in Python.
    Time complexity: O(n log n) average, O(n²) worst case.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quicksort(left) + middle + quicksort(right)


def binary_search(arr, target):
    """
    Binary search implementation.
    Requires sorted array.
    Time complexity: O(log n)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
'''

def recursive_character_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100, 
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""])
    chunks = splitter.split_text(SAMPLE_TEXT)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Original text length: {len(SAMPLE_TEXT)} characters")
    print(f"First chunk preview: {chunks[0][:100]}...")
    print(f"Chunk lengths: {[len(chunk) for chunk in chunks]}")

def chunk_size_comparison():
    sizes = [50, 100, 200]
    print("Comparing chunk sizes:")
    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, 
            chunk_overlap=size // 5,
            separators=["\n\n", "\n", " ", ""])
        chunks = splitter.split_text(SAMPLE_TEXT)
        print(f"Chunk size: {size} -> Total chunks: {len(chunks)}")

def overlap_importance():
    text = "The quick brown fox jumps over the lazy dog. " * 10
    splitter_no_overlap = RecursiveCharacterTextSplitter(
        chunk_size=50, 
        chunk_overlap=0,
        separators=["\n\n", "\n", " ", ""])
    
    splitter_overlap = RecursiveCharacterTextSplitter(
        chunk_size=50, 
        chunk_overlap=20,
        separators=["\n\n", "\n", " ", ""])
    
    chunks_no_overlap = splitter_no_overlap.split_text(text)
    chunks_overlap = splitter_overlap.split_text(text)

    print(f"Without overlap: {len(chunks_no_overlap)} chunks")
    print(f" Chunk 1 end: ...{chunks_no_overlap[0][-20:]}")
    print(f" Chunk 2 start: {chunks_no_overlap[1][:20]}...")
    
    print(f"\nWith overlap: {len(chunks_overlap)} chunks")
    print(f" Chunk 1 end: ...{chunks_overlap[0][-20:]}")
    print(f" Chunk 2 start: {chunks_overlap[1][:20]}...")

def markdown_splitter():
    headers_to_consider = [
        ('#', 'h1'),
        ('##', 'h2'),
        ('###', 'h3'),
    ]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_consider)
    chunks = splitter.split_text(SAMPLE_TEXT)

    print(f"Markdown Splitter Product {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i} ---")
        print(f" Metadata: {chunk.metadata} \n")
        print(f" Content preview: {chunk.page_content[:200]}...")
        
def code_splitter():
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=100,
        chunk_overlap=50    
    )
    chunks = python_splitter.split_text(SAMPLE_CODE)
    print(f"Code Splitter produced {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        print(f"\n Chunk {i} (length: {len(chunk)} characters)")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

def document_splitter(path: str):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document

    loader = PyPDFLoader(path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # split the docs
    split_docs = splitter.split_documents(docs)

    print(f"Original documents: {len(docs)}")
    print(f"Split documents: {len(split_docs)}")
    for i, doc in enumerate(split_docs):
        print(f"\n--- Split Document {i} ---")
        print(f" Metadata: {doc.metadata} \n")
        print(f" Content preview: {doc.page_content[:200]}...")


if __name__ == "__main__":
    #recursive_character_splitter()
    #chunk_size_comparison()
    #overlap_importance()
    #markdown_splitter()
    #code_splitter()
    document_splitter("../chat_bot/Bank_Nifty_Option_Strategies_Booklet.pdf")