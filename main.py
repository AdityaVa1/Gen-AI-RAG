import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import json

# Note: Install these packages:
# pip install streamlit pandas plotly langchain langchain-openai chromadb sentence-transformers

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    st.warning("LangChain not installed. Install with: pip install langchain langchain-openai chromadb sentence-transformers")

# Page configuration
st.set_page_config(
    page_title="Client Data RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'data_summary' not in st.session_state:
    st.session_state.data_summary = None

# Sidebar for configuration
st.sidebar.title("⚙️ Configuration")

# API Key input
api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload Client Dataset (CSV)", type=['csv'])

# Model selection
model_name = st.sidebar.selectbox(
    "Select LLM Model",
    ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
    index=0
)

# Temperature setting
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)

class DataAnalyzer:
    """Class for analyzing client data and generating insights"""
    
    @staticmethod
    def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key performance indicators from the dataset"""
        kpis = {
            "total_records": len(df),
            "columns": list(df.columns),
            "numeric_columns": list(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(df.select_dtypes(include=['object']).columns),
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict()
        }
        
        # Calculate statistics for numeric columns
        numeric_stats = {}
        for col in kpis["numeric_columns"]:
            numeric_stats[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75))
            }
        kpis["numeric_statistics"] = numeric_stats
        
        return kpis
    
    @staticmethod
    def perform_time_series_analysis(df: pd.DataFrame, date_column: str = None) -> Dict[str, Any]:
        """Perform time series analysis if date column exists"""
        if date_column is None:
            # Try to detect date column
            for col in df.columns:
                if 'date' in col.lower() or 'time' in col.lower():
                    date_column = col
                    break
        
        if date_column is None:
            return {"status": "No date column found"}
        
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            df_sorted = df.sort_values(date_column)
            
            analysis = {
                "date_column": date_column,
                "date_range": {
                    "start": str(df_sorted[date_column].min()),
                    "end": str(df_sorted[date_column].max())
                },
                "total_days": (df_sorted[date_column].max() - df_sorted[date_column].min()).days
            }
            
            # Trend analysis for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                trends = {}
                for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
                    first_val = df_sorted[col].iloc[0]
                    last_val = df_sorted[col].iloc[-1]
                    change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                    trends[col] = {
                        "change_percent": round(change, 2),
                        "direction": "increasing" if change > 0 else "decreasing"
                    }
                analysis["trends"] = trends
            
            return analysis
        except Exception as e:
            return {"status": f"Error in time series analysis: {str(e)}"}
    
    @staticmethod
    def generate_data_summary(df: pd.DataFrame) -> str:
        """Generate a comprehensive text summary of the dataset"""
        kpis = DataAnalyzer.calculate_kpis(df)
        
        summary = f"""
# Dataset Summary

## Overview
- Total Records: {kpis['total_records']}
- Total Columns: {len(kpis['columns'])}
- Numeric Columns: {len(kpis['numeric_columns'])}
- Categorical Columns: {len(kpis['categorical_columns'])}

## Column Information
Columns in dataset: {', '.join(kpis['columns'])}

## Numeric Column Statistics
"""
        for col, stats in kpis['numeric_statistics'].items():
            summary += f"\n### {col}\n"
            summary += f"- Mean: {stats['mean']:.2f}\n"
            summary += f"- Median: {stats['median']:.2f}\n"
            summary += f"- Std Dev: {stats['std']:.2f}\n"
            summary += f"- Range: [{stats['min']:.2f}, {stats['max']:.2f}]\n"
        
        # Add sample data
        summary += "\n## Sample Data (First 5 Rows)\n"
        summary += df.head().to_string()
        
        return summary

class RAGSystem:
    """Retrieval Augmented Generation System"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for RAG functionality")
        
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
    
    def create_vector_store(self, documents: List[str]) -> Chroma:
        """Create vector store from documents"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
        texts = []
        for doc in documents:
            texts.extend(text_splitter.split_text(doc))
        
        vector_store = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            collection_name="client_data"
        )
        
        return vector_store
    
    def create_qa_chain(self, vector_store: Chroma) -> RetrievalQA:
        """Create QA chain with custom prompt"""
        
        prompt_template = """You are an AI assistant specialized in analyzing client data. 
Use the following pieces of context about the client dataset to answer the question. 
If you don't know the answer based on the context, say so - don't make up information.

Context: {context}

Question: {question}

Provide a detailed, accurate answer based on the data:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        return qa_chain

def initialize_rag_system(df: pd.DataFrame, api_key: str, model_name: str, temperature: float):
    """Initialize the RAG system with client data"""
    with st.spinner("🔄 Initializing RAG system..."):
        # Generate comprehensive data summary
        data_summary = DataAnalyzer.generate_data_summary(df)
        st.session_state.data_summary = data_summary
        
        # Perform time series analysis
        time_series_analysis = DataAnalyzer.perform_time_series_analysis(df)
        
        # Create documents for RAG
        documents = [
            data_summary,
            f"Time Series Analysis:\n{json.dumps(time_series_analysis, indent=2)}"
        ]
        
        # Add categorical value distributions
        for col in df.select_dtypes(include=['object']).columns:
            value_counts = df[col].value_counts().head(10).to_dict()
            documents.append(f"Distribution of {col}:\n{json.dumps(value_counts, indent=2)}")
        
        # Initialize RAG
        rag = RAGSystem(api_key, model_name, temperature)
        vector_store = rag.create_vector_store(documents)
        qa_chain = rag.create_qa_chain(vector_store)
        
        st.session_state.vector_store = vector_store
        st.session_state.qa_chain = qa_chain
        st.session_state.rag = rag
        
    st.success("✅ RAG system initialized successfully!")

# Main UI
st.title("🤖 Client Dataset RAG Chatbot")
st.markdown("**AI-powered chatbot with Retrieval Augmented Generation for client data analysis**")

# Load and process data
if uploaded_file is not None:
    if st.session_state.data is None:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.sidebar.success(f"✅ Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Display data preview
        with st.expander("📊 Data Preview"):
            st.dataframe(df.head(10))
            
            # Display KPIs
            kpis = DataAnalyzer.calculate_kpis(df)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Records", kpis['total_records'])
            col2.metric("Total Columns", len(kpis['columns']))
            col3.metric("Numeric Columns", len(kpis['numeric_columns']))
            col4.metric("Categorical Columns", len(kpis['categorical_columns']))
        
        # Initialize RAG if API key is provided
        if api_key and LANGCHAIN_AVAILABLE:
            initialize_rag_system(df, api_key, model_name, temperature)
    
    # Chatbot interface
    if api_key and LANGCHAIN_AVAILABLE and 'qa_chain' in st.session_state:
        st.markdown("---")
        st.subheader("💬 Chat with your data")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your client data..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.qa_chain({"query": prompt})
                        answer = response['result']
                        
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Optionally show source documents
                        if 'source_documents' in response and len(response['source_documents']) > 0:
                            with st.expander("📚 Sources"):
                                for i, doc in enumerate(response['source_documents'][:3]):
                                    st.text(f"Source {i+1}:\n{doc.page_content[:200]}...")
                    except Exception as e:
                        error_msg = f"Error generating response: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    elif not api_key:
        st.info("👈 Please enter your OpenAI API key in the sidebar to start chatting")
    elif not LANGCHAIN_AVAILABLE:
        st.warning("Please install required packages: pip install langchain langchain-openai chromadb sentence-transformers")

else:
    st.info("👈 Please upload a CSV file to begin")
    
    # Show example usage
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        1. **Upload your client dataset** (CSV format) using the sidebar
        2. **Enter your OpenAI API key** in the sidebar
        3. **Review the data preview** and KPIs
        4. **Start chatting** with the AI about your data
        
        ### Example questions you can ask:
        - "What are the key trends in this dataset?"
        - "Summarize the main statistics"
        - "What insights can you find in the data?"
        - "Analyze the time series patterns"
        - "What are the correlations between variables?"
        """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Features")
st.sidebar.markdown("""
- ✅ RAG implementation
- ✅ Data analysis & KPIs
- ✅ Time series analysis
- ✅ LangChain pipelines
- ✅ Prompt engineering
- ✅ Vector embeddings
""")
