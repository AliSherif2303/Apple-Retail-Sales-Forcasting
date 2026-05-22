# -*- coding: utf-8 -*-
"""Page 6 — SQL RAG Agent (uses local Ollama)"""
import streamlit as st
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent.parent
PROJECT = APP_DIR.parent

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #000000; }
[data-testid="stSidebar"] { background: #1C1C1E; border-right: 1px solid rgba(0,240,255,0.2); }
.section-header { font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin: 28px 0 12px 0;
  padding-bottom: 8px; border-bottom: 2px solid rgba(0,240,255,0.4); }
</style>"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown("# 🤖 SQL RAG Agent")
st.markdown("*Ask questions about Apple sales data in plain English.*")

# Check dependencies
try:
    import duckdb
    from langchain_community.utilities import SQLDatabase
    from langchain_classic.chains import create_sql_query_chain
    from langchain_ollama import ChatOllama
    from langchain_community.tools import QuerySQLDatabaseTool
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.info("Install with: `pip install duckdb duckdb-engine langchain langchain-classic langchain-community langchain-ollama sqlalchemy`")
    st.stop()

# DB setup
CSV_PATH = PROJECT / "data" / "processed" / "cleaned_apple_sales_v3.csv"
DB_DIR = APP_DIR
DB_PATH = DB_DIR / "rag_agent.db"

@st.cache_resource(show_spinner="Building database...")
def build_db():
    if not CSV_PATH.exists():
        return None, None, None
    import os, duckdb

    if not DB_PATH.exists():
        st.info("First run: loading CSV into DuckDB...")
        con = duckdb.connect(str(DB_PATH))
        con.execute(f"CREATE TABLE sales AS SELECT * FROM read_csv_auto('{str(CSV_PATH)}')")
        con.close()

    custom_schema = """
CREATE TABLE sales (
    sale_date TIMESTAMP,
    store_name VARCHAR,
    city VARCHAR,
    country_norm_mapped VARCHAR,
    product_name VARCHAR,
    category_name VARCHAR,
    sales_amount_realistic DOUBLE,
    quantity_realistic DOUBLE,
    price_realistic DOUBLE,
    year BIGINT,
    month BIGINT
);
"""

    db = SQLDatabase.from_uri(
        f"duckdb:///{DB_PATH}",
        custom_table_info={"sales": custom_schema}
    )
    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = ChatOllama(model="qwen2.5-coder:3b", temperature=0, base_url=ollama_base_url)
    analyst = ChatOllama(model="qwen2.5-coder:3b", temperature=0.0, base_url=ollama_base_url)
    return db, llm, analyst

db, llm, analyst_llm = build_db()
if db is None:
    st.error("cleaned_apple_sales_v3.csv not found. Cannot build database.")
    st.stop()

# Build chains
sql_prompt = PromptTemplate.from_template(
    """You are an elite DuckDB SQL programming assistant answering questions about Apple Retail Sales data.
    Given an input question, create a syntactically correct DuckDB query to run.
    
    Never query for all the columns from a specific table, only ask for the few relevant columns given the question.
    Be careful to not query for columns that do not exist.
    
    CRITICAL APPLE RETAIL BUSINESS RULES:
    1. If asked about "Sales", "Revenue", or "Income", use the 'sales_amount_realistic' column.
    2. If asked about Volume or Item Counts, use 'quantity_realistic'.
    3. If asked about a Country, filter using 'country_norm_mapped'.
    4. There is ONLY ONE table named 'sales'. DO NOT JOIN other tables.
    5. If asked to count "transactions" or "orders", use COUNT(*).
    6. If comparing metrics across different years, use conditional aggregation: SUM(CASE WHEN year=2023 THEN column END) AS year_2023, SUM(CASE WHEN year=2024 THEN column END) AS year_2024.
    7. Output ONLY the raw SQL string. No markdown, no explanation.
    8. To find the "most" or "least", use ORDER BY + LIMIT 1, never MAX()/MIN() with unaggregated columns.
    9. When asked for the PRICE of a product, ALWAYS use 'price_realistic'. NEVER use 'sales_amount_realistic' for prices.
    10. When filtering for a specific product line, use product_name ILIKE '%ProductName%'.
    
    ADDITIONAL CRITICAL RULES:
    11. When asked about GDP or GDP per capita, ALWAYS use the 'gdp_per_capita' column. GDP is NOT revenue.
    12. IMPORTANT PRODUCT MATCHING: If a user asks for a base model like 'iPhone 13' or 'iPhone 14', they mean ONLY the base model. You MUST use exact matching: `product_name ILIKE 'iPhone 13'`. Do NOT use `%` wildcards (`ILIKE '%iPhone 13%'`) because that will accidentally include 'Pro', 'mini', and 'Plus'. ONLY use wildcards if they explicitly ask for the 'iPhone 13 family' or 'all iPhone 13s'.
    13. When asked 'which products can I afford' or about a budget, ALWAYS: (a) use AVG(price_realistic) grouped by product_name, (b) filter with HAVING AVG(price_realistic) <= budget, (c) ONLY include the product category the user asked about (e.g. if they ask about iPhones, add WHERE product_name ILIKE '%iPhone%').
    14. When comparing exactly 2 cities or locations, ALWAYS add WHERE city IN ('City1', 'City2') to filter only those cities. YOU MUST ALSO include the 'city' column in the SELECT and GROUP BY clauses.
    15. When asked if a price 'dropped' or 'changed' between years, use conditional aggregation to get separate values per year, e.g.: AVG(CASE WHEN year=2023 THEN price_realistic END) AS price_2023, AVG(CASE WHEN year=2024 THEN price_realistic END) AS price_2024.
    16. VERY IMPORTANT: ALL string values in 'country_norm_mapped' are STRICTLY LOWERCASE (e.g., 'united states'). Always use exact lowercase strings when filtering by country! Also use ILIKE for product_name to be case-insensitive.
    17. ALWAYS SELECT the columns you are GROUPING BY. If you GROUP BY year, you MUST include year in the SELECT clause.
    18. TIME SERIES GROUPING: If the user asks 'for each year' or 'by year', you MUST `GROUP BY year` and `SELECT year`. If they ask 'by month', you MUST `GROUP BY month` and `SELECT month`. Do NOT add any extra grouping columns (like city, country, or store) unless explicitly requested.

    EXAMPLES:
    User: "What was total revenue in 2024?"
    SQL: SELECT SUM(sales_amount_realistic) FROM sales WHERE year = 2024;
    
    User: "What is the average price of iPhone 13 and iPhone 14 for each year?"
    SQL: SELECT product_name, year, AVG(price_realistic) FROM sales WHERE (product_name ILIKE 'iPhone 13' OR product_name ILIKE 'iPhone 14') GROUP BY product_name, year;
    
    User: "Which country had the most transactions?"
    SQL: SELECT country_norm_mapped, COUNT(*) FROM sales GROUP BY country_norm_mapped ORDER BY COUNT(*) DESC LIMIT 1;
    
    User: "Compare the average price of the iPhone 14 between London and New York"
    SQL: SELECT city, AVG(price_realistic) FROM sales WHERE product_name ILIKE 'iPhone 14' AND city IN ('London', 'New York') GROUP BY city;
    
    User: "Did the price of MacBook Air change between 2023 and 2024?"
    SQL: SELECT product_name, AVG(CASE WHEN year=2023 THEN price_realistic END) AS price_2023, AVG(CASE WHEN year=2024 THEN price_realistic END) AS price_2024 FROM sales WHERE product_name ILIKE '%MacBook Air%' GROUP BY product_name;
    
    Only use the following tables:
    {table_info}

    Return a maximum of {top_k} results unless otherwise specified.

    Question: {input}""")

analyst_prompt = ChatPromptTemplate.from_template(
    """You are a Senior Data Analyst presenting findings to Apple's executive leadership.

Your communication style:
- Professional, confident, and concise
- Lead with the key insight first, then supporting details
- Use exact numbers with proper formatting (commas, currency symbols, percentages)
- Add brief business context or actionable takeaways when relevant
- If the data shows a trend, highlight it
- Keep responses to 2-4 sentences for simple queries, up to a short paragraph for complex ones
- Never mention SQL, databases, tables, columns, or technical implementation details

CRITICAL ANTI-HALLUCINATION RULES:
1. You MUST rely EXCLUSIVELY on the data provided in 'The data returned'. Do NOT use your pre-trained knowledge to fill in prices, dates, or sales figures.
2. If 'The data returned' is empty (e.g., `[]`, `""`, or `None`), you MUST explicitly state that there is no data available for this query. Do NOT invent a number. 
3. For example, if asked about a product in a year before it launched, the data will be empty. Explain that the product likely did not exist or had no sales in that period.
4. Do NOT perform mathematical calculations (like averaging multiple product models together). Only report the exact numbers and exact product names provided in the SQL result.

The user asked: "{question}"

The data returned: {result}

Provide your executive briefing:""")

write_query = create_sql_query_chain(llm, db, prompt=sql_prompt)
execute_query = QuerySQLDatabaseTool(db=db)
analyst_chain = analyst_prompt | analyst_llm | StrOutputParser()

# Chat interface
st.markdown('<div class="section-header">💬 Ask a Question</div>', unsafe_allow_html=True)

if "rag_messages" not in st.session_state:
    st.session_state.rag_messages = []

for msg in st.session_state.rag_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("e.g. What was total revenue in 2024?"):
    st.session_state.rag_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Generating SQL..."):
            try:
                sql = write_query.invoke({"question": question})
                clean_sql = sql.replace("```sql","").replace("```","").replace("SQLQuery:","").strip()
                if "LIMIT" not in clean_sql.upper():
                    clean_sql = clean_sql.rstrip(";") + " LIMIT 50;"
                with st.expander("🔧 Generated SQL"):
                    st.code(clean_sql, language="sql")
                raw = execute_query.invoke(clean_sql)
                with st.expander("📋 Raw Result"):
                    st.text(str(raw)[:500])
                briefing = analyst_chain.invoke({"question": question, "result": raw})
                st.markdown(briefing)
                st.session_state.rag_messages.append({"role": "assistant", "content": briefing})
            except Exception as e:
                err = f"❌ Error: {e}"
                st.error(err)
                st.session_state.rag_messages.append({"role": "assistant", "content": err})
