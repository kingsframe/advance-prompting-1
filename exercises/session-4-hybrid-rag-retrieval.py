from pgvector.psycopg import register_vector
import psycopg
from datasets import load_dataset
from openai import OpenAI

# --------------one big hack to import local util modules
import os
import sys

# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current directory
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the Python module search path
sys.path.append(parent_dir)
# print(sys.path)

from utils.fetch_embedding import fetch_embedding
from utils.count_token import num_tokens_from_string
from utils.constants import EMBEDDING_MAX_INPUT, ENCODING_NAME
# -----------------------------------------------

dataset = load_dataset(
    "example dataset", token="<your_token>", split="all"
)
# remove no answer question pairs
dataset = dataset.train_test_split(test_size=0.1).filter(
    lambda pair: len(pair["answers"]) > 0
)
# remove pairs that exceed embedding token limit
dataset = dataset.filter(
    lambda pair: num_tokens_from_string(pair["question"], ENCODING_NAME)
    + num_tokens_from_string(" ".join(pair["answers"]), ENCODING_NAME)
    < EMBEDDING_MAX_INPUT
)

# make sure intirn_vector_db database is created. see README
conn = psycopg.connect(
    dbname="intirn_vector_db",
    user="postgres",
    host="localhost",  # Localhost since we are using an SSH tunnel
    port=54321,  # The port forwarded by the SSH tunnel
    autocommit=True,
)
# Register the vector type with your connection
register_vector(conn)

query = dataset["train"][0]["question"]
print('user question: ', query)
query_embedding = fetch_embedding(query)

k = 60
sql_statement = f"""
    WITH semantic_search AS (
        SELECT id, RANK () OVER (ORDER BY question_embedding <=> '{query_embedding}') AS rank
        FROM "solana-content-3072" 
        ORDER BY question_embedding <=> '{query_embedding}'
        LIMIT 20
    ),
    keyword_search AS (
        SELECT id, RANK () OVER (ORDER BY ts_rank_cd(to_tsvector('english', question), plainto_tsquery('english', '{query}')) DESC)
        FROM "solana-content-3072" 
        WHERE to_tsvector('english', question) @@ plainto_tsquery('english', '{query}')
        ORDER BY ts_rank_cd(to_tsvector('english', question), plainto_tsquery('english', '{query}')) DESC
        LIMIT 20
    )
    SELECT
        COALESCE(semantic_search.id, keyword_search.id) AS id,
        COALESCE(1.0 / ({k} + semantic_search.rank), 0.0) +
        COALESCE(1.0 / ({k} + keyword_search.rank), 0.0) AS score,
        d.answers AS answers
    FROM semantic_search
    FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
    JOIN "solana-content-3072" d ON d.id = COALESCE(semantic_search.id, keyword_search.id)
    ORDER BY score DESC
    LIMIT 3
"""

results = conn.execute(sql_statement).fetchall()
answers = ''
for row in results:
    print("qa:", row[0], "RRF score:", row[1])
    answers += row[2]

BASE_PROMPT_V1 = "You are an intelligent Solana chat support agent, and you have been provided historical questions and answers about solana ecosystem. Use your own knowledge and the provided context to answer each question with example code or useful links if it is available in the context. Avoid any sales-speak and avoid explaining acronyms. Avoid mentioning that you've been provided company documentation or historical context for assistance. If the company documentation provided is not relevant to the question at hand, you can use your own knowledge in answering the question \n\n historical user questions and answers: "
final_prompt = BASE_PROMPT_V1 + answers
print('final prompt: ', final_prompt)

openai_client = OpenAI(
            api_key="your key"
        )

MODEL_NAME = "gpt-4o-mini"
response = openai_client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": final_prompt},
        {"role": "user", "content": f'please answer this question: ${query}'}
    ],
    seed=123,
    temperature=0.2,
)

print("llm_response: ", response.choices[0].message.content)