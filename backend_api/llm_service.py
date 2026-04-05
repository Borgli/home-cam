import json
import requests
from flask import Blueprint, request, jsonify
from .database import get_schema_info, query_db

llm_bp = Blueprint('llm', __name__)

OLLAMA_URL = 'http://localhost:11434/api/generate'
MODEL = 'gemma4:e2b'


def generate_sql_prompt(question, schema):
    """Build prompt for text-to-SQL generation."""
    return f"""You are a SQL expert. Given the following SQLite database schema and a question, generate ONLY a valid SQLite SELECT query. Do not explain, do not add markdown, just output the raw SQL.

{schema}

Important rules:
- Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
- Use SQLite syntax
- The timestamp column stores Unix epoch seconds (use datetime() to format if needed)
- Channel numbers are 0-3 (for cameras 1-4)
- Common class_name values: 'person', 'car', 'truck', 'bicycle', 'dog', 'cat'
- Keep queries efficient with LIMIT when appropriate

Question: {question}

SQL:"""


@llm_bp.route('/db/llm-query', methods=['POST'])
def llm_query():
    """Natural language to SQL query via Ollama."""
    data = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    # Get schema
    schema = get_schema_info()
    prompt = generate_sql_prompt(question, schema)

    # Call Ollama
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                'model': MODEL,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.1,
                    'num_predict': 256,
                }
            },
            timeout=60,
        )

        if response.status_code != 200:
            return jsonify({
                'error': f'Ollama returned status {response.status_code}. Is Ollama running? (ollama serve)'
            }), 502

        result = response.json()
        generated_sql = result.get('response', '').strip()

        # Clean up the SQL (remove markdown code blocks if present)
        if generated_sql.startswith('```'):
            lines = generated_sql.split('\n')
            generated_sql = '\n'.join(
                l for l in lines if not l.startswith('```')
            ).strip()

        # Remove any trailing semicolons
        generated_sql = generated_sql.rstrip(';').strip()

        if not generated_sql:
            return jsonify({'error': 'LLM returned empty response'}), 500

        # Validate it's a SELECT
        if not generated_sql.strip().upper().startswith('SELECT'):
            return jsonify({
                'error': 'Generated query is not a SELECT statement',
                'sql': generated_sql,
            }), 400

        # Execute the query
        try:
            rows = query_db(generated_sql)
            return jsonify({
                'sql': generated_sql,
                'rows': rows,
                'count': len(rows),
                'model': MODEL,
            })
        except Exception as e:
            return jsonify({
                'error': f'SQL execution error: {str(e)}',
                'sql': generated_sql,
            }), 400

    except requests.exceptions.ConnectionError:
        return jsonify({
            'error': 'Cannot connect to Ollama. Please install and start it:\n1. Install from ollama.com\n2. Run: ollama pull gemma4:e2b\n3. Ollama starts automatically on install'
        }), 503
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Ollama timed out. The model may be loading for the first time.'}), 504
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
