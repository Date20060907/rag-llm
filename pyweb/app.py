import sys
import os
from flask import Flask, render_template, request, jsonify
import markdown
sys.path.append('./')
import myapp

app = Flask(__name__)
rag = myapp.Rag()

# Create ref directory if it doesn't exist
REF_FOLDER = 'ref'
os.makedirs(REF_FOLDER, exist_ok=True)
app.config['REF_FOLDER'] = REF_FOLDER

# Global variables
selected_databases = []
rag_parameters = {
    'n_predict': 512,
    'temperature': 0.7,
    'top_k': 40,
    'rag_k': 3,
    'rag_sim_threshold': 0.3
}

def get_ai_response(user_message, selected_databases, rag_params):
    # Pass selected databases and parameters to the RAG
    response = rag.request(
        user_message, 
        selected_databases,
        rag_params['n_predict'],
        rag_params['temperature'],
        rag_params['top_k'],
        rag_params['rag_k'],
        rag_params['rag_sim_threshold']
    )
    html_response = markdown.markdown(response, extensions=['fenced_code'])
    return "Afina: " + html_response

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    selected_db = data.get('selected_databases', [])
    rag_params = data.get('rag_parameters', rag_parameters)
    
    ai_response = get_ai_response(user_message, selected_db, rag_params)
    return jsonify({'response': ai_response})

@app.route('/upload', methods=['POST'])
def upload_files():
    # Get database name and generator type from form data
    db_name = request.form.get('db_name')
    generator_type = request.form.get('generator_type', 'chunk')  # default to 'chunk'
    
    if not db_name:
        return jsonify({'error': 'Database name is required'}), 400
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400
    
    saved_filepaths = []
    try:
        # Save all files
        for file in files:
            if file and (file.filename.endswith('.txt') or file.content_type == 'text/plain'):
                filepath = os.path.join(app.config['REF_FOLDER'], file.filename)
                file.save(filepath)
                saved_filepaths.append(filepath)
            else:
                return jsonify({'error': f'Invalid file type for {file.filename}. Only .txt files allowed'}), 400
        
        # Process all files with RAG using custom name and generator type
        gen_type = myapp.GeneratorType.chunk if generator_type == 'chunk' else myapp.GeneratorType.paragraphs
        rag.createDatabase(db_name, saved_filepaths, gen_type)
        return jsonify({'message': f'Database "{db_name}" created successfully with {len(saved_filepaths)} files using {generator_type} generator'})
    except Exception as e:
        return jsonify({'error': f'Failed to process files: {str(e)}'}), 500

@app.route('/databases')
def get_databases():
    try:
        databases = rag.get_vector_database_list()
        db_list = []
        for i, db in enumerate(databases):
            db_list.append({
                'id': i,
                'filename': db.getFilename()
            })
        return jsonify(db_list)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/selected_databases', methods=['POST'])
def update_selected_databases():
    global selected_databases
    selected_databases = request.json.get('selected', [])
    return jsonify({'message': 'Selection updated'})

@app.route('/parameters', methods=['GET'])
def get_parameters():
    return jsonify(rag_parameters)

@app.route('/parameters', methods=['POST'])
def update_parameters():
    global rag_parameters
    data = request.json
    rag_parameters = {
        'n_predict': int(data.get('n_predict', 512)),
        'temperature': float(data.get('temperature', 0.7)),
        'top_k': int(data.get('top_k', 40)),
        'rag_k': int(data.get('rag_k', 3)),
        'rag_sim_threshold': float(data.get('rag_sim_threshold', 0.3))
    }
    return jsonify({'message': 'Parameters updated', 'parameters': rag_parameters})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')