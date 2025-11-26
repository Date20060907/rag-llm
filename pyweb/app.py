import sys
import os
from flask import Flask, render_template, request, jsonify
import markdown


# Добавление текущей директории в путь поиска модулей
sys.path.append('./')
import myapp


# Инициализация Flask-приложения
app = Flask(__name__)


# Инициализация RAG-системы
rag = myapp.Rag()


# Константа: директория для хранения эталонных файлов
REF_FOLDER = 'ref'


# Создание директории REF_FOLDER, если она не существует
os.makedirs(REF_FOLDER, exist_ok=True)


# Конфигурация приложения: путь к папке с эталонными файлами
app.config['REF_FOLDER'] = REF_FOLDER




## Список идентификаторов выбранных баз данных для RAG-поиска.
# Хранит идентификаторы баз данных, которые пользователь выбрал
# для использования при выполнении запроса к модели.
selected_databases = []


## Параметры генерации и RAG-поиска по умолчанию.
# Содержит настройки, влияющие на поведение языковой модели и механизм извлечения:
# - n_predict: максимальное количество генерируемых токенов
# - temperature: температура генерации (контролирует случайность)
# - top_k: ограничение на выбор из top-K наиболее вероятных токенов
# - rag_k: количество документов, извлекаемых при RAG-поиске
# - rag_sim_threshold: порог схожести для отбора релевантных документов
rag_parameters = {
   'n_predict': 512,
   'temperature': 0.7,
   'top_k': 40,
   'rag_k': 3,
   'rag_sim_threshold': 0.3
}




## @brief Формирует ответ от искусственного интеллекта на основе пользовательского сообщения.
#
# Выполняет RAG-запрос к системе с использованием выбранных баз данных и заданных параметров.
# Результат преобразуется из Markdown в HTML для корректного отображения в веб-интерфейсе.
#
# @param user_message Текст сообщения от пользователя.
# @param selected_databases Список идентификаторов баз данных, используемых при RAG-поиске.
# @param rag_params Словарь параметров RAG и генерации.
# @return Строка с HTML-представлением ответа, предварённая префиксом "Afina: ".
def get_ai_response(user_message, selected_databases, rag_params):
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




## @brief Обработчик корневого маршрута веб-приложения.
#
# Возвращает главную страницу (index.html) для отображения пользовательского интерфейса.
#
# @return Отрендеренный HTML-шаблон главной страницы.
@app.route('/')
def index():
   return render_template('index.html')




## @brief Обработчик POST-запроса для взаимодействия с чатом.
#
# Принимает JSON с сообщением пользователя, выбранными базами данных и параметрами RAG.
# Возвращает JSON с HTML-ответом от модели.
#
# @return JSON-объект с ключом "response", содержащим ответ модели в формате HTML.
@app.route('/chat', methods=['POST'])
def chat():
   data = request.json
   user_message = data.get('message')
   selected_db = data.get('selected_databases', [])
   rag_params = data.get('rag_parameters', rag_parameters)


   ai_response = get_ai_response(user_message, selected_db, rag_params)
   return jsonify({'response': ai_response})




## @brief Обработчик загрузки и обработки файлов для создания новой векторной базы данных.
#
# Принимает текстовые файлы (.txt), сохраняет их во временную директорию,
# а затем инициализирует новую RAG-базу данных с заданным именем и типом генератора фрагментов.
#
# @return JSON-объект с сообщением об успехе или ошибке.
@app.route('/upload', methods=['POST'])
def upload_files():
   db_name = request.form.get('db_name')
   generator_type = request.form.get('generator_type', 'chunk')


   if not db_name:
       return jsonify({'error': 'Database name is required'}), 400


   if 'files' not in request.files:
       return jsonify({'error': 'No files provided'}), 400


   files = request.files.getlist('files')
   if not files or all(f.filename == '' for f in files):
       return jsonify({'error': 'No files selected'}), 400


   saved_filepaths = []
   try:
       for file in files:
           if file and (file.filename.endswith('.txt') or file.content_type == 'text/plain'):
               filepath = os.path.join(app.config['REF_FOLDER'], file.filename)
               file.save(filepath)
               saved_filepaths.append(filepath)
           else:
               return jsonify({'error': f'Invalid file type for {file.filename}. Only .txt files allowed'}), 400


       gen_type = myapp.GeneratorType.chunk if generator_type == 'chunk' else myapp.GeneratorType.paragraphs
       rag.createDatabase(db_name, saved_filepaths, gen_type)
       return jsonify({'message': f'Database "{db_name}" created successfully with {len(saved_filepaths)} files using {generator_type} generator'})
   except Exception as e:
       return jsonify({'error': f'Failed to process files: {str(e)}'}), 500




## @brief Возвращает список доступных векторных баз данных.
#
# Запрашивает у RAG-системы список всех загруженных баз и преобразует их
# в JSON-совместимый формат с идентификатором и именем файла.
#
# @return JSON-массив объектов с полями "id" и "filename".
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




## @brief Обновляет глобальный список выбранных баз данных.
#
# Принимает JSON-массив идентификаторов баз и сохраняет их в глобальной переменной.
#
# @return JSON-объект с подтверждением обновления.
@app.route('/selected_databases', methods=['POST'])
def update_selected_databases():
   global selected_databases
   selected_databases = request.json.get('selected', [])
   return jsonify({'message': 'Selection updated'})




## @brief Возвращает текущие параметры RAG и генерации.
#
# @return JSON-объект с текущими значениями параметров.
@app.route('/parameters', methods=['GET'])
def get_parameters():
   return jsonify(rag_parameters)




## @brief Обновляет глобальные параметры RAG и генерации.
#
# Принимает JSON-объект с новыми значениями и обновляет глобальную переменную rag_parameters.
# Выполняется строгая типизация значений.
#
# @return JSON-объект с подтверждением и новыми параметрами.
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




# Точка входа приложения
if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0')
