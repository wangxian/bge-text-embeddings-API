import json
import logging
from flask import Flask
from flask import jsonify, request
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logging.getLogger(__name__)

app = Flask(__name__)
model = SentenceTransformer('BAAI/bge-large-zh-v1.5')

def make_embeddings(text_string):
    sentences = [text_string]

    try:
        embeddings = model.encode(sentences)
        return embeddings
    
    except Exception as e:
        error_msg = {'error': e}
        logging.error(f'got error: {error_msg["error"]}')
        return error_msg, 403

@app.route('/v1/embeddings', methods=['POST'])
def embed_text():

    try:
        data = request.get_json()
        logging.info(data)

        if 'input' not in data:
            error_msg = {'error':'missing input field in JSON_DATA.'}
            logging.error(error_msg['error'])
            return error_msg, 400

        embeddings = make_embeddings(data['input'])
        logging.info('embeddings:')
        logging.info(f'{embeddings}')
        embeddings = embeddings.tolist()
        response_json = json.dumps({'embeddings':embeddings})
        return response_json, 200

    except Exception as e:
        error_msg = {'error': e}
        logging.error(f'got error: {error_msg["error"]}')
        return error_msg, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0')
