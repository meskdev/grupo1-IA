from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import base64

app = Flask(__name__)
modelo = load_model('emnist_model.h5')

# Classes: 0-9, A-Z, a-z
classes = [str(i) for i in range(10)] + \
          [chr(i) for i in range(65, 91)] + \
          [chr(i) for i in range(97, 123)]


@app.route('/')
def inicio():
    return render_template('index.html')


@app.route('/prever', methods=['POST'])
def prever():
    try:
        dados = request.json
        if 'imagem' not in dados:
            return jsonify({'erro': 'Nenhuma imagem enviada'}), 400

        imagem_b64 = dados['imagem'].split(
            'base64,')[1] if 'base64,' in dados['imagem'] else dados['imagem']
        imagem_bytes = base64.b64decode(imagem_b64)

        # Processar imagem
        imagem = Image.open(io.BytesIO(imagem_bytes))
        imagem = imagem.convert('L').resize((28, 28))
        imagem = imagem.transpose(Image.FLIP_LEFT_RIGHT)
        imagem = Image.eval(imagem, lambda x: 255 - x)

        # Converter para array do modelo
        array_imagem = np.array(imagem) / 255.0
        array_imagem = array_imagem.reshape(1, 28, 28, 1)

        # Fazer predição
        predicao = modelo.predict(array_imagem, verbose=0)
        classe_idx = np.argmax(predicao)

        return jsonify({
            'previsao': classes[classe_idx],
            'confianca': float(predicao.max())
        })

    except Exception as e:
        return jsonify({'erro': str(e)}), 500


if __name__ == '__main__':
    print("🚀 Servidor Flask iniciando...")
    print("📱 Acesse: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
