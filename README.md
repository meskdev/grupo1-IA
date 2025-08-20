Estrutura do Projeto:
- app.py (Flask)
- modelo.py (rede neural treinada no EMnist)
- emnist_model.h5 (arquivo de modelo salvo, após dar "run" no modelo.py
- /templates
  - index.html (front end, com html e js com opção de desenho canvas, para envio para predição do modelo)
- /.venv (ambiente virtual *recomendado*)


Passo-a-passo:
  Crie uma pasta para o projeto e crie um ambiente virtual (python -m venv .venv)
  ative o ambiente virtual (.venv/Scripts/activate) e instale os requirements (pip install -r requirements.txt)
  rode o modelo.py (opcional -já há um modelo salvo-, leva em torno de 40min)
  rode o app.py e abra uma aba no navegador http://localhost:5000/

