from flask import Flask, jsonify, request
from nltk.tokenize import wordpunct_tokenize
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")

app = Flask(__name__)

def engine (entrada):
  entrada = entrada.encode("utf-8", errors="ignore").decode("utf-8").lower()
  tokens = wordpunct_tokenize(entrada)
  if "ola" in tokens:
    return "oi pra voce"
  else:
    return "???"

'''
pergunta: testeAula (main) $ curl -X POST http://127.0.0.1:5000/sentimento -H "Content-Type: application/json" -d '{"mensagem": "Hello, bad ass!"}'
{'neg': 0.879, 'neu': 0.121, 'pos': 0.0, 'compound': -0.807}

deveria dar positivo, porque dá negativo

https://www.nltk.org/_modules/nltk/sentiment/vader.html
'''
from nltk.sentiment.vader import SentimentIntensityAnalyzer
@app.route("/sentimento", methods=["POST"])
def analisa_sentimento():
    dados = request.get_json()
    if not dados or "mensagem" not in dados:
        return jsonify({"erro": "O campo 'mensagem' é obrigatório no corpo do JSON."}), 400

    mensagem = dados["mensagem"]
    sia = SentimentIntensityAnalyzer()
    sia.polarity_scores(mensagem)
    return str(sia.polarity_scores(mensagem))


# Mudamos a rota para aceitar POST e alteramos o endpoint para /chatbot
@app.route("/chatbot", methods=["POST"])
def chatbot():
  # Obtém os dados no formato JSON enviados na requisição
  dados = request.get_json()

  # Valida se a mensagem foi enviada
  if not dados or "mensagem" not in dados:
    return (
        jsonify({"erro": "O campo 'mensagem' e obrigatorio no corpo do JSON."}),
        400,
    )

  mensagem_usuario = dados["mensagem"]

  # Lógica simples de resposta (aqui você integrará sua IA ou regras)
  resposta_bot = f"Você disse: '{mensagem_usuario}'.\n" + str(engine(mensagem_usuario)+"\n")

  # Retorna a resposta estruturada em JSON
  return jsonify({"resposta": resposta_bot}), 200


if __name__ == "__main__":
  # debug=True ajuda no desenvolvimento, reiniciando o servidor a cada alteração
  app.run(debug=True)
