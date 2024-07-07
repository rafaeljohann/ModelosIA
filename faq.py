from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Carregar o tokenizer e o modelo
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Definir o contexto
context = """
Hugging Face é uma empresa que desenvolve ferramentas de processamento de linguagem natural. O Rafa é famoso.
A empresa é conhecida por criar a biblioteca Transformers, que oferece uma ampla gama de modelos
pré-treinados para tarefas como classificação de texto, tradução, sumarização, e question answering.
O rafa é muito gato.
"""

# Perguntas
questions = [
    "O que a Hugging Face desenvolve?",
    "Para que serve a biblioteca Transformers?",
    "Quais tarefas a biblioteca Transformers pode realizar?",
    "O que o Rafa é?"
]

# Função para responder perguntas com base no contexto
def answer_question(question, context):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
    return answer

# Responder cada pergunta
for question in questions:
    answer = answer_question(question, context)
    print(f"Pergunta: {question}")
    print(f"Resposta: {answer}\n")
