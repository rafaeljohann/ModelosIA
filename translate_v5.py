import speech_recognition as sr
from transformers import MarianMTModel, MarianTokenizer

# Inicializar o reconhecedor de fala
recognizer = sr.Recognizer()

# Inicializar o modelo e o tokenizador de tradução da Hugging Face
model_name = 'Helsinki-NLP/opus-mt-ROMANCE-en'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def translate(text, tokenizer, model):
    # Tokenizar o texto de entrada
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    # Traduzir o texto
    translated = model.generate(**inputs)
    # Decodificar a tradução
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translated_text

# Usar o microfone como fonte de áudio
with sr.Microphone() as source:
    print("Ajustando o ruído de fundo. Aguarde um momento...")
    recognizer.adjust_for_ambient_noise(source)
    print("Pode falar...")

    try:
        # Capturar o áudio
        while True:
            print("Ouvindo...")
            audio = recognizer.listen(source)

            # Reconhecer a fala usando o Google Web Speech API
            print("Reconhecendo...")
            try:
                text = recognizer.recognize_google(audio, language="pt-BR")
                print(f"Você disse: {text}")

                # Traduzir o texto para o inglês usando o modelo da Hugging Face
                translated_text = translate(text, tokenizer, model)
                print(f"Tradução: {translated_text}")

            except sr.UnknownValueError:
                print("Não consegui entender o que você disse. Por favor, tente novamente.")
            except sr.RequestError as e:
                print(f"Erro na requisição; {e}")

    except KeyboardInterrupt:
        print("Interrompido pelo usuário")
