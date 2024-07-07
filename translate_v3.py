import speech_recognition as sr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, MarianMTModel, MarianTokenizer
import torch

# Inicializar o reconhecedor de fala
recognizer = sr.Recognizer()

# Carregar o modelo e o processador para transcrição
transcription_model_name = "facebook/wav2vec2-large-960h"
transcription_processor = Wav2Vec2Processor.from_pretrained(transcription_model_name)
transcription_model = Wav2Vec2ForCTC.from_pretrained(transcription_model_name)

# Carregar o modelo e o tokenizer para tradução
translation_model_name = "Helsinki-NLP/opus-mt-en-roa"
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
translation_model = MarianMTModel.from_pretrained(translation_model_name)

# Função para transcrição de áudio
def transcribe_audio(audio):
    # Processar o áudio
    input_values = transcription_processor(audio, return_tensors="pt", padding="longest").input_values
    # Obter as logits do modelo
    logits = transcription_model(input_values).logits
    # Decodificar os logits em texto
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = transcription_processor.batch_decode(predicted_ids)[0]
    return transcription.lower()

# Função para traduzir texto
def translate_text(text):
    # Tokenizar o texto
    translated_tokens = translation_tokenizer(text, return_tensors="pt", padding="longest")
    # Gerar a tradução
    translated_output = translation_model.generate(**translated_tokens)
    # Decodificar a tradução
    translated_text = translation_tokenizer.decode(translated_output[0], skip_special_tokens=True)
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
                if (audio):
                    print(audio)
                    text = recognizer.recognize_google(audio, language="pt-BR")
                    print(f"Você disse: {text}")

                    # Traduzir a transcrição para inglês
                    print("Traduzindo...")
                    translation = translate_text(text)
                    print(f"Tradução: {translation}")
            except sr.UnknownValueError:
                print("Não consegui entender o que você disse")
            except sr.RequestError as e:
                print(f"Erro na requisição; {e}")
            except:
                break;
        # while True:
        #     print("Ouvindo...")
        #     # audio_data = recognizer.listen(source)
        #     # audio = audio_data.get_wav_data()
        #     audio = recognizer.listen(source)

        #     # Transcrever o áudio usando o modelo da Hugging Face
        #     print("Transcrevendo...")
        #     try:
        #         # transcription = transcribe_audio(audio)
        #         # print(f"Você disse: {transcription}")
        #         text = recognizer.recognize_google(audio, language="pt-BR")
        #         print(f"Você disse: {text}")

        #         # Traduzir a transcrição para inglês
        #         print("Traduzindo...")
        #         translation = translate_text(text)
        #         print(f"Tradução: {translation}")

        #     except Exception as e:
                print(f"Erro: {e}")

    except KeyboardInterrupt:
        print("Interrompido pelo usuário")