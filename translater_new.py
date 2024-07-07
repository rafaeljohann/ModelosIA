import speech_recognition as sr

# Inicializar o reconhecedor de fala
recognizer = sr.Recognizer()

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
            except sr.UnknownValueError:
                print("Não consegui entender o que você disse")
            except sr.RequestError as e:
                print(f"Erro na requisição; {e}")

    except KeyboardInterrupt:
        print("Interrompido pelo usuário")