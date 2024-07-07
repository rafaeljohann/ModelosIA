import pyaudio
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, MarianMTModel, MarianTokenizer
import numpy as np
import sentencepiece

# Configure seu token de acesso Hugging Face (se necessário)
# token = "seu_token_de_acesso"

# Carregar modelos
asr_model_name = "facebook/wav2vec2-large-960h"
asr_model = Wav2Vec2ForCTC.from_pretrained(asr_model_name)  # , use_auth_token=token
asr_processor = Wav2Vec2Processor.from_pretrained(asr_model_name)  # , use_auth_token=token

translation_model_name = "Helsinki-NLP/opus-mt-en-roa"
translation_model = MarianMTModel.from_pretrained(translation_model_name)  # , use_auth_token=token
translation_tokenizer = MarianTokenizer.from_pretrained(translation_model_name)  # , use_auth_token=token

# Configurações de áudio
RATE = 16000
CHUNK = 1024

# Inicializar PyAudio
p = pyaudio.PyAudio()

# Função para processar áudio em tempo real
def callback(in_data, frame_count, time_info, status):
    audio_input = np.frombuffer(in_data, dtype=np.float32)
    input_values = asr_processor(audio_input, return_tensors="pt", sampling_rate=RATE).input_values
    
    # Realizar transcrição
    with torch.no_grad():
        logits = asr_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = asr_processor.batch_decode(predicted_ids)[0]
    
    # Realizar tradução
    translated = translation_model.generate(**translation_tokenizer.prepare_seq2seq_batch(transcription, return_tensors="pt"))
    translated_text = translation_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    
    print(f"Transcrição: {transcription}")
    print(f"Tradução: {translated_text}")
    
    return (in_data, pyaudio.paContinue)

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    print(f"Device {i}: {info['name']}")

# Substitua 'device_index' pelo índice do seu microfone
device_index = 0  # Por exemplo, se o seu microfone for o dispositivo 1

# Abrir stream de áudio com o dispositivo especificado
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK,
                stream_callback=callback)

print("Transcrição e tradução em tempo real iniciadas...")
stream.start_stream()

try:
    while stream.is_active():
        pass
except KeyboardInterrupt:
    print("Interrompido pelo usuário")
finally:
    stream.stop_streasm()
    stream.close()
    p.terminate()
