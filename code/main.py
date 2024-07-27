from llama_cpp import Llama
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import re
import pygame
import time

deviceProcessing = "cuda:0"
userPrompt = "Ви розмовляєте українською мовою, в міру лаконічний. Привітайтесь та запитайте як зовуть співрозмовника."
#systemPrompt = '<|user|>\\n{userPrompt}<|end|>\\n<|assistant|>'
filePathMP3 = "C:\\Projects\\VoiceAssistant\\code\\temp_audio\\temp.mp3"

model = Llama(
    model_path="D:\\lm-studio\\lmstudio-community\\Meta-Llama-3.1-8B-Instruct-GGUF\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    chat_format="llama-3",
    n_gpu_layers=6,
    flash_attn=True,
    n_threads=8,
    n_ctx=8192,
    device=deviceProcessing,
    verbose=True) # verbose=False - debug output off


chatMessages=[{"role": "system", "content": userPrompt}]

def speechToText():
    r = sr.Recognizer()
    m = sr.Microphone()

    with m as source: 
        r.adjust_for_ambient_noise(source, duration=0.2) 
    
    with m as source:
        audio = r.listen(source)
        TextFromAudio = ""

        try:
            TextFromAudio = r.recognize_google(audio, language="uk-UA")
            print(f"> {TextFromAudio}")
            
        except Exception as e:
            print("Скажіть шось..." + str(e))

    return TextFromAudio

def playAudio(filename):
    audio = AudioSegment.from_file(filename)
    octaves = 1.0
    new_sample_rate = int(audio.frame_rate * (1.25 ** octaves))
    hipitch_sound = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
    hipitch_sound = hipitch_sound.set_frame_rate(44100)
    
    #play(hipitch_sound)
    hipitch_sound.export(filename, format="mp3")
    
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    
    # Чекаємо, поки відтворення завершиться
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.unload()
    
    #if os.path.exists(filePathMP3):
    #    os.remove(filePathMP3)         

def textToSpeech(dataInput: str):
    tts = gTTS(text=dataInput, lang='uk')
    tts.save(filePathMP3)

def LLMResponce(userText: str): #-> str:
    new_message = {"role": "user", "content": userText}
    chatMessages.append(new_message)

    output = model.create_chat_completion(messages=chatMessages, 
                                          temperature=0.5, 
                                          max_tokens=1024, 
                                          stream=True)

    LLMTextResponce = ""
    count = 0

    for chunk in output:
        delta = chunk["choices"][0]["delta"]
        if "content" not in delta:
            continue
        print(delta["content"], end="", flush=True)
        
        LLMTextResponce += delta["content"]
        
        pattern = r"[;:.!?]+"
        elements = re.split(pattern, LLMTextResponce)
        if (len(elements)-1) > 0 and count < (len(elements)-1):
            tmp = elements[count]
            if len(tmp) > 2:
                textToSpeech(tmp)
                playAudio(filePathMP3)
            count += 1


    print()
    
    #textToSpeech(LLM_Responce)
    #playAudio(filePathMP3)

    new_message = {"role": "assistant", "content": LLMTextResponce}
    chatMessages.append(new_message)

while True:
    text = speechToText()
    #text = input(">> ")
    if text != "":
        LLMResponce(text)
