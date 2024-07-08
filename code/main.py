from llama_cpp import Llama
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play
import re

Device = "cuda:0"
user_prompt = "Ви шанобливий помічник, розмовляєте українською мовою, коротко та лаконічно. Привітайтесь та запитайте як зовуть співрозмовника."
#prompt_template = "<｜begin▁of▁sentence｜>User: {user_message_1}\n\nAssistant: {assistant_message_1}<｜end▁of▁sentence｜>User: {user_message_2}\n\nAssistant:\n"

model = Llama(
    model_path="D:\\lm-studio\\lmstudio-community\\Meta-Llama-3-8B-Instruct-GGUF\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    chat_format="llama-3",
    n_gpu_layers=13,
    flash_attn=True,
    n_threads=8,
    n_ctx=8192,
    device=Device,
    verbose=False) # verbose=False - debug output off

# Simple inference example
#output = model(prompt_template, # Prompt
#    stop=["<｜end▁of▁sentence｜>"],   # Example stop token - not necessarily correct for this specific model! Please check before using.
#  #echo=True        # Whether to echo the prompt
#)

FilePathMP3 = "C:\\Projects\\vscode-basics\\GoIT-Python-Data-Science\\MyProjects\\VoiceAssistant\\temp.mp3"

messages=[{"role": "system", "content": user_prompt}]

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio, language="uk-UA")
            print(f"> {said}")
        except Exception as e:
            print("Скажіть шось..." + str(e))

    return said

def play_audio(filename):
    audio = AudioSegment.from_file(filename)
    octaves = 1.0
    new_sample_rate = int(audio.frame_rate * (1.25 ** octaves))
    hipitch_sound = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
    hipitch_sound = hipitch_sound.set_frame_rate(44100)
    play(hipitch_sound)

    #if os.path.exists(FilePathMP3):
    #    os.remove(FilePathMP3)         

def AI_speack(userText: str): #-> str:
    new_message = {"role": "user", "content": userText}
    messages.append(new_message)

    output = model.create_chat_completion(messages, temperature=0.5, max_tokens=1024, stream=True)

    LLM_Responce = ""
    count = 0

    for chunk in output:
        delta = chunk["choices"][0]["delta"]
        if "content" not in delta:
            continue
        print(delta["content"], end="", flush=True)
        
        LLM_Responce += delta["content"]
        pattern = r"[;:.!?]+"
        elements = re.split(pattern, LLM_Responce)
        if (len(elements)-1) > 0 and count < (len(elements)-1):
            tmp = elements[count]
            if len(tmp) > 2:
                tts = gTTS(text=tmp, lang='uk')
                tts.save(FilePathMP3)
                play_audio(FilePathMP3)
            count += 1


    print()
    
    #tts = gTTS(text=LLM_Responce, lang='uk')
    #tts.save(FilePathMP3)
    #play_audio(FilePathMP3)

    new_message = {"role": "assistant", "content": LLM_Responce}
    messages.append(new_message)

    #print(messages)
    #return LLM_Responce

while True:
    text = get_audio()
    if text != "":
        AI_speack(text)
