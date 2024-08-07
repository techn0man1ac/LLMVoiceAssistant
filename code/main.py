from llama_cpp import Llama
import keyboard
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment
# from pydub.playback import play
import re
import time
from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame

userPrompt = "Ви завжди відповідаєте українською мовою. Ви завжди форматуєте текст без використання двох зірочок(наприклад так '**'), інакше будете покарані. Ви - корисний, шанобливий і чесний помічник, використовуєйте лаконізацію. Завжди відповідайте максимально корисно, але в той же час безпечно, використовуйте смайлики у відповідях. Ваші відповіді не повинні містити шкідливого, неетичного, расистського, сексистського, токсичного, небезпечного або незаконного контенту. Будь ласка, переконайтеся, що ваші відповіді є соціально неупередженими та позитивними за своєю суттю. Якщо питання не має сенсу або не відповідає дійсності, поясніть, чому, замість того, щоб відповідати неправильно. Якщо ви не знаєте відповіді на запитання, будь ласка, не поширюйте неправдиву інформацію."
filePathMP3 = "C:\\Projects\\VoiceAssistant\\code\\temp_audio\\temp.mp3"

# Температура відповідей та інші налаштуван+ня моделі
responcesTtemperature = 0.3
maxOutputTokens = 512

text = "Привітайтеся та запитайте ім'я у Вашого співрозмовника. Використовуй його ім'я у розмовах з співрозмовниом як омога частіше."

# Ініціалізуємо модель Llama з вказаним шляхом до моделі та налаштуваннями
model = Llama(
    model_path="D:\\lm-studio\\bartowski\\Mistral-Nemo-Instruct-2407-GGUF\\Mistral-Nemo-Instruct-2407-Q4_K_M.gguf",
    #chat_format="llama-3",
    n_gpu_layers=11,
    flash_attn=True,
    n_threads=8,
    n_ctx=13100,
    verbose=False) # verbose=False - вимкнення дебагуючого виведення

chatMessages=[{"role": "system", "content": userPrompt}]

# Функція для перетворення мови в текст
def speechToText():
    recognize = sr.Recognizer()
    microphone = sr.Microphone(chunk_size = 2048)

    with microphone as sourceMic: 
        recognize.adjust_for_ambient_noise(sourceMic, duration=0.2) 
    
    with microphone as sourceMic:
        audio = recognize.listen(source = sourceMic)
        TextFromAudio = ""

        try:
            TextFromAudio = recognize.recognize_google(audio, language="uk-UA")
            print(f"> {TextFromAudio}")
        
        except Exception as e:
            print("Скажіть шось..." + str(e))

    return TextFromAudio

# Функція для відтворення аудіофайлу
def playAudio(filename):
    audio = AudioSegment.from_file(filename)
    octaves = 1.0
    new_sample_rate = int(audio.frame_rate * (1.25 ** octaves))
    hipitch_sound = audio._spawn(audio.raw_data, overrides={'frame_rate': new_sample_rate})
    hipitch_sound = hipitch_sound.set_frame_rate(44100)
    
    hipitch_sound.export(filename, format="mp3")
    
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.unload()

# Функція для перетворення тексту в голос
def textToSpeech(dataInput: str):
    tts = gTTS(text=dataInput, lang='uk')
    tts.save(filePathMP3)
    playAudio(filePathMP3)

# Функція для отримання відповіді від моделі Llama
def LLMResponce(userText: str): 
    new_message = {"role": "user", "content": userText}
    chatMessages.append(new_message)

    output = model.create_chat_completion(messages=chatMessages, 
                                          temperature=responcesTtemperature, 
                                          max_tokens=maxOutputTokens, 
                                          stream=True)

    LLMTextResponce = ""
    count = 0
    
    for chunk in output: 

        # Check for 'q' key press to break the loop
        if keyboard.is_pressed("q"):
            tempText = "..."
            print(tempText)
            LLMTextResponce += tempText
            break    
        else:
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
                    #pass
                count += 1

    print()
    
    #textToSpeech(LLMTextResponce)

    new_message = {"role": "assistant", "content": LLMTextResponce}
    chatMessages.append(new_message)

if __name__ == "__main__":
    # Основний цикл програми
    while True: 
        if text != "":
            LLMResponce(text)
            
        text = speechToText()
       
        match text:
            case "текст будь ласка":
                textUseMessage = "Текстовий режим активовано:"
                print(textUseMessage)
                textToSpeech(textUseMessage)
                text = input(">> ")
            
            case "бувай":
                LLMResponce("[Ім'я користувача] прощається з тобою, тому твоя відповідь має бути коротка.")
                break
