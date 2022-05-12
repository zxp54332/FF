from tkinter import *
from os import _exit as ex
import socket
import os
import sys
from scipy.io import wavfile
import pygame as py
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
#import threading
#初始化
py.mixer.init()



def callbackFunc():
    text = Entry.get()  # 取得文字
    if not text:
        return 0
    resultButton2.configure(state="active")
    lang = var.get()
    model = var2.get()
    sp = speech_regulator.get()
    if (sp*100) % 10 == 0 or sp == 1.1:
        sp-=0.01
        
    if lang == "中文":
        port = 4000
    elif lang == "英文":
        port = 3000

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(('***.***.**.***', port))    # Enter your IP
    except socket.error as msg:
        print(msg)
        print(sys.exit(1))
    s.send(text.encode())    # 傳送文字
    time.sleep(0.001)
    s.send(str(sp).encode())
    time.sleep(0.001)
    s.send(model.encode())
    
    with open("speech.wav",'w') as f:  # 以空白覆蓋檔案
        f.close     
    with open("speech.wav",'ab') as f:      
        while True:
            if  model == "FFT":
                time.sleep(0.1)
                wav = s.recv(65536)
                f.write(wav)
                if not wav:
                    break
            else:
                time.sleep(0.1)
                wav = s.recv(65536)
                f.write(wav)
                if not wav:
                    break
    f.close()       
    wav = py.mixer.Sound('speech.wav')  #文件加載
    wav.play()
    s.close()
      

def callbackFunc2():
    if os.path.exists('speech.wav'):
        wav = py.mixer.Sound('speech.wav')  #文件加載
        wav.play()
    else:
        resultButton2.configure(state="disabled")
    
def close():
    ex(0)

   
window = Tk()
window.title("Text-to-Speech")
window.geometry("700x350")


# Label
label = Label(window, text="請輸入文字", font=("Lucida Grande", 20))
label.place(x = 60, y = 40)
label2 = Label(window, text="語速", font=("Lucida Grande", 20))
label2.place(x = 325, y = 305)
label3 = Label(window, text="請輸入 \"對應\" 的語言，標點符號：英文為\"半形\"、中文為\"全形\"", 
               font=("Lucida Grande", 15), bg="yellow", fg="red")
label3.place(x = 60, y = 80)


# Text entry
Entry = Entry(window, font=("Lucida Grande", 20))
Entry.place(x = 220, y = 40)


# Dropdown menu
OptionList = ["中文", "英文"]
OptionList2 = ["FFT", "TransformerTTS"]
var = StringVar(window)
var.set("中文") # default value
var2 = StringVar(window)
var2.set("FFT")
opt = OptionMenu(window, var, *OptionList)
opt2 = OptionMenu(window, var2, *OptionList2)
opt.configure(font=("Lucida Grande", 20))
opt2.configure(font=("Lucida Grande", 20))
opt.place(x = 60, y = 125)
opt2.place(x = 60, y = 185)
menu = window.nametowidget(opt.menuname)
menu.config(font=("Lucida Grande", 20))
menu2 = window.nametowidget(opt2.menuname)
menu2.config(font=("Lucida Grande", 20))


# Scale
sp_var = IntVar(window)
sp_var.set(1)
speech_regulator = Scale(window, from_=0.35, to=2, orient="horizontal", resolution=0.01, 
                         length=200, width=25, sliderrelief="flat", font=("Lucida Grande", 20))
speech_regulator.place(x = 245, y = 230)

# Button
resultButton = Button(window, text = '     合成     ', command=callbackFunc)
resultButton.configure(font=("Lucida Grande", 20))
resultButton.place(x = 200, y = 125)

resultButton2 = Button(window, text = '     重播     ', state="disabled", command=callbackFunc2)
resultButton2.configure(font=("Lucida Grande", 20))
resultButton2.place(x = 360, y = 125)

exitButton = Button(window, text = '       Exit       ', command=close)
exitButton.configure(font=("Lucida Grande", 18))
exitButton.place(x = 530, y = 240)

window.mainloop()
