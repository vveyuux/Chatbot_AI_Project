from tkinter import *
from chatbot import predict_class, get_response

import json

import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

datastore = json.loads(open('jsons/intents.json').read())
emotionData = json.loads(open('jsons/emotions.json').read())

BG_NAVI = '#1A374D'
BG_COLOR = '#406882'
NAME_COLOR = '#6998AB'
TEXT_COLOR = '#B1D0E0'

FONT = 'Calibri 14'
FONT_BOLD = 'Calibri 13 bold'

class ChatApp:
    
    def __init__(self):
        self.window = Tk()
        self.name  = ''
        self._setup_main_window()

    def _setup_main_window(self):
        self.window.title('Chatbot Project')
        self.window.resizable(width=False, height=False)
        self.window.configure(width=470, height=550, bg=BG_NAVI)
        self.window.iconbitmap('assets/bot.ico')

        head_label = Label(self.window, bg=BG_NAVI, fg=TEXT_COLOR, text='Emotion in text : Chatbot', font=FONT_BOLD, pady=10)

        head_label.place(relwidth=1)

        line = Label(self.window, width=450, bg=BG_COLOR)
        line.place(relwidth=1, rely=0.07, relheight=0.012)

        self.text_widget = Text(self.window, width=20, height=2, bg=BG_NAVI, fg=TEXT_COLOR, padx=5, pady=5, font=FONT_BOLD)
        self.text_widget.place(relheight=0.745, relwidth=1, rely=0.08)
        self.text_widget.configure(cursor='arrow', state=DISABLED)

        scrollbar = Scrollbar(self.text_widget)
        scrollbar.place(relheight=1, relx=0.974)
        scrollbar.configure(command=self.text_widget.yview)

        bottom_label = Label(self.window, bg=BG_NAVI, height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        
        self.msg_entry = Entry(bottom_label, bg=BG_COLOR, fg=TEXT_COLOR, font=FONT)
        self.msg_entry.place(relwidth=0.74, relheight=0.06, rely=0.008, relx=0.011)
        self.msg_entry.focus()
        self.msg_entry.bind('<Return>', self._on_enter_pressed)

        send_button = Button(bottom_label, text='Send', font=FONT_BOLD, width=20, command=lambda: self._on_enter_pressed(None))
        send_button.place(relx=0.77, rely=0.008, relheight=0.06, relwidth=0.22)

        if self.name == '':
            self.text_widget.configure(state=NORMAL)
            self.text_widget.insert(END, f'BOT: What is your name?\n\n')
            self.text_widget.configure(state=DISABLED)

    def _on_enter_pressed(self,event):
        msg = self.msg_entry.get()
        self._insert_message(msg, self.name)

    def _insert_message(self, msg, sender):
        if not msg:
            return
        
        if self.name == '':
            if not msg:
                return
            
            self.name = msg.upper()
            msg_bot = f'BOT: Okay, Your name is {self.name}\n\n'
            self.text_widget.configure(state=NORMAL)
            self.text_widget.insert(END, msg_bot)
            msg_bot = f'BOT: Please, enjoy XD\n\n'
            self.text_widget.insert(END, msg_bot)
            self.text_widget.configure(state=DISABLED)
            self.msg_entry.delete(0,END)
            return

        if predict_class(msg)[0]['intent'] == 'goodbye':
            self.msg_entry.delete(0, END)
            ints = predict_class(msg)
            msg_bot = f'BOT: {get_response(ints, datastore, emotionData)}\n\n'
            self.text_widget.configure(state=NORMAL)
            self.text_widget.insert(END, msg_bot)
            msg_bot = f'BOT: Shutdown - -\n\n'
            self.text_widget.insert(END, msg_bot)
            self.text_widget.configure(state=DISABLED)
            self.exit()

        self.msg_entry.delete(0, END)
        msg_sender = f'{sender}: {msg}\n\n'
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg_sender)
        self.text_widget.configure(state=DISABLED)

        ints = predict_class(msg)
        msg_bot = f'BOT: {get_response(ints, datastore, emotionData)}\n\n'
        self.text_widget.configure(state=NORMAL)
        self.text_widget.insert(END, msg_bot)
        self.text_widget.configure(state=DISABLED)

        self.text_widget.see(END)

    def run(self):
        self.window.mainloop()

    def exit(self):
        self.window.destroy()

if __name__ == '__main__':
        app = ChatApp()
        app.run()