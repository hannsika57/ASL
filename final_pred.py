# Importing Libraries
import numpy as np
import math
import cv2

import traceback
import threading
import pyttsx3
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

hd = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 29

# ---- Theme ----
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT = "#4F8CFF"
ACCENT_HOVER = "#3B6FE0"
CARD_BG = "#1E1F26"
BG = "#15161B"
TEXT_MUTED = "#8B8D98"
DANGER = "#E5484D"
DANGER_HOVER = "#C93E42"


# Application :

class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.model = load_model('cnn8grps_rad1_model.h5')

        self.pts = None
        self.current_symbol = " "
        self.str = ""

        # ---------------- Window ----------------
        self.root = ctk.CTk()
        self.root.title("SignSpeak — Sign Language To Text & Speech")
        self.root.geometry("1280x780")
        self.root.minsize(1100, 700)
        self.root.configure(fg_color=BG)
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        # ---------------- Header ----------------
        header = ctk.CTkFrame(self.root, fg_color="transparent")
        header.pack(fill="x", padx=28, pady=(20, 10))

        title = ctk.CTkLabel(
            header, text="🤟 SignSpeak",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="bold"),
            text_color="white"
        )
        title.pack(side="left")

        subtitle = ctk.CTkLabel(
            header, text="Sign Language → Text → Speech",
            font=ctk.CTkFont(family="Segoe UI", size=14),
            text_color=TEXT_MUTED
        )
        subtitle.pack(side="left", padx=(14, 0), pady=(6, 0))

        # ---------------- Main content: two camera cards ----------------
        content = ctk.CTkFrame(self.root, fg_color="transparent")
        content.pack(fill="both", expand=True, padx=28, pady=10)
        content.grid_columnconfigure(0, weight=1)
        content.grid_columnconfigure(1, weight=1)
        content.grid_rowconfigure(0, weight=1)

        # -- Camera card --
        cam_card = ctk.CTkFrame(content, fg_color=CARD_BG, corner_radius=16)
        cam_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        ctk.CTkLabel(
            cam_card, text="Camera Feed",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color="white"
        ).pack(anchor="w", padx=18, pady=(16, 8))

        cam_holder = ctk.CTkFrame(cam_card, fg_color="#0F1014", corner_radius=12)
        cam_holder.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.panel = tk.Label(cam_holder, bg="#0F1014", bd=0)
        self.panel.pack(fill="both", expand=True, padx=6, pady=6)

        # -- Skeleton card --
        skel_card = ctk.CTkFrame(content, fg_color=CARD_BG, corner_radius=16)
        skel_card.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        ctk.CTkLabel(
            skel_card, text="Hand Landmarks",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            text_color="white"
        ).pack(anchor="w", padx=18, pady=(16, 8))

        skel_holder = ctk.CTkFrame(skel_card, fg_color="#0F1014", corner_radius=12)
        skel_holder.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        self.panel2 = tk.Label(skel_holder, bg="#0F1014", bd=0)
        self.panel2.pack(fill="both", expand=True, padx=6, pady=6)

        # ---------------- Character + Sentence panel ----------------
        bottom = ctk.CTkFrame(self.root, fg_color="transparent")
        bottom.pack(fill="x", padx=28, pady=(6, 10))
        bottom.grid_columnconfigure(0, weight=0)
        bottom.grid_columnconfigure(1, weight=1)

        # -- Character card --
        char_card = ctk.CTkFrame(bottom, fg_color=CARD_BG, corner_radius=16, width=160)
        char_card.grid(row=0, column=0, sticky="ns", padx=(0, 10))

        ctk.CTkLabel(
            char_card, text="CHARACTER",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=TEXT_MUTED
        ).pack(pady=(16, 0))

        self.panel3 = ctk.CTkLabel(
            char_card, text="—",
            font=ctk.CTkFont(family="Segoe UI", size=48, weight="bold"),
            text_color=ACCENT, width=120, height=80
        )
        self.panel3.pack(pady=(0, 16), padx=20)

        # -- Sentence card --
        sent_card = ctk.CTkFrame(bottom, fg_color=CARD_BG, corner_radius=16)
        sent_card.grid(row=0, column=1, sticky="nsew")

        ctk.CTkLabel(
            sent_card, text="SENTENCE",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=TEXT_MUTED
        ).pack(anchor="w", padx=20, pady=(16, 4))

        self.panel5 = ctk.CTkLabel(
            sent_card, text=" ",
            font=ctk.CTkFont(family="Segoe UI", size=22),
            text_color="white", anchor="w", justify="left",
            wraplength=760
        )
        self.panel5.pack(fill="x", padx=20, pady=(0, 16))

        # ---------------- Controls ----------------
        controls = ctk.CTkFrame(self.root, fg_color="transparent")
        controls.pack(fill="x", padx=28, pady=(0, 16))

        def make_button(parent, text, command, fg=ACCENT, hover=ACCENT_HOVER, width=120):
            return ctk.CTkButton(
                parent, text=text, command=command,
                font=ctk.CTkFont(family="Segoe UI", size=14, weight="bold"),
                fg_color=fg, hover_color=hover,
                corner_radius=10, height=42, width=width
            )

        make_button(controls, "＋ Add Letter", self.enter_fun).pack(side="left", padx=(0, 10))
        make_button(controls, "⎵ Space", self.space_fun, fg="#2C2D36", hover="#3A3B46").pack(side="left", padx=(0, 10))
        make_button(controls, "⌫ Backspace", self.backspace_fun, fg="#2C2D36", hover="#3A3B46").pack(side="left", padx=(0, 10))
        make_button(controls, "🗑 Clear", self.clear_fun, fg=DANGER, hover=DANGER_HOVER).pack(side="left", padx=(0, 10))
        make_button(controls, "🔊 Speak", self.speak_fun, width=140).pack(side="right")

        hint = ctk.CTkLabel(
            self.root,
            text="Shortcuts:  Enter = add letter    Space = add space    Backspace = delete    Esc = quit",
            font=ctk.CTkFont(family="Segoe UI", size=11),
            text_color=TEXT_MUTED
        )
        hint.pack(pady=(0, 14))

        # --- Keyboard controls ---
        self.root.bind('<Return>', self.enter_fun)
        self.root.bind('<BackSpace>', self.backspace_fun)
        self.root.bind('<space>', self.space_fun)
        self.root.bind('<Escape>', lambda e: self.destructor())

        self.video_loop()

    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok:
                self.root.after(1, self.video_loop)
                return

            cv2image = cv2.flip(frame, 1)
            hands = hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy = np.array(cv2image)
            cv2image_rgb = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)
            self.current_image = Image.fromarray(cv2image_rgb)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands and hands[0]:
                hand = hands[0][0]
                x, y, w, h = hand['bbox']
                image = cv2image_copy[y - offset:y + h + offset, x - offset:x + w + offset]

                white = cv2.imread("white.jpg")

                if white is not None and image.size != 0:
                    handz = hd2.findHands(image, draw=False, flipType=True)
                    if handz and handz[0]:
                        handmap = handz[0][0]
                        self.pts = handmap['lmList']

                        os_x = ((400 - w) // 2) - 15
                        os_y = ((400 - h) // 2) - 15
                        for t in range(0, 4):
                            cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                     (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        for t in range(5, 8):
                            cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                     (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        for t in range(9, 12):
                            cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                     (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        for t in range(13, 16):
                            cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                     (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        for t in range(17, 20):
                            cv2.line(white, (self.pts[t][0] + os_x, self.pts[t][1] + os_y),
                                     (self.pts[t + 1][0] + os_x, self.pts[t + 1][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[5][0] + os_x, self.pts[5][1] + os_y),
                                 (self.pts[9][0] + os_x, self.pts[9][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[9][0] + os_x, self.pts[9][1] + os_y),
                                 (self.pts[13][0] + os_x, self.pts[13][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[13][0] + os_x, self.pts[13][1] + os_y),
                                 (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y),
                                 (self.pts[5][0] + os_x, self.pts[5][1] + os_y), (0, 255, 0), 3)
                        cv2.line(white, (self.pts[0][0] + os_x, self.pts[0][1] + os_y),
                                 (self.pts[17][0] + os_x, self.pts[17][1] + os_y), (0, 255, 0), 3)

                        for i in range(21):
                            cv2.circle(white, (self.pts[i][0] + os_x, self.pts[i][1] + os_y), 2, (0, 0, 255), 1)

                        res = white
                        self.predict(res)

                        self.current_image2 = Image.fromarray(res)
                        imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
                        self.panel2.imgtk = imgtk2
                        self.panel2.config(image=imgtk2)

                        self.panel3.configure(text=self.current_symbol)

            self.panel5.configure(text=self.str if self.str.strip() else " ")

        except Exception:
            print("==", traceback.format_exc())
        finally:
            self.root.after(1, self.video_loop)

    def distance(self, x, y):
        return math.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

    # --- Manual sentence controls (keyboard + button driven) ---
    def enter_fun(self, event=None):
        """Append the currently displayed character to the sentence."""
        if self.current_symbol and self.current_symbol.strip() != "":
            self.str += self.current_symbol
            self.panel5.configure(text=self.str if self.str.strip() else " ")

    def backspace_fun(self, event=None):
        self.str = self.str[:-1]
        self.panel5.configure(text=self.str if self.str.strip() else " ")

    def space_fun(self, event=None):
        self.str += " "
        self.panel5.configure(text=self.str if self.str.strip() else " ")

    def speak_fun(self):
        text = self.str.strip()
        if not text:
            return
        threading.Thread(target=self._speak_worker, args=(text,), daemon=True).start()

    def _speak_worker(self, text):
        engine = pyttsx3.init()
        engine.setProperty("rate", 130)
        voices = engine.getProperty("voices")
        if voices:
            engine.setProperty("voice", voices[0].id)
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    def clear_fun(self):
        self.str = ""
        self.panel5.configure(text=" ")

    def predict(self, test_image):
        white = test_image
        white = white.reshape(1, 400, 400, 3)
        prob = np.array(self.model.predict(white)[0], dtype='float32')
        ch1 = np.argmax(prob, axis=0)
        prob[ch1] = 0
        ch2 = np.argmax(prob, axis=0)
        prob[ch2] = 0
        ch3 = np.argmax(prob, axis=0)
        prob[ch3] = 0

        pl = [ch1, ch2]

        # condition for [Aemnst]
        l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
             [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
             [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 0

        # condition for [o][s]
        l = [[2, 2], [2, 1]]
        if pl in l:
            if (self.pts[5][0] < self.pts[4][0]):
                ch1 = 0

        # condition for [c0][aemnst]
        l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[4][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.pts[5][0] > self.pts[4][0]:
                ch1 = 2

        # condition for [c0][aemnst]
        l = [[6, 0], [6, 6], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) < 52:
                ch1 = 2

        # condition for [gh][bdfikruvw]
        l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1] and self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 3

        # con for [gh][l]
        l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 3

        # con for [gh][pqz]
        l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[2][1] + 15 < self.pts[16][1]:
                ch1 = 3

        # con for [l][x]
        l = [[6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) > 55:
                ch1 = 4

        # con for [l][d]
        l = [[1, 4], [1, 6], [1, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) > 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 4

        # con for [l][gh]
        l = [[3, 6], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[0][0]):
                ch1 = 4

        # con for [l][c0]
        l = [[2, 2], [2, 5], [2, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[1][0] < self.pts[12][0]):
                ch1 = 4

        # con for [gh][z]
        l = [[3, 6], [3, 5], [3, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] > self.pts[10][1]:
                ch1 = 5

        # con for [gh][pq]
        l = [[3, 2], [3, 1], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][1] + 17 > self.pts[8][1] and self.pts[4][1] + 17 > self.pts[12][1] and self.pts[4][1] + 17 > self.pts[16][1] and self.pts[4][1] + 17 > self.pts[20][1]:
                ch1 = 5

        # con for [l][pqz]
        l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[4][0] > self.pts[0][0]:
                ch1 = 5

        # con for [pqz][aemnst]
        l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[0][0] < self.pts[8][0] and self.pts[0][0] < self.pts[12][0] and self.pts[0][0] < self.pts[16][0] and self.pts[0][0] < self.pts[20][0]:
                ch1 = 5

        # con for [pqz][yj]
        l = [[5, 7], [5, 2], [5, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[3][0] < self.pts[0][0]:
                ch1 = 7

        # con for [l][yj]
        l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] < self.pts[8][1]:
                ch1 = 7

        # con for [x][yj]
        l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] > self.pts[20][1]:
                ch1 = 7

        # condition for [x][aemnst]
        l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] > self.pts[16][0]:
                ch1 = 6

        # condition for [yj][x]
        l = [[7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[18][1] < self.pts[20][1] and self.pts[8][1] < self.pts[10][1]:
                ch1 = 6

        # condition for [c0][x]
        l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[8], self.pts[16]) > 50:
                ch1 = 6

        # con for [l][x]
        l = [[4, 6], [4, 2], [4, 1], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if self.distance(self.pts[4], self.pts[11]) < 60:
                ch1 = 6

        # con for [x][d]
        l = [[1, 4], [1, 6], [1, 0], [1, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 > 0:
                ch1 = 6

        # con for [b][pqz]
        l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
             [6, 3], [6, 4], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [f][pqz]
        l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
             [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 1

        # con for [d][pqz]
        l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[4][1] > self.pts[14][1]):
                ch1 = 1

        l = [[4, 1], [4, 2], [4, 4]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.distance(self.pts[4], self.pts[11]) < 50) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 1

        l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[2][0] < self.pts[0][0]) and self.pts[14][1] < self.pts[4][1]):
                ch1 = 1

        l = [[6, 6], [6, 4], [6, 1], [6, 2]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[5][0] - self.pts[4][0] - 15 < 0:
                ch1 = 1

        # con for [i][pqz]
        l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                ch1 = 1

        # con for [yj][bfdi]
        l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
        pl = [ch1, ch2]
        if pl in l:
            if (self.pts[4][0] < self.pts[5][0] + 15) and (
            (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1])):
                ch1 = 7

        # con for [uvr]
        l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if ((self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1])) and self.pts[4][1] > self.pts[14][1]:
                ch1 = 1

        # con for [w]
        fg = 13
        l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
        pl = [ch1, ch2]
        if pl in l:
            if not (self.pts[0][0] + fg < self.pts[8][0] and self.pts[0][0] + fg < self.pts[12][0] and self.pts[0][0] + fg < self.pts[16][0] and self.pts[0][0] + fg < self.pts[20][0]) and not (
                    self.pts[0][0] > self.pts[8][0] and self.pts[0][0] > self.pts[12][0] and self.pts[0][0] > self.pts[16][0] and self.pts[0][0] > self.pts[20][0]) and self.distance(self.pts[4], self.pts[11]) < 50:
                ch1 = 1

        # con for [w]
        l = [[5, 0], [5, 5], [0, 1]]
        pl = [ch1, ch2]
        if pl in l:
            if self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1]:
                ch1 = 1

        # -------------------------condn for 8 groups ends
        # -------------------------condn for subgroups starts

        if ch1 == 0:
            ch1 = 'S'
            if self.pts[4][0] < self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0]:
                ch1 = 'A'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] < self.pts[10][0] and self.pts[4][0] < self.pts[14][0] and self.pts[4][0] < self.pts[18][0] and self.pts[4][1] < self.pts[14][1] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'T'
            if self.pts[4][1] > self.pts[8][1] and self.pts[4][1] > self.pts[12][1] and self.pts[4][1] > self.pts[16][1] and self.pts[4][1] > self.pts[20][1]:
                ch1 = 'E'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][0] > self.pts[14][0] and self.pts[4][1] < self.pts[18][1]:
                ch1 = 'M'
            if self.pts[4][0] > self.pts[6][0] and self.pts[4][0] > self.pts[10][0] and self.pts[4][1] < self.pts[18][1] and self.pts[4][1] < self.pts[14][1]:
                ch1 = 'N'

        if ch1 == 2:
            if self.distance(self.pts[12], self.pts[4]) > 42:
                ch1 = 'C'
            else:
                ch1 = 'O'

        if ch1 == 3:
            if (self.distance(self.pts[8], self.pts[12])) > 72:
                ch1 = 'G'
            else:
                ch1 = 'H'

        if ch1 == 7:
            if self.distance(self.pts[8], self.pts[4]) > 42:
                ch1 = 'Y'
            else:
                ch1 = 'J'

        if ch1 == 4:
            ch1 = 'L'

        if ch1 == 6:
            ch1 = 'X'

        if ch1 == 5:
            if self.pts[4][0] > self.pts[12][0] and self.pts[4][0] > self.pts[16][0] and self.pts[4][0] > self.pts[20][0]:
                if self.pts[8][1] < self.pts[5][1]:
                    ch1 = 'Z'
                else:
                    ch1 = 'Q'
            else:
                ch1 = 'P'

        if ch1 == 1:
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'B'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'D'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'F'
            if (self.pts[6][1] < self.pts[8][1] and self.pts[10][1] < self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] > self.pts[20][1]):
                ch1 = 'I'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] > self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'W'
            if (self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and self.pts[4][1] < self.pts[9][1]:
                ch1 = 'K'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) < 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'U'
            if ((self.distance(self.pts[8], self.pts[12]) - self.distance(self.pts[6], self.pts[10])) >= 8) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]) and (self.pts[4][1] > self.pts[9][1]):
                ch1 = 'V'
            if (self.pts[8][0] > self.pts[12][0]) and (
                    self.pts[6][1] > self.pts[8][1] and self.pts[10][1] > self.pts[12][1] and self.pts[14][1] < self.pts[16][1] and self.pts[18][1] < self.pts[20][1]):
                ch1 = 'R'

        # Final classified letter shown in the Character box.
        self.current_symbol = ch1

    def destructor(self):
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


print("Starting Application...")
(Application()).root.mainloop()