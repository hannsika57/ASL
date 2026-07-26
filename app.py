import json
import math
import threading

import av
import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from cvzone.HandTrackingModule import HandDetector
from keras.models import load_model
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

# ----------------------------------------------------------------------
# Page config + styling
# ----------------------------------------------------------------------
st.set_page_config(page_title="SignSpeak", page_icon="🤟", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background-color: #15161B; }
        .header-title { font-size: 34px; font-weight: 800; color: white; margin-bottom: 0px; }
        .header-sub { font-size: 15px; color: #8B8D98; margin-top: 2px; margin-bottom: 18px; }
        .card-title { font-size: 13px; font-weight: 700; color: #8B8D98; letter-spacing: 1px;
                      text-transform: uppercase; margin-bottom: 8px; }
        .char-card {
            background: #1E1F26; border-radius: 16px; padding: 18px 0;
            text-align: center; font-size: 56px; font-weight: 800; color: #4F8CFF;
            min-height: 90px; display: flex; align-items: center; justify-content: center;
        }
        .sentence-card {
            background: #1E1F26; border-radius: 16px; padding: 20px 22px;
            font-size: 22px; color: white; min-height: 90px; display: flex; align-items: center;
            word-wrap: break-word;
        }
        div.stButton > button {
            border-radius: 10px; font-weight: 700; height: 3em; border: none;
        }
        hr { border-color: #2C2D36; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='header-title'>🤟 SignSpeak</div>", unsafe_allow_html=True)
st.markdown("<div class='header-sub'>Sign Language → Text → Speech (runs locally in your browser)</div>", unsafe_allow_html=True)

MODEL_PATH = "cnn8grps_rad1_model.h5"
WHITE_IMG_PATH = "white.jpg"
OFFSET = 29

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": [
                    "turn:openrelay.metered.ca:80?transport=tcp",
                    "turn:openrelay.metered.ca:443?transport=tcp",
                    "turns:openrelay.metered.ca:443?transport=tcp",
                ],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)


@st.cache_resource
def load_assets():
    return load_model(MODEL_PATH)


model = load_assets()


def distance(a, b):
    return math.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))


def draw_skeleton(white, pts, os_x, os_y):
    for t in range(0, 4):
        cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
    for t in range(5, 8):
        cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
    for t in range(9, 12):
        cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
    for t in range(13, 16):
        cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
    for t in range(17, 20):
        cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
    cv2.line(white, (pts[5][0] + os_x, pts[5][1] + os_y), (pts[9][0] + os_x, pts[9][1] + os_y), (0, 255, 0), 3)
    cv2.line(white, (pts[9][0] + os_x, pts[9][1] + os_y), (pts[13][0] + os_x, pts[13][1] + os_y), (0, 255, 0), 3)
    cv2.line(white, (pts[13][0] + os_x, pts[13][1] + os_y), (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
    cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), (pts[5][0] + os_x, pts[5][1] + os_y), (0, 255, 0), 3)
    cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
    for i in range(21):
        cv2.circle(white, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)


def classify(pts, model, white):
    """Same CNN + geometric-heuristic classifier as the desktop app, parameterized on pts/white."""
    img = white.reshape(1, 400, 400, 3)
    prob = np.array(model.predict(img, verbose=0)[0], dtype='float32')
    ch1 = np.argmax(prob, axis=0)
    prob[ch1] = 0
    ch2 = np.argmax(prob, axis=0)
    prob[ch2] = 0
    ch3 = np.argmax(prob, axis=0)
    prob[ch3] = 0

    pl = [ch1, ch2]

    l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
         [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
         [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 0

    l = [[2, 2], [2, 1]]
    if pl in l:
        if (pts[5][0] < pts[4][0]):
            ch1 = 0

    l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
            ch1 = 2

    l = [[6, 0], [6, 6], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) < 52:
            ch1 = 2

    l = [[1, 4], [1, 5], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1] and pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 3

    l = [[4, 6], [4, 1], [4, 5], [4, 3], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 3

    l = [[5, 3], [5, 0], [5, 7], [5, 4], [5, 2], [5, 1], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[2][1] + 15 < pts[16][1]:
            ch1 = 3

    l = [[6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) > 55:
            ch1 = 4

    l = [[1, 4], [1, 6], [1, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) > 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 4

    l = [[3, 6], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[0][0]):
            ch1 = 4

    l = [[2, 2], [2, 5], [2, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[1][0] < pts[12][0]):
            ch1 = 4

    l = [[3, 6], [3, 5], [3, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] > pts[10][1]:
            ch1 = 5

    l = [[3, 2], [3, 1], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][1] + 17 > pts[8][1] and pts[4][1] + 17 > pts[12][1] and pts[4][1] + 17 > pts[16][1] and pts[4][1] + 17 > pts[20][1]:
            ch1 = 5

    l = [[4, 4], [4, 5], [4, 2], [7, 5], [7, 6], [7, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[4][0] > pts[0][0]:
            ch1 = 5

    l = [[0, 2], [0, 6], [0, 1], [0, 5], [0, 0], [0, 7], [0, 4], [0, 3], [2, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[0][0] < pts[8][0] and pts[0][0] < pts[12][0] and pts[0][0] < pts[16][0] and pts[0][0] < pts[20][0]:
            ch1 = 5

    l = [[5, 7], [5, 2], [5, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[3][0] < pts[0][0]:
            ch1 = 7

    l = [[4, 6], [4, 2], [4, 4], [4, 1], [4, 5], [4, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] < pts[8][1]:
            ch1 = 7

    l = [[6, 7], [0, 7], [0, 1], [0, 0], [6, 4], [6, 6], [6, 5], [6, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] > pts[20][1]:
            ch1 = 7

    l = [[0, 4], [0, 2], [0, 3], [0, 1], [0, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] > pts[16][0]:
            ch1 = 6

    l = [[7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[18][1] < pts[20][1] and pts[8][1] < pts[10][1]:
            ch1 = 6

    l = [[2, 1], [2, 2], [2, 6], [2, 7], [2, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[8], pts[16]) > 50:
            ch1 = 6

    l = [[4, 6], [4, 2], [4, 1], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if distance(pts[4], pts[11]) < 60:
            ch1 = 6

    l = [[1, 4], [1, 6], [1, 0], [1, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 > 0:
            ch1 = 6

    l = [[5, 0], [5, 1], [5, 4], [5, 5], [5, 6], [6, 1], [7, 6], [0, 2], [7, 1], [7, 4], [6, 6], [7, 2], [5, 0],
         [6, 3], [6, 4], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[6, 1], [6, 0], [0, 3], [6, 4], [2, 2], [0, 6], [6, 2], [7, 6], [4, 6], [4, 1], [4, 2], [0, 2], [7, 1],
         [7, 4], [6, 6], [7, 2], [7, 5], [7, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[6, 1], [6, 0], [4, 2], [4, 1], [4, 6], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 1

    l = [[5, 0], [3, 4], [3, 0], [3, 1], [3, 5], [5, 5], [5, 4], [5, 1], [7, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[4][1] > pts[14][1]):
            ch1 = 1

    l = [[4, 1], [4, 2], [4, 4]]
    pl = [ch1, ch2]
    if pl in l:
        if (distance(pts[4], pts[11]) < 50) and (
                pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 1

    l = [[3, 4], [3, 0], [3, 1], [3, 5], [3, 6]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[2][0] < pts[0][0]) and pts[14][1] < pts[4][1]):
            ch1 = 1

    l = [[6, 6], [6, 4], [6, 1], [6, 2]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[5][0] - pts[4][0] - 15 < 0:
            ch1 = 1

    l = [[5, 4], [5, 5], [5, 1], [0, 3], [0, 7], [5, 0], [0, 2], [6, 2], [7, 5], [7, 1], [7, 6], [7, 7]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
            ch1 = 1

    l = [[1, 5], [1, 7], [1, 1], [1, 6], [1, 3], [1, 0]]
    pl = [ch1, ch2]
    if pl in l:
        if (pts[4][0] < pts[5][0] + 15) and (
        (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1])):
            ch1 = 7

    l = [[5, 5], [5, 0], [5, 4], [5, 1], [4, 6], [4, 1], [7, 6], [3, 0], [3, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if ((pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1])) and pts[4][1] > pts[14][1]:
            ch1 = 1

    fg = 13
    l = [[3, 5], [3, 0], [3, 6], [5, 1], [4, 1], [2, 0], [5, 0], [5, 5]]
    pl = [ch1, ch2]
    if pl in l:
        if not (pts[0][0] + fg < pts[8][0] and pts[0][0] + fg < pts[12][0] and pts[0][0] + fg < pts[16][0] and pts[0][0] + fg < pts[20][0]) and not (
                pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and distance(pts[4], pts[11]) < 50:
            ch1 = 1

    l = [[5, 0], [5, 5], [0, 1]]
    pl = [ch1, ch2]
    if pl in l:
        if pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1]:
            ch1 = 1

    if ch1 == 0:
        ch1 = 'S'
        if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
            ch1 = 'A'
        if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
            ch1 = 'T'
        if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
            ch1 = 'E'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
            ch1 = 'M'
        if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
            ch1 = 'N'

    if ch1 == 2:
        ch1 = 'C' if distance(pts[12], pts[4]) > 42 else 'O'

    if ch1 == 3:
        ch1 = 'G' if (distance(pts[8], pts[12])) > 72 else 'H'

    if ch1 == 7:
        ch1 = 'Y' if distance(pts[8], pts[4]) > 42 else 'J'

    if ch1 == 4:
        ch1 = 'L'

    if ch1 == 6:
        ch1 = 'X'

    if ch1 == 5:
        if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
            ch1 = 'Z' if pts[8][1] < pts[5][1] else 'Q'
        else:
            ch1 = 'P'

    if ch1 == 1:
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'B'
        if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'D'
        if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'F'
        if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
            ch1 = 'I'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'W'
        if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] < pts[9][1]:
            ch1 = 'K'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'U'
        if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and (pts[4][1] > pts[9][1]):
            ch1 = 'V'
        if (pts[8][0] > pts[12][0]) and (
                pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
            ch1 = 'R'

    return ch1


def speak_in_browser(text):
    """Trigger the VISITOR's own browser to read the text aloud (works locally and once hosted,
    since speech plays on whichever device is viewing the page, not the server)."""
    safe_text = json.dumps(text)
    components.html(
        f"""
        <script>
            window.speechSynthesis.cancel();
            const msg = new SpeechSynthesisUtterance({safe_text});
            msg.rate = 1.0;
            window.speechSynthesis.speak(msg);
        </script>
        """,
        height=0,
        width=0,
    )


# ----------------------------------------------------------------------
# Video processor: runs the webcam + hand-tracking + model on each frame
# ----------------------------------------------------------------------
class SignProcessor(VideoProcessorBase):
    def __init__(self):
        self.hd = HandDetector(maxHands=1)
        self.hd2 = HandDetector(maxHands=1)
        self.lock = threading.Lock()
        self.current_symbol = " "
        self.skeleton_img = None
        self.frame_count = 0
        self.hand_count = 0
        self.last_error = None

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        with self.lock:
            self.frame_count += 1
        try:
            hands = self.hd.findHands(img, draw=False, flipType=True)
            img_copy = np.array(img)
            if hands and hands[0]:
                with self.lock:
                    self.hand_count += 1
                hand = hands[0][0]
                x, y, w, h = hand['bbox']
                crop = img_copy[y - OFFSET:y + h + OFFSET, x - OFFSET:x + w + OFFSET]
                white = cv2.imread(WHITE_IMG_PATH)
                if white is None:
                    with self.lock:
                        self.last_error = f"white.jpg failed to load (cv2.imread returned None)"
                elif crop.size == 0:
                    with self.lock:
                        self.last_error = "hand crop was empty (bbox near frame edge)"
                else:
                    handz = self.hd2.findHands(crop, draw=False, flipType=True)
                    if handz and handz[0]:
                        handmap = handz[0][0]
                        pts = handmap['lmList']
                        os_x = ((400 - w) // 2) - 15
                        os_y = ((400 - h) // 2) - 15
                        draw_skeleton(white, pts, os_x, os_y)
                        symbol = classify(pts, model, white.copy())
                        with self.lock:
                            self.current_symbol = symbol
                            self.skeleton_img = white
                            self.last_error = None
                    else:
                        with self.lock:
                            self.last_error = "hd2 (cropped re-detect) found no hand"
        except Exception as e:
            with self.lock:
                self.last_error = f"{type(e).__name__}: {e}"
            print("processing error:", e)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
if "sentence" not in st.session_state:
    st.session_state.sentence = ""

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("<div class='card-title'>Camera Feed</div>", unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="signspeak",
        video_processor_factory=SignProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
    )

with col2:
    st.markdown("<div class='card-title'>Hand Landmarks</div>", unsafe_allow_html=True)
    skeleton_placeholder = st.empty()

st.markdown("<hr/>", unsafe_allow_html=True)


@st.fragment(run_every=0.15)
def live_panel():
    current_symbol = " "
    skel = None
    if ctx.video_processor:
        with ctx.video_processor.lock:
            current_symbol = ctx.video_processor.current_symbol
            skel = ctx.video_processor.skeleton_img

    if skel is not None:
        skeleton_placeholder.image(cv2.cvtColor(skel, cv2.COLOR_BGR2RGB), use_container_width=True)

    # --- Diagnostics: temporary, helps pinpoint why detection may not be working ---
    if ctx.video_processor:
        with ctx.video_processor.lock:
            fc = ctx.video_processor.frame_count
            hc = ctx.video_processor.hand_count
            le = ctx.video_processor.last_error
        st.caption(f"🔍 Frames processed: {fc} · Hands detected: {hc} · Last error: {le or 'none'}")

    c1, c2 = st.columns([1, 3])
    with c1:
        st.markdown("<div class='card-title'>Character</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='char-card'>{current_symbol}</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card-title'>Sentence</div>", unsafe_allow_html=True)
        shown = st.session_state.sentence if st.session_state.sentence.strip() else "&nbsp;"
        st.markdown(f"<div class='sentence-card'>{shown}</div>", unsafe_allow_html=True)

    st.write("")
    b1, b2, b3, b4, b5 = st.columns(5)
    with b1:
        if st.button("➕ Add Letter", use_container_width=True):
            if current_symbol.strip():
                st.session_state.sentence += current_symbol
    with b2:
        if st.button("⎵ Space", use_container_width=True):
            st.session_state.sentence += " "
    with b3:
        if st.button("⌫ Backspace", use_container_width=True):
            st.session_state.sentence = st.session_state.sentence[:-1]
    with b4:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.sentence = ""
    with b5:
        if st.button("🔊 Speak", use_container_width=True):
            text = st.session_state.sentence.strip()
            if text:
                speak_in_browser(text)


live_panel()
