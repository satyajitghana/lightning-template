# pip install opencv-python-headless

import gradio as gr
import cv2
import numpy as np
import time


def sepia(input_img):
    print(f"{type(input_img)=}, {input_img.shape=}")

    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()

    blurred_img = cv2.blur(sepia_img, (5, 5))

    flipped = cv2.flip(blurred_img, 0)

    return flipped


demo = gr.Interface(fn=sepia, inputs=gr.Image(
    shape=(200, 200), type="numpy"), outputs="image")
demo.queue(concurrency_count=3)
demo.launch(share=True)
