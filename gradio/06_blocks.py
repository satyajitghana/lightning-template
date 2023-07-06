# pip install transformers

import gradio as gr
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


def predict(inp: str) -> str:
    encoded_input = tokenizer(inp, return_tensors='pt')

    model_out = model.generate(**encoded_input, max_new_tokens=128)

    # note: this is not streaming output, once we get the entire output, we then decode it

    decoded = tokenizer.decode(model_out[0].tolist())

    return decoded


with gr.Blocks(title="Example with Textbox", theme=gr.themes.Glass()) as demo:

    gr.Markdown("Start typing below and then click **Run** to see the output.")

    with gr.Row():

        input_txt = gr.Textbox(placeholder="Input Text", label="Model Input")

        input_txt.change(lambda inp: print(
            f"got change {inp}"), inputs=input_txt)

        output_txt = gr.Textbox()

    btn = gr.Button("Run")
    btn.click(fn=predict, inputs=input_txt, outputs=output_txt)


demo.launch()
