import gradio as gr


def test(name, checkbox):
    return f"{name=}, {checkbox=}"


demo = gr.Interface(
    fn=test, inputs=[gr.Text(), gr.Checkbox()], outputs=gr.Text())

demo.launch()
