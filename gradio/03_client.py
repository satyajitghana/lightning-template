# pip install gradio_client

from gradio_client import Client

client = Client(
    "http://localhost:7860")
result = client.predict(
    "Howdy!",  # str  in 'name' Textbox component
    True,  # bool  in 'checkbox' Checkbox component
    # int | float (numeric value between 0 and 100) in 'value' Slider component
    0,
    api_name="/predict"
)
print(result)
