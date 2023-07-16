# pip install gradio_client

from gradio_client import Client

client = Client(
    "http://localhost:7860")
result = client.predict(
    "Howdy!",
    True,
    api_name="/predict"
)
print(result)
