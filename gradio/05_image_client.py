from gradio_client import Client

client = Client("https://d9596cb629454bb339.gradio.live/")
result = client.predict(
    # str (filepath or URL to image) in 'input_img' Image component
    "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png",
    api_name="/predict"
)
print(result)
