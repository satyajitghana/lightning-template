
# load dataset
dataset = load_dataset("jxie/flickr8k")

# load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
# text_features = model.get_text_features(**inputs)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# inputs = processor(images=image, return_tensors="pt")
# image_features = model.get_image_features(**inputs)

# # normalized features
# image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
# text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

# # cosine similarity as logits
# logit_scale = self.logit_scale.exp()
# logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
# logits_per_image = logits_per_text.t()

# loss = None
# if return_loss:
#     loss = clip_loss(logits_per_text)

# if not return_dict:
#     output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
#     return ((loss,) + output) if loss is not None else output

# return CLIPOutput(
#     loss=loss,
#     logits_per_image=logits_per_image,
#     logits_per_text=logits_per_text,
#     text_embeds=text_embeds,
#     image_embeds=image_embeds,
#     text_model_output=text_outputs,
#     vision_model_output=vision_outputs,
# )

# takes 10 minutes to compute all the image embeddings
# for batch in tqdm(dataset['train'], total=len(dataset['train'])):
#     with torch.no_grad():
#         inputs = processor(images=batch['image'], return_tensors="pt")
#         image_features = model.get_image_features(**inputs)


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)

# inputs = processor(text=["a photo of a cat", "a photo of a dog"],
#                    images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# # this is the image-text similarity score
# logits_per_image = outputs.logits_per_image
# # we can take the softmax to get the label probabilities
# probs = logits_per_image.softmax(dim=1)
