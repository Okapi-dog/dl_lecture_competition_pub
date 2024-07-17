import timm
from pprint import pprint
import os
import torch
from torchinfo import summary

model_names = timm.list_models(pretrained=True)
pprint(model_names)
model = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=2)
model.to("cuda:0")
#print(model)
summary(model, (16, 3, 224, 224))
o = model(torch.randn(16, 3, 224, 224).to("cuda:0"))
print(f'Original shape: {o.shape}')
o = model.forward_features(torch.randn(16, 3, 224, 224).to("cuda:0"))
print(f'Unpooled shape: {o.shape}')
image_text_name = os.listdir("images")
image_text_name.remove(".gitkeep")
image_text_name.remove(".gitignore")
corecct_num=0
incorrect_num=0
for image_title in image_text_name:
    images=[]
    img_folder = "images\\" + image_title
    for img in os.listdir(img_folder):
        images.append(preprocess(Image.open(img_folder + "\\" + img)))
    images_input = torch.tensor(np.stack(images)).to(device)
    with torch.no_grad():
        image_features = model.encode_image(images_input)
        text_features = model.encode_text(text_tokens)
        
        logits_per_image, logits_per_text = model(images_input, text_tokens)
    del images_input
    probs2 = logits_per_image.softmax(dim=-1)
    # 類似度が高い順にソートし、上位5つのインデックスを取得
    top_probs, top_indices = probs2[1].topk(5)
    if image_title == image_text_name[top_indices[0]]:
        corecct_num+=1
    else:
        incorrect_num+=1
        print(f"Correct:{image_title} Incorrect:{image_text_name[top_indices[0]]}")
    print(f"Correct:{corecct_num} Incorrect:{incorrect_num}")
