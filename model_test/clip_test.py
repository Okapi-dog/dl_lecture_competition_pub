import torch
import clip
from PIL import Image
import os
import numpy as np
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
print(clip.available_models())
model, preprocess = clip.load("ViT-B/16", device=device)
image_text_name = os.listdir("images")
image_text_name.remove(".gitkeep")
image_text_name.remove(".gitignore")
replace_list = {"button1":"cloth button","button2":"push button","button3":"",
                "bat1":"bat(animal)","bat2":"baseball bat",
                "baton1":"conductor's baton","baton2":"baton(Nightstick)","baton3":"baton used by baton twirler","baton4":"baton used in relay race",
                "calf1":"calf(cow)","calf2":"calf(human leg)",
                "camera1":"SLR camera","camera2":"television camera",
                "chest1":"chest of human body","chest2":"chest of drawers",
                "chicken1":"chicken(food)","chicken2":"chicken(bird)",
                "hook1":"hook(latch)","hook2":"hook used for hanging",
                "juicer1":"manual juicer","juicer2":"automatic juicer",
                "mold1":"mold(used for shaping)","mold2":"mold(fungus)",
                "mouse1":"mouse(animal)","mouse2":"computer mouse",
                "pepper1":"pepper(spice)","pepper2":"pepper(vegetable)",
                "punch1":"punch(beverage)","punch2":"punch(tool)",
                "pipe1":"pipe(smoking)","pipe2":"pipe(water pipe)",
                "screen1":"screen(projection screen)","screen2":"screen(mesh)",
                "shell1":"shell(bullet)","shell2":"shell(sea creature)","shell3":"shell(tree nut)",
                "stove1":"stove(cooking stove)","stove2":"stove(heating stove)",
                "straw1":"straw(plant)","straw2":"straw(drinking straw)",
                "tank1":"tank(armored vehicle)","tank2":"tank(storage container)",
                "walker1":"walker(elderly aid)","walker2":"walker(child's toy)",
                "_":" "}
image_text_replaced = copy.deepcopy(image_text_name)  # ディープコピーで新しいリストを作成
for i in range(len(image_text_replaced)):
    for key in replace_list.keys():
        image_text_replaced[i] = image_text_replaced[i].replace(key,replace_list[key])
    image_text_replaced[i] = "This is a photo of "+ image_text_replaced[i]
    
text_tokens = clip.tokenize(image_text_replaced).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens).to(device)
print(image_text_name)
correct_num=0
incorrect_num=0
for image_title in image_text_name:
    images=[]
    img_folder = "images\\" + image_title
    for img in os.listdir(img_folder):
        images.append(preprocess(Image.open(img_folder + "\\" + img)))
    images_input = torch.tensor(np.stack(images)).to(device)
    with torch.no_grad():
        image_features = model.encode_image(images_input).to(device)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, top_indices = similarity[0].topk(5)
        
        #logits_per_image, logits_per_text = model(images_input, text_tokens)
    del images_input
    #probs2 = logits_per_image.softmax(dim=-1)
    # 類似度が高い順にソートし、上位5つのインデックスを取得
    #top_probs, top_indices = probs2[1].topk(5)
    correct = False
    for top_idx in top_indices:
        if image_text_name[top_idx] == image_title:
            correct_num += 1
            correct = True
            break
    if not correct:
        incorrect_num += 1
        print(f"Correct:{image_title} Incorrect:{image_text_name[top_indices[0]]}")

        print(f"Correct:{correct_num} Incorrect:{incorrect_num}")
    