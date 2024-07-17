import torch
import clip
from PIL import Image
import os
import numpy as np
import copy
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(clip.available_models())

with open("data\\train_image_paths.txt", "r") as file:
    train_paths = file.readlines()
train_paths = [path.strip().replace("/","\\") for path in train_paths] # 改行文字を削除とwindowsようにパスの/を\に変換
train_img_emb_path = "data\\train_img_emb.pt"

with open("data\\val_image_paths.txt", "r") as file:
    val_paths = file.readlines()
val_paths = [path.strip().replace("/","\\") for path in val_paths] # 改行文字を削除とwindowsようにパスの/を\に変換
val_img_emb_path = "data\\val_img_emb.pt"

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
    text_features /= text_features.norm(dim=-1, keepdim=True)

BATCH_SIZE = 16  # バッチサイズを適切な値に設定
train_img_emb = torch.empty(0, 512).to(device)  # 空のテンソルを初期化
if not os.path.exists(train_img_emb_path):
    for i in tqdm(range(0, len(train_paths), BATCH_SIZE)):
        with torch.no_grad():
            batch_paths = train_paths[i:min(i + BATCH_SIZE, len(train_paths))]
            images = [preprocess(Image.open("images\\" + path)) for path in batch_paths]
            images_input = torch.tensor(np.stack(images)).to(device)

            image_features = model.encode_image(images_input).to(device)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            train_img_emb = torch.cat((train_img_emb, image_features), dim=0)  # 結合

            del images_input  # メモリ節約のため削除
    torch.save(train_img_emb, train_img_emb_path)
train_img_emb = torch.load(train_img_emb_path)

val_img_emb = torch.empty(0, 512).to(device)  # 空のテンソルを初期化
if not os.path.exists(val_img_emb_path):
    for i in tqdm(range(0, len(val_paths), BATCH_SIZE)):
        with torch.no_grad():
            batch_paths = val_paths[i:min(i + BATCH_SIZE, len(val_paths))]
            images = [preprocess(Image.open("images\\" + path)) for path in batch_paths]
            images_input = torch.tensor(np.stack(images)).to(device)

            image_features = model.encode_image(images_input).to(device)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            val_img_emb = torch.cat((val_img_emb, image_features), dim=0)  # 結合

            del images_input  # メモリ節約のため削除
    torch.save(val_img_emb, val_img_emb_path)
val_img_emb = torch.load(val_img_emb_path)

print(train_img_emb.shape)
print(val_img_emb.shape)
image_features_list=[]
for train_path in tqdm(train_paths):
    if(len(image_features_list) > 256):
        break
    images=[]
    images.append(preprocess(Image.open("images\\"+train_path)))
    images_input = torch.tensor(np.stack(images)).to(device)
    with torch.no_grad():
        image_features = model.encode_image(images_input).to(device) # 画像の特徴量を取得(1,512)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features_list.append(image_features)
        #similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        #values, top_indices = similarity[0].topk(5)
    del images_input
    continue
    #print(image_text_name[top_indices[0]])
    correct = False
    for top_idx in top_indices:
        if image_text_name[top_idx] == train_path.split("\\")[0]:
            correct_num += 1
            correct = True
            break
    if not correct:
        incorrect_num += 1
        #print("Correct:{} Incorrect:{}".format(train_path.split('\\')[0], image_text_name[top_indices[0]]))
        #print(f"Correct:{correct_num} Incorrect:{incorrect_num}")