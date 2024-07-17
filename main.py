import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

import clip
from PIL import Image
import copy

from src.datasets import ThingsMEGDataset
from src.models2 import BasicConvClassifier
from src.utils import set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"Logdir: {logdir}")
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")
        
    
    
    #CLIPのモデルを読み込む
    model, preprocess = clip.load("ViT-B/16", device=args.device)
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
        
    text_tokens = clip.tokenize(image_text_replaced).to(args.device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens).to(args.device)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsMEGDataset("train", args.data_dir)
    # 条件に一致するインデックスを取得(このへんで被験者ごとのモデルを作成したときの精度を確認するために、被験者0のデータのみを取得している)=>結果大体4%ぐらい
    #subject_idxs = train_set.subject_idxs == 0
    #filtered_indices = [i for i, match in enum  enumerate(subject_idxs) if match]

    # torch.utils.data.Subsetを使用して新しいDatasetを作成
    #train_set_filtered = torch.utils.data.Subset(train_set, filtered_indices)

    # 修正されたtrain_setを使用してDataLoaderを作成
    #train_loader = torch.utils.data.DataLoader(train_set_filtered, shuffle=True, **loader_args)
    
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir)
    test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, **loader_args)

    # ------------------
    #       Model
    # ------------------
    model = BasicConvClassifier(
        train_set.num_classes, train_set.seq_len, train_set.num_channels,hid_dim=args.hid_dim,p_drop=args.p_drop
    ).to(args.device)

    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)
    y_pred=""
    img_emb=""
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        train_iter = iter(train_loader)  # DataLoaderからイテレータを作成
        
        model.train()
        for X, y, subject_id, img_emb in tqdm(train_loader, desc="Train"):
             # X.shape: torch.Size([271,281])なのは、(channel, time)の順番であることを示している.280は-100msから1300msまでのデータで、サンプリングレートは200Hzである
            X, subject_id, y, img_emb = X.to(args.device), subject_id.to(args.device), y.to(args.device), img_emb.to(args.device)
            #print("img_emb")
            #print(img_emb)
            y_pred = model(X, subject_id)
            #print("y_pred")
            #print(y_pred)
            similarity = y_pred @ img_emb.T  # コサイン類似度を計算
            log_softmax_similarity = F.log_softmax(similarity, dim=1)
            loss = -log_softmax_similarity.diag().mean()  # 負の対数尤度損失を計算
            #criterion = nn.MSELoss()  # MSELossのインスタンスを作成
            #loss = criterion(y_pred/std, img_emb/std)  # forwardメソッドに引数を渡す
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred = y_pred.half()
            similarity = (100.0 * y_pred @ text_features.T).softmax(dim=-1)
            _, top_indices_batch = torch.topk(similarity, k=10, dim=1)
            correct = 0
            not_correct = 0
            for i in range(len(top_indices_batch)):
                is_correct = False
                for j in range(len(top_indices_batch[i])):
                    if y[i] == top_indices_batch[i][j]:
                        is_correct = True
                        correct += 1
                        break
                if not is_correct:
                    not_correct += 1
            
            acc = correct / (correct + not_correct)
            train_acc.append(acc)

        model.eval()
        for X, y, subject_id, img_emb in tqdm(val_loader, desc="Validation"):
            X, subject_id, y = X.to(args.device), subject_id.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X, subject_id)
            similarity = y_pred @ img_emb.T  # コサイン類似度を計算
            log_softmax_similarity = F.log_softmax(similarity, dim=1)
            loss = -log_softmax_similarity.diag().mean()  # 負の対数尤度損失を計算
            #criterion = nn.HuberLoss()  # MSELossのインスタンスを作成
            #std = torch.std(img_emb)
            #loss = criterion(y_pred/std, img_emb/std)  # forwardメソッドに引数を渡す
            val_loss.append(loss.item())
            
            y_pred = y_pred.half()
            similarity = (100.0 * y_pred @ text_features.T).softmax(dim=-1)
            _, top_indices_batch = torch.topk(similarity, k=10, dim=1)
            correct = 0
            not_correct = 0
            for i in range(len(top_indices_batch)):
                is_correct = False
                for j in range(len(top_indices_batch[i])):
                    if y[i] == top_indices_batch[i][j]:
                        is_correct = True
                        correct += 1
                        break
                if not is_correct:
                    not_correct += 1
            
            acc = correct / (correct + not_correct)
            val_acc.append(acc)

            #val_loss.append(F.cross_entropy(y_pred, img_emb).item())
            #val_acc.append(accuracy(y_pred, img_emb).item())

        #print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | val loss: {np.mean(val_loss):.3f}")
        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        print("img_emb")
        print(img_emb)
        print("y_pred")
        print(y_pred)
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "val_loss": np.mean(val_loss)})
            #wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        """
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
        """
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
