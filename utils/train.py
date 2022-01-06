from tqdm import tqdm
import torch
import numpy as np
from utils.test import Tester
from transformers import AdamW, get_linear_schedule_with_warmup as get_lwp
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from utils.draw import plot_loss


class Trainer:

    def __init__(self):
        pass

    def train(self, max_epochs, model, optimizer, scheduler, project_name, output_dir, tester, loss_fn, train_loader, valid_loader, test_loader=None, show_loss=True, scheduler_type='lwp', better_score=lambda x, y: x > y):
        if test_loader:
            test_best_score = {"RMSE": 10000, "MAE": 10000}
        val_best_score = {"RMSE": 10000, "MAE": 10000}
        train_loss = []
        for epoch in tqdm(range(max_epochs)):
            model.train()
            running_loss = []
            # tq_loader = tqdm(train_loader)
            # for batch in train_loader:
            train_loader_len = len(train_loader)
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                inputs, label = batch[:-1], batch[-1]
                # pred, moe_loss = model(inputs)
                pred, moe_loss = model(inputs, i < 3 or train_loader_len - i <= 3)
                # print('Debug pred {} {}, label {} {}'.format(pred.dtype, pred.shape, label.dtype, label.shape))
                label = label.long() if type(loss_fn) == torch.nn.CrossEntropyLoss else label
                model_loss = loss_fn(pred, label)
                loss = model_loss + moe_loss
                loss.backward()
                if i < 3 or train_loader_len - i <= 3:
                    print("solu moe grad=", model.Solu_MoE.w_gate.grad.sum(0).tolist())
                    print("solv moe grad=", model.Solv_MoE.w_gate.grad.sum(0).tolist())
                    print("mix moe grad=", model.Mix_MoE.w_gate.grad.sum(0).tolist())

                optimizer.step()
                running_loss.append(loss.cpu().detach())
                if scheduler_type == 'lwp':
                    scheduler.step()

                # if show_loss:
                #     tq_loader.set_description("Epoch:{} Loss:{:.4f}".format(epoch + 1, np.mean(np.array(running_loss))))
                # else:
                #     tq_loader.set_description("Epoch:{}".format(epoch + 1))
            model.eval()
            val_rmse_score, val_loss, val_mae_score, val_pred = tester.test(model, valid_loader)
            if scheduler_type == 'plateau':
                scheduler.step(val_rmse_score)
            cur_train_loss = np.mean(np.array(running_loss))
            train_loss.append(cur_train_loss)
            print("Epoch:{} Train Loss:{:.4f} Val Loss:{:.4f} Val RMSE Score:{:.4f} Val MAE Score:{:.4f} lr:{:.6f}".format(epoch + 1, cur_train_loss, val_loss, val_rmse_score, val_mae_score, optimizer.param_groups[0]['lr']))
            if better_score(val_rmse_score, val_best_score["RMSE"]):
                val_best_score["RMSE"] = val_rmse_score
                torch.save(model.state_dict(), output_dir + "best_model_RMSE.pt")
                print(f'Get new RMSE best! valid {val_rmse_score}\n')
                with open(output_dir + "valid_pred_RMSE.json", 'w') as writer:
                    json.dump(val_pred, writer)
                if test_loader:
                    test_rmse_score, test_loss, test_mae_score, test_pred = tester.test(model, test_loader)
                    test_best_score["RMSE"] = test_rmse_score
                    with open(output_dir + "test_pred_RMSE.json", 'w') as writer:
                        json.dump(test_pred, writer)
            if better_score(val_mae_score, val_best_score["MAE"]):
                val_best_score["MAE"] = val_mae_score
                torch.save(model.state_dict(), output_dir + "best_model_MAE.pt")
                print(f'Get new MAE best! valid {val_mae_score}\n')
                with open(output_dir + "valid_pred_MAE.json", 'w') as writer:
                    json.dump(val_pred, writer)
                if test_loader:
                    test_rmse_score, test_loss, test_mae_score, test_pred = tester.test(model, test_loader)
                    test_best_score["MAE"] = test_mae_score
                    with open(output_dir + "test_pred_MAE.json", 'w') as writer:
                        json.dump(test_pred, writer)
        with open(output_dir + "train_loss.json", 'w') as writer:
            writer.write(str(train_loss))
        plot_loss(train_loss, output_dir)
        with open(output_dir + "../best.txt", "a") as writer:
            writer.write("{};RMSE;{};{}\n".format(project_name, val_best_score["RMSE"], test_best_score["RMSE"] if test_loader else 0))
            writer.write("{};MAE;{};{}\n".format(project_name, val_best_score["MAE"], test_best_score["MAE"] if test_loader else 0))


def make_optimizer_scheduler(opt, model, train_steps):
    if opt.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=float(opt.lr))
    elif opt.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=float(opt.lr), weight_decay=opt.weight_decay, correct_bias=True)
    else:
        raise ValueError("Wrong choice for optimizer")

    if opt.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=opt.patience, mode='min', verbose=True)
    elif opt.scheduler == "lwp":
        warmup_steps = int(opt.warmup_proportion * train_steps)
        print("debug steps warmup {} train {}".format(warmup_steps, train_steps))
        scheduler = get_lwp(optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps)
    else:
        raise ValueError("Wrong choice for scheduler")

    return optimizer, scheduler
