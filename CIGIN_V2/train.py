# from torch.nn.modules import flatten
from tqdm import tqdm
import torch
import numpy as np
import math
import json
import wandb

mse_loss = torch.nn.MSELoss()
mae_loss = torch.nn.L1Loss()
# rmse_loss_fn = torch.nn.MSELoss()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# output_dir = '/users4/ythou/Projects/Medical/CIGIN/runs/-'

flatten = lambda t: [item for sublist in t for item in sublist]


def get_metrics(model, data_loader):
    pred = []
    label = []
    for solute_graphs, solvent_graphs, solute_lens, solvent_lens, labels in data_loader:
        outputs, i_map = model([solute_graphs.to(device), solvent_graphs.to(device), torch.tensor(solute_lens).to(device), torch.tensor(solvent_lens).to(device)])
        pred += outputs.cpu().detach().numpy().tolist()
        label += labels
        # loss = loss(outputs, torch.tensor(labels).to(device).float())
        # valid_outputs += outputs.cpu().detach().numpy().tolist()
        # valid_loss.append(loss.cpu().detach().numpy())

    pred = flatten(pred)
    label = flatten(label)
    mse = mse_loss(torch.tensor(pred), torch.tensor(label))
    mae = mae_loss(torch.tensor(pred), torch.tensor(label))
    # loss = np.mean(np.array(valid_loss).flatten())
    # rmse_loss = math.sqrt(loss)
    return math.sqrt(mse), mse, mae, torch.tensor(pred).tolist()


def train(max_epochs, model, optimizer, scheduler, train_loader, valid_loader, test_loader, project_name, output_dir):
    if test_loader:
        test_best_score = {"RMSE": 10000, "MAE": 10000}
    val_best_score = {"RMSE": 10000, "MAE": 10000}
    for epoch in tqdm(range(max_epochs)):
        model.train()
        running_loss = []
        # tq_loader = tqdm(train_loader)
        for samples in train_loader:
            optimizer.zero_grad()
            outputs, interaction_map = model([samples[0].to(device), samples[1].to(device), torch.tensor(samples[2]).to(device), torch.tensor(samples[3]).to(device)])
            if interaction_map is not None:
                l1_norm = torch.norm(interaction_map, p=2) * 1e-4
                loss = mse_loss(outputs, torch.tensor(samples[4]).to(device).float())
                # loss = loss_fn(outputs, torch.tensor(samples[4]).to(device).float()) + l1_norm

                # loss = loss - l1_norm
            else:
                loss = mse_loss(outputs, torch.tensor(samples[4]).to(device).float())
            loss.backward()
            optimizer.step()

            # print(type(scheduler) != torch.optim.lr_scheduler.ReduceLROnPlateau, type(scheduler), torch.optim.lr_scheduler.ReduceLROnPlateau)

            if type(scheduler) != torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step()
                # print("debug!! success to debug")
            current_loss = loss.cpu().detach()
            running_loss.append(current_loss)
            # tq_loader.set_description(
            # "Epoch: {} Training loss： {}".format(epoch + 1, "%.4f" % np.mean(np.array(running_loss))))
            # tq_loader.set_description(
            #     "Epoch: " + str(epoch + 1) + "  Training loss: " + str(np.mean(np.array(running_loss))))
        model.eval()
        val_rmse_score, val_loss, val_mae_score, val_pred = get_metrics(model, valid_loader)
        if type(scheduler) is torch.optim.lr_scheduler.ReduceLROnPlateau:
            scheduler.step(val_loss)
        train_loss = np.mean(np.array(running_loss))
        print("Epoch:{} Train Loss:{:.4f} Val Loss:{:.4f} Val RMSE Score:{:.4f} Val MAE Score:{:.4f}".format(epoch + 1, train_loss, val_loss, val_rmse_score, val_mae_score))
        metrics = {project_name + "train_loss": train_loss, project_name + "val_rmse_loss": val_rmse_score, project_name + "val_mae_loss": val_mae_score}
        wandb.log(metrics)
        # print("Epoch: " + str(epoch + 1) + "  train_loss " + str(np.mean(np.array(running_loss))) + " Val_loss " + str(
        #     val_loss) + " RMSE_Val_loss " + str(val_rmse_loss) + " ")
        if val_rmse_score < val_best_score["RMSE"]:
            val_best_score["RMSE"] = val_rmse_score
            print(f'Get new RMSE best! valid {val_rmse_score}\n')
            torch.save(model.state_dict(), output_dir + "best_model_RMSE.pt")
            with open(output_dir + "valid_pred_RMSE.json", 'w') as writer:
                json.dump(val_pred, writer)
            if test_loader:
                test_rmse_score, test_loss, test_mae_score, test_pred = get_metrics(model, test_loader)
                test_best_score["RMSE"] = test_rmse_score
                with open(output_dir + "test_pred_RMSE.json", 'w') as writer:
                    json.dump(test_pred, writer)
        if val_mae_score < val_best_score["MAE"]:
            val_best_score["MAE"] = val_mae_score
            torch.save(model.state_dict(), output_dir + "best_model_MAE.pt")
            print(f'Get new MAE best! valid {val_mae_score}\n')
            with open(output_dir + "valid_pred_MAE.json", 'w') as writer:
                json.dump(val_pred, writer)
            if test_loader:
                test_rmse_score, test_loss, test_mae_score, test_pred = get_metrics(model, test_loader)
                test_best_score["MAE"] = test_mae_score
                with open(output_dir + "test_pred_MAE.json", 'w') as writer:
                    json.dump(test_pred, writer)
    with open(output_dir + "../best.txt", "a") as writer:
        writer.write("{};RMSE;{};{}\n".format(project_name, val_best_score["RMSE"], test_best_score["RMSE"] if test_loader else 0))
        writer.write("{};MAE;{};{}\n".format(project_name, val_best_score["MAE"], test_best_score["MAE"] if test_loader else 0))
    wandb.log({project_name + "best_val_rmse_loss": val_best_score["RMSE"], project_name + "best_val_mae_loss": val_best_score["MAE"]})
    return val_best_score["RMSE"], val_best_score["MAE"]