import time
import gc
import collections
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from kaggle_environments import make
from data_processing import get_dataset, GeeseDataset,extract_dataset
from model import GeeseNet
from lib import create_folders_if_not_exist, get_path_list, create_submission_file


criterion = {
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss()
}
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")





def create_dataloader(X_train, p_train, v_train, X_val, p_val, v_val, batch_size=2048, num_workers=1):
    train_dataset = GeeseDataset(X=X_train, p=p_train, v=v_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    val_dataset = GeeseDataset(X=X_val, p=p_val, v=v_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = val_loader

    return loaders

def train_epoch(loader, model, criterion, optimizer, scaler, device, scheduler):
    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (X, p, v) in bar:
        X = X.to(device)
        p = p.to(device).type(torch.cuda)
        v = v.to(device).reshape((-1,1)).type(torch.cuda)

        optimizer.zero_grad()
        with autocast():
            p_pred, v_pred = model(X)
            p_loss = criterion["ce"](p_pred, p)
            v_loss = criterion["mse"](v_pred, v)
            loss = (0.3 * p_loss + 0.7 * v_loss)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        loss_np = loss.detach().cpu().numpy()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-20:]) / min(len(train_loss), 20)
        bar.set_description(f"loss: {loss_np:.5f}, smth: {smooth_loss:.5f}")
    return train_loss

def val_epoch(loader, model, criterion, device):
    model.eval()
    val_loss = []
    val_p_loss = []
    val_v_loss = []
    outputs = []
    p1, p2, p3 = [], [], []
    masks = []
    acc = 0.0
    length = 0.0

    # initialize time counter
    infer_time = 0

    with torch.no_grad():
        for (X, p, v) in tqdm(loader):
            X = X.to(device)
            p = p.to(device).type(torch.cuda.LongTensor)
            v = v.to(device).reshape((-1, 1)).type(torch.cuda.FloatTensor)

            # start timer
            start_time = time.time()

            p_pred, v_pred = model(X)

            # get elapsed time
            infer_time += time.time() - start_time

            p_loss = criterion["ce"](p_pred, p)
            v_loss = criterion["mse"](v_pred, v)
            loss = (0.3 * p_loss + 0.7 * v_loss) # value is more important than policy

            pred = p_pred.argmax(1).detach()
            outputs.append(pred)

            acc += (p == pred).sum().cpu().numpy()
            length += len(p)

            val_loss.append(loss.detach().cpu().numpy())
            val_p_loss.append(p_loss.detach().cpu().numpy())
            val_v_loss.append(v_loss.detach().cpu().numpy())

        val_loss = np.mean(val_loss)
        val_p_loss = np.mean(val_p_loss)
        val_v_loss = np.mean(val_v_loss)
        acc = acc / length * 100

        # average inference time
        avg_infer_time = infer_time / len(loader)

    return val_loss, val_p_loss, val_v_loss, acc, avg_infer_time

def train(args):


    layers = args.layers
    filters = args.filters
    batch_size = args.batch_size
    data_folder = args.data_folder
    val_size = args.val_size
    n_epochs = args.n_epochs
    chunk_size = args.chunk_size
    chunk_num = args.chunk_num
    
    print(f'layers :{layers}, filters: {filters}')
    
    # download training data
    zip_path, _ = extract_dataset(data_folder)

    # create necessary folders
    folders_to_create = ['./models', './videos', './agents']
    create_folders_if_not_exist(folders_to_create)

    # prepare validation set
    path_list = get_path_list(data_folder)
    path_list_val = path_list[:val_size]
    X_val, p_val, v_val = get_dataset(path_list_val, 0, len(path_list_val))
    path_list_train = path_list[val_size:]

    # Initialize the table with column names
    columns = [
        "visualized episode",
        "epoch",
        "layers",
        "filters",
        "batch_size",
        "win_rate",
        "val_loss", 
        "val_policy_loss",
        "val_value_loss", 
        "acc", 
        "avg_infer_time", 
    ]


    scaler = GradScaler()

    model = GeeseNet(layers=layers, filters=filters)
    model = model.to(device)

    for n in tqdm(range(chunk_num)):
        print('Convert json episodes to np.array...')
        X_train, p_train, v_train = get_dataset(path_list_train, n, chunk_size)

        loaders = create_dataloader(X_train, p_train, v_train, X_val, p_val, v_val, batch_size=batch_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr = 2e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=10, optimizer=optimizer)

        print('Start training...')
        for epoch in range(1, n_epochs+1):
            print(time.ctime(), 'Epoch:', epoch)
            scheduler.step(epoch-1)
            train_loss = train_epoch(loaders["train"], model, criterion, optimizer, scaler, device, scheduler)
            val_loss, val_p_loss, val_v_loss, acc, avg_infer_time = val_epoch(loaders["valid"], model, criterion, device)

            name = f'layers{layers}_filters{filters}_chunk{n}_epoch{epoch}'

            # save model
            model_path = f'./models/{name}.pth'
            torch.save(model.state_dict(), model_path)

            # create agent file
            create_submission_file(model_path,  f'./agents/{name}.py', layers=layers, filters=filters)

            # evaluate agent
            win_rate = evaluate_agent(f'./agents/{name}.py', n_matches=50)



            content = time.ctime() + ' ' + f'Training Number {n}, Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {np.mean(val_loss):.5f}, val policy loss: {np.mean(val_p_loss):.5f}, val value loss: {np.mean(val_v_loss):.5f}, acc: {(acc):.5f}'
            print(content)

        gc.collect()
        del X_train, p_train, v_train, loaders


    del model
    torch.cuda.empty_cache()

def evaluate_agent(agent_file, n_matches=50):
    env = make("hungry_geese", debug=False)
    win_rates = []
    for _ in range(n_matches):
        result = env.run([agent_file, "greedy", "greedy", "greedy"])
        ranks = [i + 1 for i, agent in sorted(enumerate(result[-1]), key=lambda x: x[1]["reward"], reverse=True)]
        win_rate = [(4 - rank) / 3 for rank in ranks]
        win_rates.append(win_rate[0])

    return sum(win_rates) / len(win_rates)

if __name__ == '__main__':
    layers = 12,
    filters = 8,
    batch_size = 2048,
    data_folder = './geese',
    val_size = 100,
    n_epochs = 3,
    chunk_size = 500,
    chunk_num = 1,
    project = 'reinforcelearning',
    entity = None,
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--layers', type=int, default=layers, help='number of layers')
    argparser.add_argument('--filters', type=int, default=filters, help='number of filters')
    argparser.add_argument('--batch_size', type=int, default=batch_size, help='batch size')
    argparser.add_argument('--data_folder', type=str, default=data_folder,
                           help='path of training data folder')
    argparser.add_argument('--val_size', type=int, default=val_size, help='size of validation set')
    argparser.add_argument('--n_epochs', type=int, default=n_epochs,
                           help='number of epochs for each chunk')
    argparser.add_argument('--chunk_size', type=int, default=chunk_size,
                           help='number of samples in each chunk')
    argparser.add_argument('--chunk_num', type=int, default=chunk_num,
                           help='number of chunks to be used for training')
    args = argparser.parse_args()


    train(args)
