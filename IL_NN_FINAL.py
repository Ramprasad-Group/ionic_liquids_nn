import numpy as np 
import pandas as pd
import seaborn as sns 
from tqdm.notebook import tqdm 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import QuantileTransformer 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score, log_loss, mean_absolute_percentage_error 
from sklearn.model_selection import GroupShuffleSplit 
import math 
import statistics
import random


#import pytorch_forecasting

df = pd.read_csv("C:[INSERT TARGET FILE LOCATION AND NAME].csv", encoding = "ISO-8859-1") 
df.head()

x = df.iloc[:,2:-1]
y = df.iloc[:,-1].values

def antilog(val): 
    fin = 10 ** val 
    return fin

rs = random.randint(1,10000)

rs_list = [rs]
pc_list = [0.0000000001]
rmse_test_list = [] 
rmse_train_list = [] 
r2_list = [] 
learning_rates = [0.001] 
train_len_list = []



for pc in pc_list:

    for rs in rs_list:
        
        neurons = 160 #160 was found to be the optimal number of neurons per layer. Alternate architectures could be used

        #####################TRAIN-TEST##############################
        
        groups = df.iloc[:,0]
        groups = np.array(groups)


        x = np.array(x)
        y = np.array(y)

        n_splits=10

        gss = GroupShuffleSplit(n_splits,test_size=0.1,random_state=rs)
        gss_index=gss.split(x, y, groups)

        x_trainval = []
        x_test = []
        Group_trainval = []
        y_trainval=[]
        y_test=[]


        for train_idx, test_idx in gss_index:

                x_trainval=x[train_idx]
                x_test=x[test_idx]
                y_trainval=y[train_idx]
                y_test=y[test_idx]   
                Group_trainval=groups[train_idx]


        #####################TRAIN-VAL##############################

        n_splits=2

        gss = GroupShuffleSplit(n_splits,test_size=(0.1/0.9),random_state=rs)
        gss_index2=gss.split(x_trainval, y_trainval, Group_trainval)

        print ('GSS INDEX')


        x_train = []
        x_val = []
        Group_train = []
        y_train =[]
        y_val=[]

        for train_index, test_index in gss_index2:

                x_train=x_trainval[train_index]
                x_val=x_trainval[test_index]
                y_train=y_trainval[train_index]
                y_val=y_trainval[test_index]   
                Group_train=groups[train_index]



        #####################TRAIN-EDIT############################## 
        ###############(ONLY USE FOR LEARNING CURVES)################

        n_splits=2

        gss = GroupShuffleSplit(n_splits,test_size=(pc),random_state=rs)
        gss_index3=gss.split(x_train, y_train, Group_train)

        x_train_fin = []
        x_excess = []
        Group_train_fin = []
        y_train_fin =[]
        y_excess=[]

        for train_ind, test_ind in gss_index3:

                x_train_fin=x_train[train_ind]
                x_excess=x_train[test_ind]
                y_train_fin=y_train[train_ind]
                y_excess=y_train[test_ind]   
                Group_train_fin=groups[train_ind]        

        print ('test len:', len(x_test))
        print ('validation len:', len(x_val))
        print ('train len:', len(x_train))
        print ('total len:', len(y))
        print ('excess len', len(y_excess))
        print ('Done with train/test setup')

        #REMOVE '#' if running for learning curves
        
        #x_train = x_train_fin
        #y_train = y_train_fin

        
        dataa = df.iloc[:,1:-2]
        count = 1

        y_t_ll = []
        count2 = 1
        for i in range(len(test_idx)-1):
            idx1 = test_idx[i]
            idx2 = test_idx[i+1]
            a = dataa.iloc[idx1]
            b = dataa.iloc[idx2]
            truth = a.equals(b)
            if truth == False:
                y_t_ll.append(1)

        y_t_d = len(y_test)
        y_t_l = len(y_t_ll)
            

#DATA SCALING BETWEEN 0 AND 1
        scaler = MinMaxScaler()


        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        x_test = scaler.transform(x_test)
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_val, y_val = np.array(x_val), np.array(y_val)
        x_test, y_test = np.array(x_test), np.array(y_test)

        y_train, y_test, y_val = y_train.astype(float), y_test.astype(float), y_val.astype(float)

#INITALIZE DATASET

        class RegressionDataset(Dataset):

            def __init__(self, X_data, y_data):
                self.X_data = X_data
                self.y_data = y_data

            def __getitem__(self, index):
                return self.X_data[index], self.y_data[index]

            def __len__ (self):
                return len(self.X_data)

        train_dataset = RegressionDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
        val_dataset = RegressionDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float())
        test_dataset = RegressionDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())

#SET MODEL PARAMETERS
        EPOCHS = 200 #450,500
        BATCH_SIZE = 32
        LEARNING_RATE = 0.002
        WEIGHT_DECAY = 0.001
        NUM_FEATURES = len(x[1])

#DATALOADERS
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)

#DEFINE ARCHITECTURE
        class MultipleRegression(nn.Module):
            def __init__(self, num_features):
                super(MultipleRegression, self).__init__()

                self.layer_1 = nn.Linear(num_features, neurons)
                self.layer_2 = nn.Linear(neurons, neurons)
                self.layer_3 = nn.Linear(neurons, neurons)
                self.layer_out = nn.Linear(neurons, 1)

                self.relu = nn.ReLU()
                

            def forward(self, inputs):
                x = self.relu(self.layer_1(inputs))
                x = self.relu(self.layer_2(x))
                x = self.relu(self.layer_3(x))
                x = self.layer_out(x)
                return (x)
            def predict(self, test_inputs):
                x = self.relu(self.layer_1(test_inputs))
                x = self.relu(self.layer_2(x))
                x = self.relu(self.layer_3(x))
                x = self.layer_out(x)
                return (x)

#GPU?
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

#model/optimizer/loss function
        model = MultipleRegression(NUM_FEATURES)
        model.to(device)
        print(model)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

#loss/epoch dict
        loss_stats = {
            'train': [],
            "val": []
        }


#BEGIN TRAINING

        print("Begin training.")
        y_val_preds = []
        val_list = []
        train_loss_list = []

        for e in tqdm(range(1, EPOCHS+1)):

# TRAINING
            train_epoch_loss = 0

            model.train()
            for x_train_batch, y_train_batch in train_loader:
                    x_train_batch, y_train_batch = x_train_batch.to(device), y_train_batch.to(device)
                    optimizer.zero_grad()

                    y_train_pred = model(x_train_batch)

                    train_loss = criterion(y_train_pred, y_train_batch.unsqueeze(1))
                    
                    
                    train_loss.backward()
                    optimizer.step()

                    train_epoch_loss += train_loss.item()



# VALIDATION    
            with torch.no_grad():

                    val_epoch_loss = 0

                    model.eval()
                    for x_val_batch, y_val_batch in val_loader:
                        x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)

                        y_val_pred = model(x_val_batch)
                        y_val_preds.append(y_val_pred)

                        val_loss = criterion(y_val_pred, y_val_batch.unsqueeze(1))

                        val_epoch_loss += val_loss.item()

            len_train = len(train_loader)
            tl = float(train_epoch_loss)
            val_t = (tl/len_train) 
            loss_stats['train'].append(val_t) #(int(train_epoch_loss/len(train_loader)))

            len_val = len(val_loader)
            vl = float(val_epoch_loss)
            val_l = (vl/len_val)
            loss_stats['val'].append(val_l) #(int(val_epoch_loss/len(val_loader)))   


            print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')
            val_list.append(val_epoch_loss/len(val_loader))

    
    
        #GETTING TRAIN LOSS IN RMSE 

        train_loss_rmse = min(loss_stats['train'])
        print (train_loss_rmse)

        train_loss_rmse = math.sqrt(train_loss_rmse)
        rmse_train_list.append(train_loss_rmse)
    
        train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})

        min_e = min(val_list)
        min_in = val_list.index(min_e) + 1 
        print (f'Min: {min_in}')

        

        #PREDICTIONS
        
        y_pred_list = []
        with torch.no_grad():
            model.eval()
            for x_batch, _ in test_loader:
                x_batch = x_batch.to(device)
                y_test_pred = model(x_batch) #THIS IS WHERE WE INPUT THE TEST VALUES
                y_pred_list.append(y_test_pred.cpu().numpy())

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

        
        #CALCULATES THE R^2 VALUE AND RMSE

        r_square_test = r2_score(y_test, y_pred_list)
        rmse_test = mean_squared_error(y_test, y_pred_list, squared=False)
        print("R^2 Train/Test Parity Plot:",r_square_test)
        print("Root Mean Squared Error Train/Test:",rmse_test)
        
        rmse_test_list.append(rmse_test)
        r2_list.append(r_square_test)

        
        #CHANGES RESULTS VALUES OUT OF LOG SCALE  
        new_ytest = []
        new_ypred = []
        
        for item in y_test:
            testi = 10**item
            new_ytest.append(testi)
            
        for item in y_pred_list:
            predi = 10**item
            new_ypred.append(predi)
        
        def truncate(n, decimals=0):
            multiplier = 10 ** decimals
            return int(n * multiplier) / multiplier

        r2 = truncate(r_square_test, 2)
        rmse = 10**rmse_test
        rmse = truncate(rmse, 2)
        
        #PLOTS TRAIN-VAL LOSS

        plt.figure()
        plt.plot(train_val_loss_df["epochs"][:EPOCHS], train_val_loss_df["value"][:EPOCHS], label='Train')
        plt.plot(train_val_loss_df["epochs"][EPOCHS:], train_val_loss_df["value"][EPOCHS:], label = 'Val')
        plt.ylim(0,1.2)
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss")
        plt.title("Train/Val Loss")
        #h = xline(min_in, 'r--', 'LineWidth', 4);
        plt.axvline(x = min_in, linestyle = '--', color = 'green', label = 'Best Epoch')  
        plt.legend()

        #GENERATES THE PARITY PLOT FOR THE EXPERIMENTAL AND PREDICTED VALUES

        plt.rcParams.update({'font.size': 15})
        pltfont = {'fontname':'Arial'}

        plt.figure(figsize=(6, 6))
        scat1 = sns.scatterplot(x = new_ytest,y = new_ypred, label=f'Test ($R^2$={r2}, RMSE={rmse})') #{y_t_l} ILs,
        plt.xlabel("Experimental Ionic Conductivity (S/m)", fontsize=18, **pltfont)
        plt.ylabel("Predicted Ionic Conductivity (S/m)", fontsize=18, **pltfont)
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(10**-7,10**1.5)
        plt.ylim(10**-7,10**1.5)
        plt.plot([10**-7,10**1.5], [10**-7,10**1.5], '--k', label='_nolegend_')
        plt.legend(fontsize=14)


print ("RMSE TEST VALUES:", rmse_test_list) 
print ("RMSE TRAIN VALUES:", rmse_train_list) 
print ("R2 VALUES:", r2_list) 