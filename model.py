import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Conv_QNet(nn.Module):
    def __init__(self, input_shape, input_logic_size, output_size):
        """
        input_shape: tuple of (Channels, Height, Width), e.g., (4, 32, 24)
        output_size: int, number of actions (e.g., 3)
        """
        super().__init__()

        self.model_folder_path = './model'

        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU()
        )

        # Run dummy input to find conv output tensor size
        dummy_input = torch.zeros(1, *input_shape) # tensore with size (3, 32, 24)
        flattened_size = self._get_conv_output(dummy_input)

        self.cnn_compress = nn.Sequential(
            nn.Linear(flattened_size, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(256 + input_logic_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

        # Smart weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Normal - excellent for Conv layers using ReLU
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                # Xavier Normal - classic and stable for Linear layers
                nn.init.xavier_normal_(m.weight)

    def _get_conv_output(self, x):
        """Helper func to find conv output tensor size"""
        x = self.conv_layers(x)
        return int(torch.prod(torch.tensor(x.size())))

    def forward(self, x_img, x_logic):
        x = self.conv_layers(x_img)
        
        # Flatten and keep the batches
        x = x.view(x.size(0), -1) 

        x = self.cnn_compress(x)
        
        # Combining conv output with x_logic to one tensor
        x = torch.cat((x, x_logic), dim=1)

        x = self.linear_layers(x)
        return x
        
    def save(self, file_name='model.pth'):
        if not os.path.exists(self.model_folder_path):
            os.makedirs(self.model_folder_path)

        file_name = os.path.join(self.model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        file_path = os.path.join(self.model_folder_path, file_name)
        
        if os.path.exists(file_path):
            self.load_state_dict(torch.load(file_path))
            print(f"Model successfully loaded from {file_path}")
        else:
            print("No saved model found, starting fresh.")


class QTrainer:
    def __init__(self, model, target_model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.target_model = target_model

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_func = nn.SmoothL1Loss()

    def train_step(self, state_img, state_logic, action, reward, next_state_img, next_state_logic, done):
        """
        The parameters can be one list/number each parameter or each paramter can be a tuple of lists/values
        """

        self.model.train()

        # single value/list parameters -> 1 dim tensors, tuple paramters -> 2 dim tensors
        state_img = torch.tensor(np.array(state_img), dtype=torch.float).to(device)
        state_logic = torch.tensor(np.array(state_logic), dtype=torch.float).to(device)
        next_state_img = torch.tensor(np.array(next_state_img), dtype=torch.float).to(device)
        next_state_logic = torch.tensor(np.array(next_state_logic), dtype=torch.float).to(device)
        action = torch.tensor(np.array(action), dtype=torch.float).to(device)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(device)

        if len(state_img.shape) == 3:
            state_img = torch.unsqueeze(state_img, 0)
            state_logic = torch.unsqueeze(state_logic, 0)
            next_state_img = torch.unsqueeze(next_state_img, 0)
            next_state_logic = torch.unsqueeze(next_state_logic, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: pedicted Q values with current state
        pred = self.model(state_img, state_logic)

        # Predicted the reward on next state of every state
        with torch.no_grad():
            next_preds = self.target_model(next_state_img, next_state_logic)

        target = pred.clone().detach()
        for idx in range(len(done)):
            # If done, onlt set Q_new to R (current reward)
            Q_new = reward[idx]

            # If not done set Q_new = r + y * max(next_prediced Q value)
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(next_preds[idx])

            target[idx][torch.argmax(action[idx]).item()] = Q_new


        self.optimizer.zero_grad()
        loss = self.loss_func(target, pred)
        loss.backward()
        self.optimizer.step()

        return loss.item()