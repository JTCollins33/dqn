from dqn import DQN
from env import get_env
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

NUM_EPOCHS=1000

def inside_square(loc):
    return loc[0]>=0 and loc[0]<32 and loc[1]>=0 and loc[1]<32

def save_plot(values, type, num_epochs):
    plt.clf()
    plt.plot(values)
    plt.title(str(type)+" After "+str(num_epochs)+" epochs")
    plt.xlabel("Epoch Number")
    plt.ylabel(type)
    plt.savefig("./results/"+str(type)+".png")

def get_expected_moves(env, loc):
    expected_moves = []
    if(env[0,])

if __name__ == '__main__':
    model = DQN()
    lossF = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    steps_arr = []
    env_sums = []
    loss_sums = []

    for ep in range(NUM_EPOCHS):
        print("Epoch "+str(ep)+"/"+str(NUM_EPOCHS), end='\r')
        # set up environment and location
        env = get_env()
        loc = torch.tensor([[3,4]], dtype=torch.float32)
        steps = 0
        loss_sum = 0.0

        while(torch.sum(env).item()>0 and inside_square(loc[0]) and steps<1000):
            move_probs = model.forward(env, loc)

            expected_move_probs = get_expected_moves(env, loc)

            move_index = torch.argmax(move_probs).item()

            if(move_index==0):
                loc[0,0] -= 1
            elif(move_index==1):
                loc[0,1] += 1
            elif(move_index==2):
                loc[0,0] += 1
            else:
                loc[0,1] -= 1


            # if(not inside_square(loc[0])):
            #     reward = -1
            # elif(torch.sum(env).item()==0):
            #     reward=5
            # else:
            #     reward=env[0,0,int(loc[0,0]), int(loc[0,1])]

            if(inside_square(loc[0])):
                env[0,0,int(loc[0,0]), int(loc[0,1])] = 0

            loss = lossF(move_probs, expec)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            steps+=1
        
        steps_arr.append(steps)
        env_sums.append(torch.sum(env).item())
        loss_sums.append(loss_sum)

        if(ep%10==0):
            save_plot(steps_arr, "Number_of_Steps_Completed", ep)
            save_plot(env_sums, "Environment_Sums", ep)
            save_plot(loss_sums, "Loss_Sums", ep)
