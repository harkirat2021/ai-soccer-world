"""
All the code's here to start
"""

import numpy as np
import torch

import torch
from torch import nn, optim

from soccer_world import SoccerAgent, SoccerWorld, SoccerWorldGraphics

#%%

# Different objective functions for soccer world

# MSE between agent position and desired destination position
def agent_destination_mse(agent, destx, desty):
    return (destx - agent.pos[0]) ** 2 + (desty - agent.pos[1]) ** 2

#%%

# Initialize world for train
world = SoccerWorld()
world.red_team.append(SoccerAgent(-1, 0))
world.red_team[0].model.train()


train_losses = []

#%%

NUM_EPOCHS = 30
TRAIN_SIM_STEPS = 20

# Set optimizer for agent model
optimizer = optim.Adam(world.red_team[0].model.parameters(), lr=1e-3)
    
# Train model for specified epochs
for e in range(NUM_EPOCHS):
    # Reset loss and gradients
    train_loss = 0
    optimizer.zero_grad()
        
    # Run world forward some fixed number of steps
    world.reset()
    for i in range(TRAIN_SIM_STEPS):
        world.update()
        vu = world.red_team[0].move(world.get_observation(world.red_team[0]), train_mode=True)
        #print(vu)

    # Compute loss
    # Minimize the distance between the red agent and the ball
    loss = agent_destination_mse(world.red_team[0], world.ball.pos[0].item(), world.ball.pos[1].item())
    
    # Backprop loss and update agent model
    loss.backward()
    train_loss += loss.item()
    optimizer.step()
        
    # Update train losses
    train_losses.append(train_loss)
    print('Epoch {} Loss: {:.4f}\n'.format(e, train_loss))


#%%

# Reset and render the world
world.reset()
graphics = SoccerWorldGraphics(world)


graphics.update_paths()
    
# Step through soccer world a number of times
for i in range(TRAIN_SIM_STEPS):
    world.update()
    vu = world.red_team[0].move(world.get_observation(world.red_team[0]))
    graphics.update_paths()

graphics.draw_world()

