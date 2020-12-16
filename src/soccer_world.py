import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from PIL import ImageDraw
import torch.nn as nn
import torch.nn.functional as F
from agent_models import SAM0

# Most units moved in 1 time step in this world
MAX_TIMESTEP_MOVE = 0.4
# Width (vertical) of the world
WORLD_WIDTH = 6
# Length (horizontal) of the world
WORLD_LENGTH = 12
# Distance from the center to the goal
GOAL_DIST = 6


"""
Object in the soccer world with basic properties like position, radius, friction, and velocity.
"""  
class SoccerEntity():
    def __init__(self, x, y, friction, radius):
        self.pos = torch.tensor([x, y], dtype=torch.float)
        self.vel = torch.tensor([0, 0], dtype=torch.float)
        
        self.friction = friction
        self.radius = radius
    
    # Update entity 1 time step forward
    def update(self):
        # Keep velocity between set max timestep move
        self.vel = MAX_TIMESTEP_MOVE * F.tanh(self.vel)
        
        # Update position
        self.pos += self.vel
        
        # Update velocity with friction
        self.vel *= self.friction
        
    # Compute distance between this and other entity
    def dist(self, e):
        return torch.norm(self.pos - e.pos)

    # Set the velocity of this object
    def set_vel(self, vx, vy):
        self.vel[0] = vx
        self.vel[1] = vy

"""
The soccer ball
"""
class SoccerBall(SoccerEntity):
    def __init__(self, x, y):
        super(SoccerBall, self).__init__(x=x, y=y, friction=0.95, radius=0.1)

"""
The soccer agent/player
"""
class SoccerAgent(SoccerEntity):
    def __init__(self, x, y, friction=0.7, radius=1):
        super(SoccerAgent, self).__init__(x=x, y=y, friction=0.95, radius=0.1)
        self.model = SAM0(input_dim=6)
        
    # Update velocity based on the agents observation of the world state
    def move(self, observation, train_mode=False):
        # v_update is at most, the max move for 1 time step
        vel_update = MAX_TIMESTEP_MOVE * self.model(observation)
        if not train_mode:
            vel_update = vel_update.detach()
        
        # Update agent velocity
        self.vel += vel_update
        
        # Return update values if needed for debugging
        return vel_update

"""
The soccer world which controls the state of the game and all objects in it
"""
class SoccerWorld():
    def __init__(self, length=WORLD_LENGTH, width=WORLD_WIDTH, goal_dist=GOAL_DIST):
        self.goal_dist = goal_dist
        self.length = length
        self.width = width
        self.ball = SoccerBall(0, 0)
        self.red_team = []
        self.blue_team = []
    
    # Get the observation of the world state from a particular agents perspective
    def get_observation(self, agent):
        ball_diff = self.ball.pos - agent.pos
        goal_post_1_diff = torch.tensor([self.goal_dist, self.width/2], dtype=torch.float) - agent.pos
        goal_post_2_diff = torch.tensor([self.goal_dist, -self.width/2], dtype=torch.float) - agent.pos
        return torch.cat((ball_diff, goal_post_1_diff, goal_post_2_diff), dim=0)
    
    # Update two objects that bounced off each other
    def bounce(self, e1, e2):
        # Save each velocity value
        vel1 = torch.clone(e1.vel)
        vel2 = torch.clone(e2.vel)
        
        # Switch velocity values
        e1.vel = vel2
        e2.vel = vel1
        
        e1.update()
        e2.update()
    
    # Reset all objects on the board for a new game
    # TODO - randomize agent positions
    def reset(self):
        # Restart ball in the center
        self.ball.vel = torch.zeros(2)
        self.ball.pos = torch.zeros(2)
        
        # Restart all red team agents to random spot on the right side
        for a in self.red_team:
            a.vel = torch.tensor([0, 0], dtype=torch.float)
            a.pos = torch.tensor([-3, 0], dtype=torch.float)
        
        # Restart all blue team agents to random spot on the left side
        for a in self.blue_team:
            a.vel = torch.tensor([0, 0], dtype=torch.float)
            a.pos = torch.tensor([3, 0], dtype=torch.float)

    # Update all objects in the world - will assume the necessary agent moves are made before this call
    def update(self):
        # Update all agents
        for agent in self.red_team:
            agent.update()
            
        for agent in self.blue_team:
            agent.update()
        
        # Update ball
        self.ball.update()
    
        all_agents = self.red_team + self.blue_team
        
        # For each agent, check if it is in collision with any other agent or the ball
        for i in range(0, len(all_agents)):
            for j in range(i, len(all_agents)):
                
                # Add collision if there is an overlap these are different agents
                agent_overlap = all_agents[i].radius + all_agents[j].radius - all_agents[i].dist(all_agents[j])
                if i != j and agent_overlap > 0:
                    self.bounce(all_agents[i], all_agents[j])
                
            # Add collision if there is an overlap with the ball
            ball_overlap = all_agents[i].radius + self.ball.radius - all_agents[i].dist(self.ball)
            if ball_overlap > 0:
                self.bounce(all_agents[i], self.ball)
        
        # Check if agents collides with boundary
        for agent in all_agents:
            if agent.pos[1].item() < -self.width / 2:
                agent.pos[1] = -self.width / 2
                agent.vel[1] *= -1
            elif agent.pos[1].item() > self.width / 2:
                agent.pos[1] = self.width / 2
                agent.vel[1] *= -1
                
            if agent.pos[0].item() < -self.length / 2:
                agent.pos[0] = -self.length / 2
                agent.vel[0] *= -1
            elif agent.pos[0].item() > self.length / 2:
                agent.pos[0] = self.length / 2
                agent.vel[0] *= -1
        
        # Check if ball collides with boundary
        if self.ball.pos[1].item() < -self.width / 2:
            self.ball.pos[1] = -self.width / 2
            self.ball.vel[1] *= -1
        elif self.ball.pos[1].item() > self.width / 2:
            self.ball.pos[1] = self.width / 2
            self.ball.vel[1] *= -1
            
        if self.ball.pos[0].item() < -self.length / 2:
            self.ball.pos[0] = -self.length / 2
            self.ball.vel[0] *= -1
        elif self.ball.pos[0].item() > self.length / 2:
            self.ball.pos[0] = self.length / 2
            self.ball.vel[0] *= -1

"""
Render the soccer world with different tools
"""
class SoccerWorldGraphics():
    def __init__(self, world):
        self.world = world
        
        # Initialize lists to keep track of agent and ball positions
        self.ball_path = []
        self.red_agents_path = []
        self.blue_agents_path = []
        
        for i in range(len(self.world.red_team)):
            self.red_agents_path.append([])
            
        for i in range(len(self.world.blue_team)):
            self.blue_agents_path.append([])

    # Update agent and ball paths
    def update_paths(self):
        self.ball_path.append(self.world.ball.pos.detach().numpy().tolist().copy())
        
        for i in range(len(self.world.red_team)):
            self.red_agents_path[i].append(self.world.red_team[i].pos.detach().numpy().tolist().copy())
        
        for i in range(len(self.world.blue_team)):
            self.blue_agents_path[i].append(self.world.blue_team[i].pos.detach().numpy().tolist().copy())
    
    # Draw image of how objects moved over time
    def draw_world(self):
        im = Image.new('RGBA', (100*self.world.length, 100*self.world.width), (255,255,255,255))
        draw = ImageDraw.Draw(im)
        th = 4
        
        # Draw goals
        draw.line(((100*-self.world.goal_dist, 100*-self.world.width / 2), (-self.world.goal_dist, self.world.width / 2)), fill=(0, 0, 0, 255), width=1)
        draw.line(((100*self.world.goal_dist, 100*-self.world.width / 2), (self.world.goal_dist, self.world.width / 2)), fill=(0, 0, 0, 255), width=1)
        
        
        # Draw red team paths
        for i in range(len(self.world.red_team)):
            for j in range(1, len(self.red_agents_path[i])):
                p1 = (100*(self.red_agents_path[i][j-1][0] + self.world.length / 2), 100*(self.red_agents_path[i][j-1][1] + self.world.width / 2))
                p2 = (100*(self.red_agents_path[i][j][0] + self.world.length / 2), 100*(self.red_agents_path[i][j][1] + self.world.width / 2))
                draw.line((p1, p2), fill=(255, 0, 0, 255), width=1)
                draw.line(((p2[0]-th, p2[1]-th), (p2[0]+th, p2[1]+th)), fill=(255, 0, 0, 255), width=3*th)
        
        # Draw blue team paths
        for i in range(len(self.world.blue_team)):
            for j in range(1, len(self.blue_agents_path[i])):
                p1 = (100*(self.blue_agents_path[i][j-1][0] + self.world.length / 2), 100*(self.blue_agents_path[i][j-1][1] + self.world.width / 2))
                p2 = (100*(self.blue_agents_path[i][j][0] + self.world.length / 2), 100*(self.blue_agents_path[i][j][1] + self.world.width / 2))
                draw.line((p1, p2), fill=(255, 0, 0, 255), width=1)
                draw.line(((p2[0]-th, p2[1]-th), (p2[0]+th, p2[1]+th)), fill=(0, 0, 255, 255), width=3*th)
        
        # Draw ball path
        for i in range(1, len(self.ball_path)):
            p1 = (100*(self.ball_path[i-1][0] + self.world.length / 2), 100*(self.ball_path[i-1][1] + self.world.width / 2))
            p2 = (100*(self.ball_path[i][0] + self.world.length / 2), 100*(self.ball_path[i][1] + self.world.width / 2))
            draw.line((p1, p2), fill=(0, 0, 0, 255), width=1)
            draw.line(((p2[0]-th, p2[1]-th), (p2[0]+th, p2[1]+th)), fill=(0, 0, 0, 255), width=3*th)
            
        plt.imshow(np.asarray(im), origin='lower')
        plt.show()