# ai-soccer-world

run main.py to train the agent and get a plot its path over time

Soccer world has two teams (red and blue) and a soccer ball. All of the computations are done in torch tensors so they can be differentiated. At each time step update, each agent makes a move (changes velocity given state obsevation) and all objects in the game are updated.

You can see a plot of how the objects moved over time using SoccerWorld Graphics.