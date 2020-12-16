
TS = 50


# Initialize graphics
wn = turtle.Screen()
wn.bgcolor("green")
wn.title("Soccer World")

ball = turtle.Turtle()
ball.shape("circle")
ball.color("black")
ball.penup()
ball.speed(0)
ball.goto(world.ball.pos[0], world.ball.pos[0])

# Update loop
while True:
    # Update world
    world.update()
    
    # Update graphics
    wn.update()
    ball.goto(TS * world.ball.pos[0], TS * world.ball.pos[1])
wn.mainloop()
turtle.done()
turtle.byte()

