Chess DQN AI
This project implements a Deep Q-Network (DQN) agent to play chess using reinforcement learning. The AI learns by playing against itself, gradually improving through experience replay and neural network updates.

Features
Uses python-chess for chess board management and rules.
Deep neural network built with PyTorch (torch).
Experience replay buffer for stable training.
Epsilon-greedy exploration for action selection.
Trains via self-play and receives rewards for wins, losses, and draws.
Requirements
Install dependencies with:

pip install python-chess torch numpy flask
Usage
Web App
Start the Flask web app:

python app.py
Visit http://localhost:5000/ for a simple homepage.
POST to /train with JSON like {"episodes": 10} to trigger training.
Example using curl:

curl -X POST -H "Content-Type: application/json" -d '{"episodes": 10}' http://localhost:5000/train
Script
Run the training script directly:

python dqn_chess_app.py
You will see output per episode showing the total reward and the final board position (FEN string).

How it works
Board Encoding: Each chess board is converted to a numeric vector representing the pieces.
Action Space: The agent considers all legal moves at each step.
Learning: The DQN updates its neural network using sampled experiences from past games.
Reward Signal: +1 for a win, -1 for a loss, 0 for a draw, and a small penalty per move to encourage quick victories.
Customization
Change episodes in the script or API to train longer.
Adjust neural network architecture for more complex learning.
Integrate with a GUI or play against other chess engines for advanced evaluation.
License
MIT License

Author
kinoi254 otienoallan1920@gmail.com.
