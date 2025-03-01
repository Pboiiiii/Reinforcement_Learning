# Deep Q-Network VS Chrome Dinosaur Game

I implemented a Deep Q-Network (DQN) using PyTorch to play the Chrome dinosaur game.

I have worked around two already existing videos, one about implementing the algorithm to play flappy bird and another video coding the Dinosaur game with pygame.

### Process and "Improvements"

At first, I planned to create a custom Gymnasium environment to seamlessly integrate the existing DQN algorithm code. However, I encountered some issues while following various guides. As a result, I decided to implement the entire algorithm directly within the Dinosaur game’s main code. You’ll notice that the structure remains similar to how it would be if I were using a Gymnasium environment.



I also attempted to implement a render parameter inside the INIT() function, but it didn’t work as expected. If I remember correctly, using render=False would arise some issues with pygame. That’s why the code will always displays the Pygame window. On the bright side, this allows you to see the agent’s progress in real time.



Another improvement in the code is the ability to load a pre-trained model. When training is resumed, if a saved model of the agent already exists, the program will automatically load it and continue training from that point. To start training from scratch, simply delete the .pt file created from previous runs.



And that would be it regarding my additions to the code from [Johnny Code](https://www.youtube.com/@johnnycode/featured). I think that some improvements could be done to the state space in the algorithm. 



After 5 hours of training, the agent stated to consistenly jump above the cactus and ducking under the birds. Unfortunately I don't have any footage about the training and, by doing modifications to the code and doing some testing, I ended up deleting the .log file of the 5 hours run.



I'll edit the repository once I have footage of a run.

### Libraries

- [`pygame`](https://www.pygame.org/docs/): For handling the game's graphics and events.
- `os`: To manage file paths and system-related operations.
- `random`: For generating random elements in the game.
- [`numpy`](https://numpy.org/doc/): For numerical operations, such as handling state space data.
- [`yaml`](https://pyyaml.org/wiki/PyYAMLDocumentation): To read and write configuration settings.
- `itertools.count`: To create an infinite counter for tracking frames.
- [`torch`](https://pytorch.org/docs/stable/index.html) & [`torch.nn`](https://pytorch.org/docs/stable/nn.html): The PyTorch framework for deep learning and building neural networks.
- [`matplotlib.pyplot`](https://matplotlib.org/stable/api/pyplot_api.html) & [`matplotlib`](https://matplotlib.org/stable/contents.html): For visualizing training progress and results.
- [`datetime`](https://docs.python.org/3/library/datetime.html) & [`timedelta`](https://docs.python.org/3/library/datetime.html#datetime.timedelta): For managing timestamps in logging and evaluation.
- [`argparse`](https://docs.python.org/3/library/argparse.html): For parsing command-line arguments.

### Video References

- [Implement Deep Q-Learning with PyTorch and Train Flappy Bird! | DQN PyTorch Beginners Tutorial by Johnny Code](https://www.youtube.com/watch?v=arR7KzlYs4w\&list=PL58zEckBH8fCMIVzQCRSZVPUp3ZAVagWi)
- [Chrome Dinosaur in Pygame](https://www.youtube.com/playlist?list=PL30AETbxgR-fAbwiuU1vDl3owNUPUuVrz)

