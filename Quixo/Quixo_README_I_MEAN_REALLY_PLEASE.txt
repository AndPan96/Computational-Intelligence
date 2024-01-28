I have built two players: MyPlayer and MyPlayer_dqn.

Consider MyPlayer for your benchmark.

Please import everything from the main.py file that you can find in this folder.

You can call my pretrained models, respectively, with:

player1 = MyPlayer(trained=True, model_path=["my_model_90_90.pkl"]) (use just this for testing please °u°)

player1 = MyPlayer_dqn(trained=True, model_path=["my_model0_75.pth"]) (use this 4 fun *u*)

You also have to add to the same folder files "my_model_90_90.pkl" and "my_model0_75.pth"

P.s. model_path is a list because I wanted to use an ensemble of models to reduce variance 
but, given the performance of the 1st model, I didn't find it interesting anymore ~(ù.ù)~
(don't blame on me, but pay attention not to forget squared brackets) 