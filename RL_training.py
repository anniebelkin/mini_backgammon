from RL.train import pretrain, training, test_model, test_specific_model
from RL.game import sample_random
import torch

def main():
    print("Choose an option to run:")
    print("1. Pretrain")
    print("2. Training")
    print("3. testing")
    print("4. test specific model")
    choice = input("Enter the number of your choice: ").strip()

    if choice == "1":
        pretrain(
            learning_rate=0.001,
            batch_update_size=50,
            games=5000, 
            improvement_threshold=0.0005,
            patience=10
        )
    elif choice == "2":
        training(
            learning_rate=0.001,
            batch_update_size=50,
            epochs_per_round=100,
            max_rounds=1000,
            test_games=10,
            desired_win_rate=1.0
        )
    elif choice == "3":
        test_model()
    elif choice == "4":
        print("Enter the path of the model")
        path = input("Enter the path: ").strip()
        test_specific_model(path)
    else:
        print("Invalid choice. Please enter '1' or '2'.")

if __name__ == "__main__":
    main()