from RL.train import pretrain

if __name__ == "__main__":
    pretrain(
        learning_rate=0.001,
        batch_update_size=50,
        games=5000, 
        improvement_threshold=0.0001,
        patience=10
    )