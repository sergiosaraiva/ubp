import sys
import json
from ubp import ubp

def main():
    if len(sys.argv) < 4:
        print("Usage: python main.py <predict | train> <CSV file> <model to save/load> [<config file>]")
        sys.exit(1)

    mode = sys.argv[1]
    csv_file_path = sys.argv[2]
    model_path = sys.argv[3]
    config_path = sys.argv[4] if len(sys.argv) > 4 else 'config.json'

    with open(config_path, 'r') as file:
        config = json.load(file)

    predictor = ubp(config)
    if mode == 'train':
        X, y = predictor.preprocess_data(csv_file_path, return_labels=True)
        predictor.build_model()
        predictor.train_model(X, y, epochs=config['epochs'], batch_size=config['batch_size'])
        predictor.save_model(model_path)
    elif mode == 'predict':
        predictor.load_model(model_path)
        X = predictor.preprocess_data(csv_file_path, return_labels=False)
        results = predictor.predict_next_steps(X)
        for i, (sample, result) in enumerate(zip(X, results)):
            print(f"Sample {i+1} next step predictions: {sample.flatten()}")
            for rank, (pred, prob) in enumerate(zip(result['predictions'], result['probabilities'])):
                print(f"  {pred} -> {prob*100:.2f}%")
    else:
        print(f"Invalid mode {mode}")
        print("Usage: python main.py <predict | train> <CSV file> <model to save/load> [<config file>]")
        sys.exit(1)

if __name__ == '__main__':
    main()
