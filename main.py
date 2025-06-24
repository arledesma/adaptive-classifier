import torch
import numpy as np
import random
from adaptive_classifier import AdaptiveClassifier
from src.attack import AttackUtility
from src.attack.data import TEST_DATA


def main():

    # Initialize classifier
    classifier = AdaptiveClassifier("distilbert/distilbert-base-cased")

    # Initial training data with attack tactics
    utility: AttackUtility = AttackUtility(data_path=AttackUtility.get_attck_cache_path(), debug=True)
    attack_data = utility.get_attack_data()

    texts: list[str] = []
    labels: list[str] = []
    # texts: list[str] = [tac['description'] for tac in attack_data]
    # labels: list[str] = [f"{tac['tactic_id']}-{tac['name'].replace(' ', '_')}" for tac in attack_data]

    for tac in attack_data:
        classification_label = tac['name'].replace(' ', '')  # f"{tac['tactic_id']}-{tac['name'].replace(' ', '')}"
        texts.append(tac['description'])
        labels.append(classification_label)
        for tech in tac['techniques']:
            texts.append(tech['description'])
            labels.append(classification_label)

    print(f"Number of initial examples: {len(texts)}")
    print(f"Number of initial labels: {len(set(labels))}")
    assert len(texts) == len(labels), "Texts and labels must have the same length"

    # Add examples
    print("Adding initial examples...")
    classifier.add_examples(texts, labels)

    # Test predictions
    test_texts: list[str] = [data['displayName'] for data in TEST_DATA]

    print("\nTesting predictions:")
    classifier.model.eval()

    with torch.no_grad():
        for text in test_texts:
            predictions = classifier.predict(text)
            print(f"\nText: {text}")
            print("Predictions:")
            for label, score in predictions:
                print(f"{label}: {score:.4f}")

    # Save the classifier
    print("\nSaving classifier...")
    classifier.save("./demo_classifier")

    # Load the classifier
    print("\nLoading classifier...")
    loaded_classifier = AdaptiveClassifier.load("./demo_classifier")

    # Add new technical class with more examples
    print("\nAdding new technical class...")
    technical_texts = [
        "Error code 404 appeared",
        "System crashed after update",
        "Cannot connect to database",
        "Memory allocation failed",
        "Null pointer exception detected",
        "API endpoint not responding",
        "Stack overflow in main thread"
    ]
    technical_labels = ["technical"] * len(technical_texts)

    for data in TEST_DATA:
        for tactic in data.get('tactics', []):
            technical_texts.append(data['displayName'])
            technical_labels.append(tactic.replace(' ', ''))

    loaded_classifier.add_examples(technical_texts, technical_labels)

    # Test new predictions
    print("\nTesting technical classification:")
    technical_test = "API giving null pointer exception"

    loaded_classifier.model.eval()

    with torch.no_grad():
        for text in technical_texts:
            predictions = classifier.predict(text)
            print(f"\nText: {text}")
            print("Predictions:")
            for label, score in predictions:
                print(f"{label}: {score:.4f}")


if __name__ == "__main__":
    main()
