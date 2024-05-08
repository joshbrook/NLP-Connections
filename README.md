# NLP-Connections

#### An NYT Connections style game, generated using word embeddings and played in the terminal

## Prerequisites
- Python 3.x installed on your system.
- Clone this repo to your PC
- Install required packages via pip:

```bash
pip install -r requirements.txt
```
- Execute the Python script.

```bash
python main.py
```

## Playing the Game

- Upon running the script, you will be prompted to guess the connections between words in each group.
- Enter one word at a time for each group, pressing enter after each entry.
- The game will inform you if your guess is correct or incorrect.
- You have four chances to guess incorrectly. After four incorrect guesses, the correct answers will be displayed.
- If you correctly guess all four connections, you'll be given the option to play again.

### Notes

- The game uses NLP techniques to generate word groups with common connections.
- It employs word vectors to find similar words and part-of-speech tagging to ensure connections within each group.
- Each game is randomly generated, ensuring a unique experience with every playthrough.
