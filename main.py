import random

from gensim.models import KeyedVectors
from nltk.tag import pos_tag
from prettytable import PrettyTable
from Levenshtein import distance as lev


def make_groups():
    """Generate 4 groups of 4 words each, with a common connection between words in each group"""

    wv = KeyedVectors.load_word2vec_format('vectors.bin', binary=True, unicode_errors='ignore')

    with open('wordlist.txt', 'r') as f:
        wordlist = f.read().splitlines()

        g = 0
        groups = []

        while g < 4:
            choices = [random.choice(wordlist).strip()]
            wvs = wv.most_similar(choices[0], topn=30, restrict_vocab=16000)
            
            for word, _ in wvs:

                word = word.lower().strip()

                if (lev(choices[0], word) > 4 and 
                        word in wordlist and
                            word not in choices and
                                word not in choices[0] and
                                    pos_tag([word])[0][1] == pos_tag([choices[0]])[0][1]):

                    choices.append(word)
                    
                if len(choices) == 4:
                    g += 1
                    groups.append(choices)
                    break
                else:
                    continue

            if len(choices) < 4:
                continue

        return groups



def colour(item, color):
    """Add chosen colour to given text in terminal output"""
    colour_code = {1: '\u001b[36;1m', 2: '\u001b[33;1m',
                   3: '\u001b[35;1m', 4: '\u001b[31;1m',
                   "g": '\u001b[32m', "r": '\u001b[31m'}
    
    return f'{colour_code[color]}{item}\u001b[0m'



def generate_table(groups, guessed=[]):
    """Generate a table of words in groups, with guessed groups coloured in terminal output"""

    for i, g in enumerate(groups):
        if g in guessed:
            for j, word in enumerate(g):
                g[j] = colour(word, i+1)
    
    words = [word for group in groups for word in group]
    random.shuffle(words)

    table = PrettyTable()
    table.header = False
    table.padding_width = 5

    for i in range(4):
        row = []
        for j in range(len(groups)):
            word_index = i + 4 * j
            row.append(words[word_index])
        table.add_row(row, divider=True)

    return table
        


def respond(groups):
    """Prompt user to guess the common connection between words in each group"""

    groups = [sorted(group) for group in groups]

    correct = 0
    incorrect = 0
    guessed = []

    while correct < 4:
        print(f"Group {correct+1}:")
        print("Enter one word at a time, pressing enter after each\n")
        common = []
        for j in range(4):
            response = input()
            common.append(response)

        if sorted(common) in groups:
            correct += 1
            print("\n" + colour("Correct!", "g") + "\n")
            guessed.append(sorted(common))
            print(generate_table(groups, guessed=guessed))
            print("")
        else:
            incorrect += 1
            print("\n" + colour("Incorrect. ", "r") + f"You have {4-incorrect} incorrect guesses remaining\n")
            print(generate_table(groups))
            print("")

        if incorrect == 4:
            print("You have made 4 incorrect guesses. The correct answers were:\n")
            print(generate_table(groups, guessed=groups))
            print("")
            break

        if correct == 4:
            print("That's all 4 connections, well done!\n")
            restart = input("Would you like to play again? (y/n)\n")
            if restart == 'y':
                play()
            else:
                break
            break

    
def play():
    """Play the Connections Game"""

    print("\nWelcome to an NLP-based Connections Game!\n")
    print("The goals is to make 4 groups of 4 words each, where each group has a common connection\n")

    groups = make_groups()

    table = generate_table(groups)
    print(table)
    print("")

    respond(groups)
    

play()
        

