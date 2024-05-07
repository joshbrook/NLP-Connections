import random
import gensim.downloader as api

from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

from collections import Counter

from prettytable import PrettyTable
from Levenshtein import distance as lev
from loaders import SpinningLoader


def get_words():
    lem = WordNetLemmatizer()
    brown_words = [word for word in brown.words() if word.isalpha() and word.islower()]
    wordlist = {lem.lemmatize(word) for (word, count) in Counter(brown_words).items() if count > 10 and count < 300 and len(word) > 2}
    with open('wordlist.txt', 'w') as f:
        for word in wordlist:
            f.write(word + '\n')

# get_words()


def make_groups():

    loader = SpinningLoader("Loading W2V Model...", colour="blue", complete_text="W2V Loaded")
    loader.start()
    wv = api.load('word2vec-google-news-300')
    loader.stop()
    print("")

    with open('wordlist.txt', 'r') as f:
        wordlist = f.read().splitlines()

        loader = SpinningLoader("Generating Connections...", colour="blue", complete_text="Connections Generated")
        loader.start()

        g = 0
        groups = []

        while g < 4:
            choices = [random.choice(wordlist).strip()]
            wvs = wv.most_similar(choices[0], topn=50)
            
            for word, _ in wvs:

                word = word.lower().strip()

                if (lev(choices[0], word) > 3 and 
                        word in wordlist and
                            word not in choices and
                                word not in choices[0] and
                                    pos_tag([word])[0][1] == pos_tag([choices[0]])[0][1]):

                    choices.append(word)
                    
                if len(choices) == 4:
                    g += 1
                    groups.append(choices)
                    break

            if len(choices) < 4:
                continue

            """
            print(f"Group {g+1}:")
            print("Options:")
            for j, word in enumerate(choices):
                print(f"{j+1}. {word}")
            print("")
            """

        loader.stop()
        print("")

        return groups

        

def generate_table(groups):
    words = [word for group in groups for word in group]
    random.shuffle(words)

    table = PrettyTable()
    table.header = False
    table.padding_width = 5
    for i in range(4):
        table.add_row([words[i], words[i+4], words[i+8], words[i+12]], divider=True)

    return table
        


def respond(groups):

    groups = [sorted(group) for group in groups]

    correct = 0
    incorrect = 0

    while correct < 4:
        print(f"Group {correct+1}:")
        print("Enter one word at a time, pressing enter after each\n")
        common = []
        for j in range(4):
            response = input()
            common.append(response)
        if sorted(common) in groups:
            correct += 1
            print("\nCorrect!\n")
        else:
            incorrect += 1
            print("\nIncorrect. Try again.\n")
        if incorrect == 4:
            print("You have made 4 incorrect guesses. The correct answers were:\n")
            for group in groups:
                print(group + "\n")
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

    print("\nWelcome to an NLP-based Connections Game!\n")
    print("The goals is to make 4 groups of 4 words each, where each group has a common connection\n")

    groups = make_groups()

    table = generate_table(groups)
    print(table)
    print("")

    respond(groups)
    

play()
        

