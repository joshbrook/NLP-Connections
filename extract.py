import gensim
import gensim.downloader as api
import numpy as np

from tqdm.notebook import tqdm

from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer

from collections import Counter


def save_word2vec_format(fname, vocab, vector_size):
    """Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

    Parameters
    ----------
    fname : str
        The file path used to save the vectors in.
    vocab : dict
        The vocabulary of words.
    vector_size : int
        The number of dimensions of word vectors.

    """

    total_vec = len(vocab)

    with gensim.utils.open(fname, 'wb') as fout:

        fout.write(gensim.utils.to_utf8("%s %s\n" % (total_vec, vector_size)))

        # store in sorted order: most frequent words at the top
        for word, row in tqdm(vocab.items()):
            row = row.astype(np.float32)
            fout.write(gensim.utils.to_utf8(word) + b" " + row.tobytes())



def get_words():
    lem = WordNetLemmatizer()
    brown_words = [word for word in brown.words() if word.isalpha() and word.islower()]
    wordlist = {lem.lemmatize(word) for (word, count) in Counter(brown_words).items() if count > 1 and count < 300 and len(word) > 2}
    with open('wordlist.txt', 'w') as f:
        for word in wordlist:
            f.write(word + '\n')

    vectors = {}
    wv = api.load('word2vec-google-news-300')
    for word in wordlist:
        try:
            vectors[word] = wv[word]
        except KeyError:
            pass

    save_word2vec_format('vectors.bin', vectors, 300)

get_words()