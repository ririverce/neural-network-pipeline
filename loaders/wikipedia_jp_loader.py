import os
import copy
import pickle

import tqdm



class WikipediaJPLoader:

    document_file = 'wiki_wakati.txt'
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        """ word """
        if not os.path.exists(os.path.join(self.root_dir, 'temp_words.pickle')):
            dictionary = {}
            with open(os.path.join(self.root_dir, self.document_file),
                      'r', encoding='utf-8') as f:
                dictionary = self.__count_words(f)
            dictionary = self.__remove_words(dictionary)
            words = set(dictionary.keys())
            with open(os.path.join(self.root_dir, 'temp_words.pickle'),
                      mode='wb') as f:
                pickle.dump(words, f)
        else:
            with open(os.path.join(self.root_dir, 'temp_words.pickle'),
                      mode='rb') as f:
                words = pickle.load(f)
        self.words = {}
        self.words['\n'] = 0
        for i, w in enumerate(words, 0):
            self.words[w] = i + 1
        """ document """
        document = []
        with open(os.path.join(self.root_dir, self.document_file),
                  'r', encoding='utf-8') as f:
            pbar = tqdm.tqdm(total=26708684)
            line = f.readline()
            while line:
                if line != '\n':
                    if not line[0] in ['<']:
                        document.append(line)
                line = f.readline()
                pbar.update(1)
            pbar.close()
        self.document = document
        
    def __count_words(self, f):
        dictionary = dict()
        pbar = tqdm.tqdm(total=26708684)
        line = f.readline()
        while line:
            if line != '\n':
                sentence = line.split(' ')
                if not sentence[0] in ['<', '</']:
                    for word in sentence:
                        if word != '\n':
                            if not word in dictionary.keys():
                                dictionary[word] = 1
                            else:
                                dictionary[word] += 1
            line = f.readline()
            pbar.update(1)
        pbar.close()
        return dictionary
    
    def __remove_words(self, dictionary, num_words=65535):
        symbol = ["・", "（", "）", "「", "」", "『", "』", "【", "】",
                  "《", "》", "〈", "〉", "｛", "｝", "(", ")", "{", "}",
                  "[", "]", ",", ".", "，", "．"]
        for s in symbol:
            dictionary[s] = -1
        dictionary['\n'] = -1
        distribution = list(dictionary.values())
        distribution.sort()
        distribution = distribution[::-1]
        tmp_dictionary = copy.deepcopy(dictionary)
        pbar = tqdm.tqdm(total=len(dictionary.keys()))
        for key in dictionary.keys():
            if dictionary[key] <= distribution[num_words]:
                tmp_dictionary.pop(key)
            pbar.update(1)
        pbar.close()
        dictionary = tmp_dictionary
        return dictionary

    def load(self):
        return self.document, self.words



def unit_test():
    loader = WikipediaJPLoader('../datasets/WikipediaJP')
    document, words = loader.load()
    print(document[:10])
    print(words)


if __name__ == '__main__':
    unit_test()