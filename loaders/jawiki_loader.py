import os
import pickle

import tqdm



class WikipediaJPLoader:

    document_file = 'wiki_wakati.txt'
    dictionary_file = 'wikipedia_jp_65536_words.pickle'
    
    def __init__(self, root_dir):
        self.root_dir = root_dir
        document = []
        with open(os.path.join(self.root_dir, self.document_file),
                  'r', encoding='utf-8') as f:
            pbar = tqdm.tqdm(total=26708684)
            line = f.readline()
            while line:
                if line != "\n":
                    sentence = line.split(" ")
                    if not sentence[0] in ["<", "</"]:
                        document.append(sentence)
                line = f.readline()
                pbar.update(1)
        print(len(document))
        

                    

    def load(self):
        return 0



def unit_test():
    loader = WikipediaJPLoader("./")
    



if __name__ == "__main__":
    unit_test()