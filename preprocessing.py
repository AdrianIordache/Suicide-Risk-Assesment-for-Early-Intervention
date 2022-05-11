from utils import *

class Preprocessor:
    def __init__(self, data: pd.DataFrame):
        self.dataset    = data
        self.lemmatizer = WordNetLemmatizer()
        
        self.social_processor = TextPreProcessor(
            # terms that will be normalized
            normalize = ['url', 'email', 'percent', 'money', 'phone', 'user', 'time', 'url', 'date', 'number'],
            # terms that will be annotated
            annotate  = {"hashtag", "allcaps", "elongated", "repeated",'emphasis', 'censored'},
            fix_html  = True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used 
            # for word segmentation 
            segmenter = "twitter", 

            # corpus from which the word statistics are going to be used 
            # for spell correction
            corrector = "twitter", 

            unpack_hashtags     = True,  # perform word segmentation on hashtags
            unpack_contractions = True,  # Unpack contractions (can't -> can not)
            spell_correct_elong = False,  # spell correction for elongated words

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer = SocialTokenizer(lowercase = True).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts = [emoticons]
        )


    def text_preprocessing(self, forward: List[str] = [], column_name = "prepocessed_text") -> pd.DataFrame:
        # lower case
        self.dataset[column_name] = self.dataset['text'].apply(lambda x: x.lower())

        if "remove_url" in forward:
            self.dataset[column_name] = self.dataset[column_name].apply(lambda x: re.sub(r'http\S+', '', x))

        # remove numbers
        if "remove_numbers" in forward:
            self.dataset[column_name] = self.dataset[column_name].apply(lambda x: "".join([char for char in x if not char.isdigit()]))

        # remove stop words
        if "stop_words" in forward:
            self.dataset[column_name] = self.dataset[column_name].apply(lambda x: " ".join([word for word in word_tokenize(x) if not word in STOP]))

        # remove punctuation
        if "remove_punctuation" in forward:
            self.dataset[column_name] = self.dataset[column_name].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

        # word lematization
        if "lemmatization" in forward:
            self.dataset[column_name] = self.dataset[column_name].apply(lambda x: 
                " ".join([self.lemmatizer.lemmatize(word, self.get_wordnet_pos(pos_tag)) 
                    for (word, pos_tag) in nltk.tag.pos_tag(word_tokenize(x))]))

        return self.dataset

    def social_processing(self, column = "social_prepocessed_text") -> pd.DataFrame:
        self.dataset[column] = self.dataset['text'].apply(lambda x: " ".join(self.social_processor.pre_process_doc(x)))
        return self.dataset

    def emotions_extraction(self) -> pd.DataFrame:
        emotions = ['happy', 'angry', 'surprise', 'sad', 'fear']
        self.dataset[emotions] = self.dataset['text'].apply(lambda x: pd.Series(te.get_emotion(x)))
        return self.dataset

    def features_extraction(self) -> pd.DataFrame:
        self.dataset['c_counts'] = self.dataset['text'].apply(lambda x: len(x))
        self.dataset['w_counts'] = self.dataset['text'].apply(lambda x: len(x.split(" ")))
        return self.dataset

    def composed_preprocessing(self) -> pd.DataFrame:
        
        pass

if __name__ == "__main__":
    dataset      = pd.read_csv("suicide_squad_preprocessed.csv")
    preprocessor = Preprocessor(data = dataset)
    
    if True:
        dataset      = preprocessor.text_preprocessing(
            forward  = ['remove_url', 'remove_punctuation', 'remove_numbers', 'lemmatization', "stop_words"]
        )
        
        # dataset      = preprocessor.social_processing()
        # dataset      = preprocessor.emotions_extraction()

        # display(dataset)
        # dataset.to_csv("suicide_squad_preprocessed.csv", index = False)