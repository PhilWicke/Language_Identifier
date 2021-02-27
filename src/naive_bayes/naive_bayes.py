class NaiveBayesClassifier:

  def __init__(self):
    self.docLanguageCounts = dict()
    self.docCount = 0
    self.dictionary = set()

    self.languageWordCounts = dict()
    self.globalLanguageWords = dict()
  
  def train(self, data, class_limit=235):
    for sample in tqdm(data):

      if len(self.docLanguageCounts.keys()) >= class_limit:
          continue

      nGrams = sample[0]
      label = sample[1]

      self.docCount += 1
      if label not in self.docLanguageCounts.keys():
        self.docLanguageCounts[label] = 1
      else:
        self.docLanguageCounts[label] +=1

      wordCount = 0
      for word in nGrams:
        wordCount += 1
        self.dictionary.add(word)
        if (label, word) not in self.languageWordCounts.keys():
          self.languageWordCounts[(label, word)] = 1
        else:
          self.languageWordCounts[(label, word)] += 1
      
      if label not in self.globalLanguageWords.keys():
        self.globalLanguageWords[label] = wordCount
      else:
        self.globalLanguageWords[label] += wordCount

  def p_wordGivenLang(self, word, lang_label, lambda_value):
    if (lang_label, word) not in self.languageWordCounts.keys():
      return 0
    else:
      return (self.languageWordCounts[(lang_label, word)] + lambda_value) / (self.globalLanguageWords[lang_label] + (lambda_value * len(self.dictionary)))
    
  def p_Lang(self, lang_label):
    return self.docLanguageCounts[lang_label] / self.docCount

  def p_docAndLang(self, nGrams, lang_label, lambda_value):
    probWordsInDoc = 1
    for word in nGrams:
      probWordsInDoc *= self.p_wordGivenLang(word, lang_label, lambda_value)
    return self.p_Lang(lang_label) * probWordsInDoc

  def mostLikelyLanguage(self, nGrams, lambda_value):
    maxProbability = 0
    bestGuess = "---"
    for language in self.docLanguageCounts.keys():
      probability = self.p_docAndLang(nGrams, language, lambda_value)
      if probability > maxProbability:
        maxProbability = probability
        bestGuess = language
    return bestGuess
	
# Text processing tools
def collectNGrams(n, text):
  text = "#"*n+text+n*"#"
  return [text[i:i+n] for i in range(len(text)-(n))][1:]
  

def loadDataToNGrams(n, path):
  train_data = []
  with open(path , "r") as train_in:
    train_lines = train_in.readlines()
    for line in train_lines:
      content = line.split("|")
      id = content[0]
      text = content[1]
      label = content[2].strip()
      nGrams = collectNGrams(n, text)

      train_data.append((nGrams, label))
  return train_data
  
def convertWiLiData():
	with open("data/wili-2018/x_train.txt", "r") as x_train_file:
		x_train_lines = x_train_file.readlines()
		x_train_lines = [line.replace(",", "") for line in x_train_lines]

	with open("data/wili-2018/y_train.txt", "r") as y_train_file:
	  y_train_lines = y_train_file.readlines()

	with open("data/wili-2018/train_data.csv" , "w") as train_out:
	  index_no = 1
	  for x_line, y_line in zip(x_train_lines, y_train_lines):
		train_out.write("train.s"+str(index_no)+"|"+x_line.strip()+"|"+y_line.strip()+"\n")
		index_no+=1

######################################		
### Run Naive Bayes Classification ###
######################################

convertWiLiData()
train_data = loadDataToNGrams(2, "data/wili-2018/train_data.csv")
nbModel = NaiveBayesClassifier()
nbModel.train(train_data)