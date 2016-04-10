# code by Dong-Sig Han
# Manually Copied/Modified Facebook/MemNN/MemN2N-babi-matlab/parseBabiTask.m
# Using data in en/*.txt
# data_path : python list that stores paths of data 
# word_dict : word -> embedding index
# Python2.7

import numpy as np

def parseBabiTask(data_path, word_dict, given_max_word=-1, given_max_sentences=-1, include_question = False):
  # data initilization
  story     = np.zeros((20, 1000, 1000*len(data_path)), np.int32)
  questions = -np.ones((15, 1000*len(data_path)), np.int32)
  qstory    = np.zeros((20, 1000*len(data_path)), np.int32)

  story_ind    = -1
  sentence_ind = -1

  max_words     = 0
  max_sentences = 0

  question_ind = -1

  for fi in range(len(data_path)):
    line_ind = -1
    f = open(data_path[fi])
    for line in f:
      line_ind += 1
      words = line.split()
      
      if words[0] == '1':
        story_ind += 1
        sentence_ind = -1
        map_ = []

      if line.find('?') < 0:
        is_question = False
        sentence_ind += 1
      else:
        is_question = True
        question_ind += 1
        questions[0, question_ind] = story_ind
        questions[1, question_ind] = sentence_ind
        if include_question:
          sentence_ind += 1

      map_.append(sentence_ind)

      for k in range(1, len(words)):
        w = words[k]
        w = w.lower()
        if w[len(w)-1] == '.' or w[len(w)-1] == '?':
          w = w[0:len(w)-1]
        if not(word_dict.has_key(w)):
          word_dict[w] = len(word_dict)
        max_words = max(max_words, k)

        if not(is_question):
          story[k-1, sentence_ind, story_ind] = word_dict[w]
        else:
          qstory[k-1, question_ind] = word_dict[w]
          if include_question:
            story[k-1, sentence_ind, story_ind] = word_dict[w]

          if words[k][len(words[k])-1] == '?':
            answer = words[k+1]
            answer = answer.lower()
            if not(word_dict.has_key(answer)):
              word_dict[answer] = len(word_dict);

            questions[2, question_ind] = word_dict[answer]
            for h in range(k+2, len(words)):
              questions[1+h-k, question_ind] = map_[int(words[h])-1]
            questions[14, question_ind] = line_ind
            break
      max_sentences = max(max_sentences, sentence_ind+1)
    
  story     = np.transpose(story[0:max_words, 0:max(max_sentences, given_max_sentences), 0:story_ind+1])
  questions = np.transpose(questions[:, 0:question_ind+1])
  qstory    = np.transpose(qstory[0:max_words, 0:question_ind+1])

  return (story, questions, qstory)