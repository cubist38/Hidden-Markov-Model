"""
# Gán nhãn từ loại (Part-of-Speech tagging)

1. **Gán nhãn từ loại** là gì?

  Như hồi ở tiểu học, chúng ta đã được học về những từ loại tạo nên một câu trong tiếng anh chẳng hạn như: noun, verb, adjective, adverb... Việc xác định các chức năng ngữ pháp của từ trong câu hay là quá trình gán từng từ trong đoạn văn bản với các đánh dấu từ loại hoặc cấu trúc ngữ pháp được gọi là gán nhãn từ loại. Đây là bước cơ bản trước khi phân tích cú pháp hay các vấn đề xử lý ngôn ngữ phức tạp khác.

2. Nó được ứng dụng ở đâu?
 
  **Gán nhãn từ loại (Part-of-Speech tagging)** có lẽ là bài toán sớm nhất được nghiên cứu và được mọi người biết đến khi nhập môn chuyên ngành **xử lý ngôn ngữ tự nhiên (Natural Language Processing - NLP)**.


3. Mô tả
  
  Trong gán nhãn từ loại, mục tiêu của chúng ta khi đưa chuỗi đầu vào là một câu ví dụ như "Peter really loves dogs" thì chuỗi đầu ra sẽ là nhãn NOUN ADVERB VERB NOUN.

4. Ứng dụng mô hình Markov ẩn

  Mô hình Markov ẩn (tiếng Anh là Hidden Markov Model - HMM) là một kĩ thuật ngẫu nhiên được sử dụng trong gán nhãn từ loại. Ta có thể mô hình hoá bài toán này như sau:
  * Từ đại diện cho quan sát (observations)
  * Từ loại đại diện cho trạng thái (states)
 
  Bài toán đặt ra là ta sẽ dùng thuật toán viterbi để tìm ra chuỗi từ loại cho một chuỗi trạng thái được nhập vào hay nói cách mô hình hơn là ta đi tìm chuỗi trạng thái khả dĩ nhất Q đã phát sinh ra chuỗi quan sát O.

  Dựa vào ví dụ trên "Peter really loves dogs" ta có thể mô hình hoá như sau:
  * $O = \{Peter,really,loves,dogs\}$
  * $Q = \{NOUN,ADVERB,VERB,NOUN\}$

5. Dữ liệu

  Ta sẽ dùng tập dữ liệu của thư viện NLTK - Natural Language Toolkit là một trong những thư viện open-source xử lí ngôn ngữ tự nhiên. Được viết bằng Python và với ưu điểm là dễ dàng sử dụng nên thư viện này ngày càng trở nên phổ biến và có được một cộng đồng lớn mạnh. Thư viện cung cấp hơn 50 kho dữ liệu văn bản khác nhau (corpora) và nhiều chức năng để xử lí dữ liệu văn bản để phục vụ cho nhiều mục đích khác nhau
"""

from google.colab import drive
drive.mount('/content/drive')

# Gọi thư viện
import pandas as pd 
import nltk
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import pprint, time

def readTreeBankCorpusFromNLTK():
  # Tải ngân hàng cây (kho dữ liệu văn bản) từ nltk
  nltk.download('treebank')
  # Tải kiểu gán nhãn universal từ nltk
  nltk.download('universal_tagset')
  
  # Đọc dữ liệu từ ngân hàng cây thành những câu được gán nhãn
  data = list(nltk.corpus.treebank.tagged_sents(tagset = 'universal'))
  return data

data = readTreeBankCorpusFromNLTK()

"""# Tiền dữ lý dữ liệu
  Ta sẽ chia dữ liệu ban đầu ra làm tập train và tập test:

* Tập train dùng để huấn luyện ma trận chuyển trạng thái (trainsition_matrix).

* Tập test dùng để tìm ra chuỗi trạng thái mà trong bài này là chuỗi từ loại khả dĩ nhất đã sinh ra chuỗi quan sát trong bài này là chuỗi từ bằng thuật toán viterbi để chạy kiểm tra xem độ chính xác.
"""

# chia dữ liệu ra với tỉ lệ train:test là 80:20
train_dataset,test_dataset = train_test_split(data, train_size=0.80, test_size= 0.20, random_state = 101)

"""Hàm bên dưới trích xuất các từ loại từ tập dữ liệu. 

Chẳng hạn ở bài này ta trích xuất ra danh từ (NOUN), động từ (VERB), trạng từ (ADV) và tính từ (ADJ).
"""

def dataTaggedPreprocessing(data, tagset):
  tagged_words = []
  for sent in data:
    for tagged_word in sent:
      if tagged_word[1] in tagset:
        tagged_words.append(tagged_word) 
  return tagged_words

tagset = ['NOUN', 'VERB', 'ADV', 'ADJ'] # tập từ loại cần trích xuất
train_tagged_words = dataTaggedPreprocessing(train_dataset, tagset)  #tập dữ liệu dùng để huấn luyên (sau khi trích xuất).
test_tagged_words = dataTaggedPreprocessing(test_dataset, tagset)  # tập dữ liệu dùng để đánh giá (sau khi trích xuất).

# Hàm tính xác suất có điều kiện của trạng thái sau là "state2" khi biết trước trạng thái trước là "state1"

def probOfState2GivenState1(state2, state1, observation_state_list):
  states = []
  for tup in observation_state_list:
    states.append(tup[1])
  count_state1 = 0
  count_state2_given_state1 = 0
  length = len(states)
  for i in range(length - 1):
    if states[i] == state1:
      count_state1 = count_state1 + 1
  for i in range(length - 1):
    if states[i] == state1 and states[i+1] == state2:
      count_state2_given_state1 = count_state2_given_state1 + 1
  return count_state2_given_state1/count_state1

states = tagset # Các trạng thái
N = len(states) # Số lượng trạng thái

# Hàm tính xác suất có điều kiện của quan sát "observation" ở trạng thái "state" 

def probOfObservationGivenState(observation, state, observation_state_list = train_tagged_words):
  count_State = 0
  count_Observation_Given_State = 0
  for t in observation_state_list:
    if t[1] == state:
      if t[0] == observation:
        count_Observation_Given_State = count_Observation_Given_State + 1
      count_State = count_State + 1
  return count_Observation_Given_State/count_State

"""Ta sẽ tính toán **transition_matrix** (ma trận xác suất chuyển trạng thái) bằng tập huấn luyện đã được gán nhãn **train_tagged_words** (ở trên). """

transition_matrix = np.empty((N, N), dtype = 'float32')  # ma trận chuyển trạng thái tính từ tập huấn luyện.

for i in range(N):
  for j in range(N):
    transition_matrix[i][j] = probOfState2GivenState1(states[j], states[i], train_tagged_words) # xác suất chuyển từ trạng thái j sang trạng thái i

pd.DataFrame(transition_matrix).to_csv("/content/drive/MyDrive/POST/trainsition_matrix.csv")
print(transition_matrix)

"""# Thuật toán Viterbi"""

def viterbi_HMMs(O, A, pi, Q):
    # O là chuỗi quan sát đầu vào
    # A là ma trận xác suất chuyển đổi trạng thái được tính dựa trên tập train
    # pi là phân phối xác suất khởi đầu
    # Q là tập các trạng thái đã cho
    hidden_states = [] # chuỗi trạng thái cần tìm


    prev_state = -1
    for index, observation in enumerate(O):
      delta = []
      for s in range(len(Q)):
        if index == 0:
          transition_prob = pi[s]
        else:
          transition_prob = A[prev_state, s]

        emission_prob = probOfObservationGivenState(observation, Q[s])
        delta.append(emission_prob*transition_prob)

      delta_max = max(delta)

      prev_state = delta.index(delta_max)
      hidden_states.append(Q[prev_state])
    
    return hidden_states

test_untagged_words = [tup[0] for tup in test_tagged_words] # tập từ chưa được gán nhãn từ loại dùng để đánh giá
pi = [0.3, 0.3, 0.2, 0.2]

print(test_untagged_words)

"""# Phát biểu bài toán

Tập **test_untagged_words** là tập từ chưa được gán nhãn từ loại. Đứng trên góc nhìn của mô hình Markov ẩn ta dễ dàng ánh xạ qua như sau:
* test_untagged_words là các quan sát được cho trước.
* states = {"NOUN", "VERB", "ADV", "ADJ"} là tập các trạng thái 
* pi = [0.3, 0.3, 0.2, 0.2] là phân phối xác suất khởi đầu
* transition_matrix là ma trận xác suất chuyển đổi trạng thái

Do đó, ta dễ dàng gán nhãn từ loại của tập **test_untagged_words** bằng việc áp dụng thuật toán **Viterbi**.
"""

tagged_seq = viterbi_HMMs(test_untagged_words, transition_matrix, pi, states)

print(tagged_seq)

"""# Đánh giá

Bây giờ, ta sẽ đánh giá mô hình bằng cách tính độ chính xác giữa chuỗi trạng thái được thuật toán **Viterbi** tìm ra và chuỗi trạng thái thực của tập **test_tagged_words**.


"""

cnt = 0
for i in range(len(test_tagged_words)):
  if test_tagged_words[i][1] == tagged_seq[i]:
    cnt = cnt + 1

print("Accuray Of Viterbi Algorithm = %.4f %%"%(cnt/len(test_tagged_words)*100))

"""# Nhận xét và đưa ra cách cải tiến

Ở cách làm trên, ta nhận thấy rằng sẽ có nhiều trường hợp mà từ trong tập đánh giá **test_untagged_words** chưa từng xuất hiện trong tập huấn luyện **train_tagged_words**, điều này làm cho mô hình luôn dự đoán rằng từ này là danh từ (NOUN). Thật vậy, giả sử từ này là word thì ta sẽ có:
* $P(word|NOUN) = 0$
* $P(word|VERB) = 0$
* $P(word|ADV) = 0$
* $P(word|ADJ) = 0$

Điều này sẽ làm cho thuật toán Viterbi đưa về dự đoán là trạng thái 0 tức là danh từ (NOUN) vì $delta_{max} = 0$. Do đó, để cải thiện vấn đề này ta sẽ dùng một kĩ thuật khá đơn giản được gọi là "rule_based". Chúng ta sẽ áp dụng kĩ thuật này như sau:
* Nếu **untagged_word** (từ chưa được gán nhãn ở tập đánh giá **test_untagged_words**) đã xuất hiện trong tập huấn luyện **train_tagged_words** thì ta sẽ dùng thuật toán **Viterbi** (cũ) để gán nhãn từ loại.
* Nếu **untagged_word** (từ chưa được gán nhãn ở tập đánh giá **test_untagged_words**) chưa từng xuất hiện trong tập huấn luyện **train_tagged_words** thì ta sẽ xem xét đến những cụm chữ cái ở đuôi. Chẳng hạn như "ing", "ed" thì ta sẽ gán **untagged_word** này là động từ (VERB), còn "s" "es", "\'s" ta sẽ gán cho nó là danh từ (NOUN). Có thể tham khảo thêm tại bài báo này https://aclanthology.org/A92-1021.pdf được đề xuất bởi Eric Brill. 

Để áp dụng ý tưởng này vào bài toán nhóm đã sử dụng lớp RegexpTagger của thư viện NLTK để dễ dàng hơn trong việc xử lý. Có thể tham khảo về mã nguồn của lớp này ở đường link: https://tedboy.github.io/nlps/generated/generated/nltk.RegexpTagger.html
"""

patterns = [
    (r'.*ing$', 'VERB'),              # gerund
    (r'.*ed$', 'VERB'),               # past tense 
    (r'.*es$', 'VERB'),               # verb    
    (r'.*\'s$', 'NOUN'),              # possessive nouns
    (r'.*s$', 'NOUN'),                # plural nouns 
    (r'.*', 'NOUN')                   # nouns           
]
rule_based_tagger = nltk.RegexpTagger(patterns)

def viterbi_HMMs_rule_based(O, A, pi, Q):
    # O là chuỗi quan sát đầu vào
    # A là ma trận xác suất chuyển đổi trạng thái được tính dựa trên tập train
    # pi là phân phối xác suất khởi đầu
    # Q là tập các trạng thái đã cho
    hidden_states = [] # chuỗi trạng thái cần tìm


    prev_state = -1
    for index, observation in enumerate(O):
      delta = []
      for s in range(len(Q)):
        if index == 0:
          transition_prob = pi[s]
        else:
          transition_prob = A[prev_state, s]

        emission_prob = probOfObservationGivenState(observation, Q[s])
        delta.append(emission_prob*transition_prob)

      delta_max = max(delta)
      if delta_max == 0:
        hidden_states.append(rule_based_tagger.tag([observation])[0][1])
      else:
        prev_state = delta.index(delta_max)
        hidden_states.append(Q[prev_state])
    
    return hidden_states

tagged_seq_rule_based = viterbi_HMMs_rule_based(test_untagged_words, transition_matrix, pi, states)
print(tagged_seq_rule_based)

cnt = 0
for i in range(len(test_tagged_words)):
  if test_tagged_words[i][1] == tagged_seq_rule_based[i]:
    cnt = cnt + 1

print("Accuray Of Viterbi Algorithm Using Rule Based = %.4f %%"%(cnt/len(test_tagged_words)*100))

"""#Nhận xét
Ta có thể thấy rằng độ chính xác đã tăng lên nhờ sử dụng **rule_based**.
"""

def confusion_matrix(y_true, y_pred):
  N = len(list(set(y_true)))
  confusion_matrix = np.zeros((N, N), dtype = 'int')
  for i in range(len(y_true)):
    true_pos = states.index(y_true[i])
    pred_pos = states.index(y_pred[i])
    confusion_matrix[true_pos, pred_pos] += 1
  return confusion_matrix

test_tagged = [tup[1] for tup in test_tagged_words]
print(test_tagged)
