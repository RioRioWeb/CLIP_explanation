import torch
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer
import spacy

tokenizer = SimpleTokenizer()

# SCOWLの単語ファイルから単語のリストを作成
def load_scowl_wordlist(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        words = file.read().splitlines() # 改行で単語に区切る
    return words

# SCOWLの単語リストからBPEで複数のトークンになる単語を除外→サブワードに自然言語化されない
def filter_single_token_words(word_list, tokenizer):
    single_token_words = []
    for word in word_list:
        encoded = tokenizer.encode(word)
        if len(encoded) == 1: # トークンIDのリストの長さが1の場合
            single_token_words.append(word)
    return single_token_words

file_path = 'scowl/scowl-words-20.txt' 
word_list = load_scowl_wordlist(file_path) # SCOWLの高頻度単語のリスト
filtered_words = filter_single_token_words(word_list, tokenizer) # 自然言語化に使う辞書

# 制限する品詞（名詞, 前置詞, 限定詞, 数詞, 動詞, 代名詞、形容詞, 副詞, 助動詞）
parts_of_speech = ('NOUN', 'ADP', 'DET', 'NUM', 'VERB', 'PRON','ADJ', 'ADV', 'AUX' )

# 自然言語化に使う辞書をspaCyで指定した品詞に制限
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"]) # spaCyモデルのロード
# vocab_words = [word for word in filtered_words if nlp(word)[0].pos_ in ('NOUN', 'ADP', 'DET') and len(nlp(word)) == 1]
vocab_words = [word for word in filtered_words if nlp(word)[0].pos_ in (parts_of_speech) and len(nlp(word)) == 1]
len(vocab_words)

# vocab_wordsをテキストファイルに保存
with open('vocab_words.txt', 'w', encoding='utf-8') as file:
    for word in vocab_words:
        file.write(word + '\n')








# ---------------------debag---------------------

# ---BPEとsprayで日常単語はサブワード化されないことを明らかにする---
text = "car vehicle automobile Electric convertible saloon sedan truck SUV steering wheel trunk gas hood take a car" # winker, cabrioletは複数トークン化
# text = "food dish meal cooking making diet taste smell apple sweet dessert ice drink sake yummy healthy weekly"

# # 日常単語は頻度が高いから、BPEトーカナイザーでトークン化してもサブワード化されない
# print(f"テキスト：{text}")
# print(f"テキスト内の単語数：{len(text.split())}")
# print(f"各単語に対応するBPE辞書のID：{tokenizer.encode(text)}")    
# print(f"トークン化されたテキストのトークン数：{torch.tensor(tokenizer.encode(text)).shape[0]}")    

# # 日常単語は頻度が高いから、spaCyの言語モデルで解析しても、サブワード化されない
# nlp = spacy.load("en_core_web_sm")     # spaCyの事前学習言語モデルをロード
# doc = nlp(text)                        # テキストを解析して、docオブジェクトを生成
# for token in doc: 
#     print(token.text, token.pos_)      # 単語とその品詞を出力
# print(f"spaCy_docの単語数{len(doc)}")

# # 以上から、日常単語のみを格納した辞書リストがあれば、自然言語化で日常単語意外の意味不明なサブワードが出力することはない。



# # # ---SCOWL辞書、自然言語化用の辞書、名詞に制限した自然言語化用の辞書について調べる---
# print(f"SCOWL辞書の単語数{len(word_list)}")
# print(word_list[:30])
# print(f"自然言語化に使う辞書の単語数{len(filtered_words)}")
# print(filtered_words[:30])
# print(f"名詞に制限した自然言語化用の辞書の単語数{len(NOUN_words)}")
# print(NOUN_words[:100])

# # SCOWL辞書に特定の単語が含まれるか調べる
# word_set = set(word_list) # 辞書をリストからセットに変換
# words_to_check = text.split() # 調べたい単語のリスト
# for word in words_to_check: # 各単語が辞書に含まれているか確認
#     if word in word_set:
#         print(f"'{word}' is in the vocab.")
#     else:
#         print(f"'{word}' is NOT in the vocab.")

# print("---------------------------------------")

# # 自然言語化に使う辞書に特定の単語が含まれるか調べる
# word_set = set(filtered_words) 
# words_to_check = text.split() 
# for word in words_to_check: 
#     if word in word_set:
#         print(f"'{word}' is in the vocab.")
#     else:
#         print(f"'{word}' is NOT in the vocab.")

# # 名詞に制限した自然言語化用の辞書に特定の単語が含まれるか調べる
# word_set = set(NOUN_words) 
# words_to_check = text.split() 
# for word in words_to_check: 
#     if word in word_set:
#         print(f"'{word}' is in the vocab.")
#     else:
#         print(f"'{word}' is NOT in the vocab.")

# # 名詞に制限した自然言語化用の辞書の語の品詞を調べる
# for word in NOUN_words[:100]:
#     doc = nlp(word)
#     for token in doc:
#         print(token.text, token.pos_)