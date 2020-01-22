論文標註競賽
===
比賽給予多篇論文的Abstract，並且依照段落做切割，每個段落都會對應到標籤，標籤種類有Background、Objectives、Methods、Results、Conclusions與Others，而每個段落可能包含多個標籤，是屬於多類別問題。目的是希望可以準確的分類段落的標籤。

作法概念
---
首先將整句話轉換成向量的形式，再依據向量去做分類。

BERT是一個依據Transformer的Encoder模型所轉化而來，利用Multi-head Attention機制作為模型學習的方法，再利用MLM(Masked Language Model)以及NSP(Next Sentence Prediction)作為訓練任務，並利用深層(12或24層)、多neuron(768或1024，代表向量維度)、多head(12或16)，總參數高達110或340M。

1.每個字詞透過Self-Attention機制可以學習到與其他字詞的關聯強度；

2.透過Multi-Head Attention可得知在不同語境下，兩字之間的關聯強度；

3.在MLM訓練任務下，每個字詞可以得到雙向語句資訊，即從該字的左邊或從該字的右邊學習到相關資訊；

4.多層隱藏層的學習，可以累積字詞的資訊到每個字身上。假設整篇有20個字，第10個字(word10)會學習到前9個字累積而來的資訊(word1->word2->...->word10)以及第20個字往前累積而來的資訊(word20->word19->...->word10)。

我們利用KERAS所開發出的套件，keras_bert，作為我們使用的工具。將語句的每個字先轉換成BERT輸入的格式後，通過模型即可得到語句向量。再將語句向量接續分類演算法即可達到分類效果。我們利用別人的pre-trained model做使用(感謝前人的貢獻)，BERT模型好處是可以做fine-tune，根據後續的分類任務去微調整體模型的參數，但在設備資源的關係下就沒有微調所有參數，只單純調整最後分類的參數。

作法過程
---
參考方法: https://github.com/bojone/bert_in_keras/blob/master/sentiment.py

檔案: 5foldCV_keras-bert-binaryBackground.ipynb

1.載入事先下載的Pre-trained Model，我載的是cased_L-12_H-768_A-12(來源:https://github.com/google-research/bert#pre-trained-models)。

2.將pre-trained model的字典載入，給予每個字在字典內的索引位置(index)，利用tokenizer後的字包含index以及segment。

3.將檔案切割成我們需要的輸入格式，由於每個標籤都要做一次二元分類，所以每個模型僅選擇單一標籤，此處選擇background。CSV格式為(text,label)。

4.文本長短不一樣，為了要讓長度一樣，我們會塞入padding。這邊作法是以batch內句子長度最長者為依據，其他句子長度則塞到跟最長者一樣。

5.data generator。將data每次依據batch size大小輸入，並塞入padding。最後將文字轉換成keras_bert輸入格式，包含indices list以及segment list。輸入到模型的格式為[X1,X2,Y]，X1為indices list，X2為segment list，Y為label。

6.模型驗證使用5-fold cross validation做交叉驗證，避免模型過擬於單一結果。最終模型是先取得BERT模型最後一層的CLS(符號代表文本向量)，並多加一層分類器在CLS後面，activation function使用sigmoid，loss function則是binary-crossentropy，利用Adam方式學習，最後用f1-score評估成效。
