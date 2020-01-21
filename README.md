論文標註競賽
===
比賽給予多篇論文的Abstract，並且依照段落做切割，每個段落都會對應到標籤，標籤種類有Background、Objectives、Methods、Results、Conclusions與Others，而每個段落可能包含多個標籤，是屬於多類別問題。目的是希望可以準確的分類段落的標籤。

作法概念
---
將文本輸入到BERT模型後，取得[CLS]向量，即為文本向量，再利用文本向量接續分類演算法。由於分類問題是屬於多類別問題，故將每個標籤拆開，每篇文本都會針對每個標籤做一個二元分類，屬於該類別或不屬於該類別。

作法過程
---
參考方法: https://github.com/bojone/bert_in_keras/blob/master/sentiment.py

1.載入事先下載的Pre-trained Model，我載的是cased_L-12_H-768_A-12(來源:https://github.com/google-research/bert#pre-trained-models)。

2.將pre-trained model的字典載入，給予每個字在字典內的索引位置(index)，利用tokenizer後的字包含index以及segment。

3.將檔案切割成我們需要的輸入格式，由於每個標籤都要做一次二元分類，所以每個模型僅選擇單一標籤，此處選擇background。CSV格式為(text,label)。

4.文本長短不一樣，為了要讓長度一樣，我們會塞入padding。這邊作法是以batch內句子長度最長者為依據，其他句子長度則塞到跟最長者一樣。

5.data generator。將data每次依據batch size大小輸入，並塞入padding。最後將文字轉換成keras_bert輸入格式，包含indices list以及segment list。輸入到模型的格式為[X1,X2,Y]，X1為indices list，X2為segment list，Y為label。

6.模型驗證使用5-fold cross validation做交叉驗證，避免模型過擬於單一結果。最終模型是先取得BERT模型最後一層的[CLS](符號代表文本向量)，並多加一層分類器在[CLS]後面，activation function使用sigmoid，loss function則是binary-crossentropy，利用Adam方式學習，最後用f1-score評估成效。
