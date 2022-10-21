# 卒論テーマ
深層強化学習を用いたオセロAIの作成

## 研究概要

## 研究背景

## Log

### 前期（2022401~2022901)

主に強化学習の基礎を勉強

- Q学習・DQNの勉強

参考：https://www.oreilly.co.jp/books/9784873119755/　

- Q学習の実装

参考：https://github.com/TadaoYamaoka/creversi　


- DQNの実装

参考：https://github.com/TadaoYamaoka/creversi_gym　


#### 結果

|  model  |  対RandomAgentとの勝率  |
| ---- | ---- |
|  Q学習  |  0.4253  |
|  DQN  |  0.4088  |
|  教師ありDQN  |  0.6436  |
|  教師ありDQN+DQN  |  0.6009  |


### 後期
### 9月～10月

- 教師ありDQN→教師ありDQN+DQNの性能の低下の原因
  - 探索方法が悪い
  - 方策制御にランダムな挙動があるため学習結果にムラが出てしまう
  
- 探索方法の変更

  参考：https://qiita.com/pocokhc/items/fd133053fa309bdb58e6

  実際に試したもの

  - 焼きなましe-greedy法の減衰率の変更
  - softmax法
  - 焼きなましsoftmax法（私オリジナル）
  - 焼きなましe-greedy法 & softmax法（私オリジナル）
  
 結果
  - 焼きなましe-greedy法のeの減衰率の変更
    - egp2000~18000 値が大きくなるほど減衰率が低下し、ランダムな挙動が増える
    - liner：減衰率を線形に
  
  オレンジ：eの減衰率、青：loss
 
  ![image](https://user-images.githubusercontent.com/75050667/197097356-381b4918-34ac-4688-ac57-c28ac266732c.png)


 
    






