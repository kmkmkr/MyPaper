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
  
####  探索方法が悪い→探索方法の変更

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
  
  グラフ名(対RandomPlayerとの勝率）、オレンジ：eの減衰率、青：loss　
 
![image](https://user-images.githubusercontent.com/75050667/197100984-4307f957-87b9-4d75-8c7a-ea4688a251f4.png)

  - softmax法、焼きなましsoftmax法

  焼きなましsoftmaxは調べた限り存在しなかった。しかし、おもしろそうだったので実装してみた。

  ![image](https://user-images.githubusercontent.com/75050667/197101070-04967a7a-6765-41a4-af6f-30c2d05d959f.png)

  - 焼きなましe-greedy法 & softmax法（私オリジナル）

  焼きなましe-greedy法の問題点はランダムな挙動をしたときに最大Q値もそれ以外のQ値と同列に扱ってしまう。いくら探索とはいえQ値が高いものを優先的に選んだほうが新たなQ値の高い行動をみつけやすい。そのため、完全にランダムではなくsoftmaxで最大Q値が選びやすくされるようにした。つまり、確率eでsoftmax、確率1-eで最大Q値となるようなe-greedy法を実装した。
  
  ---実験中---
 
 
#### 方策制御にランダムな挙動があるため学習結果にムラが出てしまうー＞同一モデルを複数回学習を繰り返し、勝率の分散を調べる

e-greedyのegp2000とegp4000では減衰率もほとんど変わらず、lossもうまく収束しているにもかかわらず勝率がそれぞれ69.25%、38.69%と大きく違う。また、egp10000ではlossが収束していないにもかかわらず勝率は67.14%と高い。私はDQNの特性上、lossが収束することと勝率は必ずしも一致しないと考えた。さらに、減衰率の違いによって勝率の分散も変化するのではないかと考えた。そのため、egp2000とegp10000のモデルの学習を複数回繰り返し、lossと勝率の関係、減衰率と勝率の分散について実験した。


---実験中---



### 20221024

参考：https://www.slideshare.net/juneokumura/dqnrainbow


実装済み
- qn_parallel.py->並列実行, DuelingN, DoubleDQN

- dqn_per->並列実行, DuelingN, DoubleDQN, PER

- per:priorized experimence replay

未実装

- Categorical DQN

- Multi-Step RL

- Noisy Net


気づき

- ReplayMemoryのmemoryはサイズが決まっている。そして、データの残量が0になってもmemory内のデータを削除する機能はない。よって、最初のメモリサイズ以上の経験データは記録されず、学習に反映されない。

- memoryサイズは256*512。つまり、バッチを512個作れる。また、1エピソード約60個なので、メモリは約2185エピソード保存できる

- originalのイプシロン減衰率は0～1000エピソードで0.5、2000エピソードで0.35と急激に下がる。これは、メモリサイズを意識したもの？

- 2step処理にするとエピソードが60手で終わらず、59手で終わってしまうと59手目と次エピソードの1手目が経験データとして記録されてしまう。doneで識別？

Done

MultiStepReplayMemory作成（MultiStep.ipynb)


### 20221025

気づき

- ReplayMemoryのmemoryはサイズが決まっている。そのため、イプシロン減衰率が緩やかなものはランダムな行動をしたものが多く記録されgreedyな行動は記録されない。

Done
- dqn_ms.pyを作成し、実行。探索：ただおe-greedy、メモリ：MultiStepReplayMemory

python dqn_ms.py --model ./model/dqn_ms/resume_model_egreedy.pt --resume ./model/train_from_training_data/model_data5.pt --log ./log/dqn_ms/egreedy.log --num_episodes 160000

- dqn_ms.py でgameplay.pyを実行できるように変更
    






