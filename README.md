# task3-toxic_comment_classification

标签（空格分隔）： python tensorflow

---

## 一、比赛结果 ##
网址链接： https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/leaderboard
我的最高分数：0.9828  比赛最高分数： 0.9885  比赛排名：1835/4551
![image.png-21.4kB][1]


![image.png-20.3kB][2]


![image.png-27kB][3]


## 二、代码说明 ##
使用python + tensorflow实现
1. test.csv train.csv 官方给的训练和测试数据
2. main.py 程序执行的主文件，其中包括数据预处理、模型的训练测试、生成预测结果
3. utils/model.py: 基于tensorflow实现的RNN模型
4. utils/embedding: 生成词的embedding的函数
5. utils/mynltk.py: 实现分词、统一转小写字母、词字典构建
6. esemble.py: 对不同模型预测的结果进行esemble，生成最后提交的文件。

## 三、实现技术细节 ##
1. 使用预训练好的300维词向量crawl-300d-2M.vec

   下载链接https://github.com/facebookresearch/fastText/blob/master/docs/english-vectors.md
2.  数据预处理：包括去除低频词、大写转小写

3.  模型细节：

    a. 预训练好的词向量
    
    b. 两层双向GRU网络
    
    c. 两层GRU之间引入dropout
    
    d. 最后一层GRU引入attention
    
    e. 将GRU最后的输出经过全连接网络，最后再输出到6个神经元并用sigmoid激活得到对应的6种预测概率。
    
    f. 计算log损失并用RMSProp优化算法进行优化
    
4.  训练过程：训练集与测试集比为9：1，保存训练中最好的模型参数，并且若连续三次测试集损失不再下降，则停止训练。

5.  esemble: 训练多次模型将预测概率进行esemble(和平均或者积平均)作为最后的输出。

## 四、实验各个模型结果 ##
model: 不同的训练模型ID，有相同参数的模型和也有不同参数的模型

score: 官网测试对应模型的评分

dr: GRU的dropout概率

lr: 模型学习率

cut: 去除的最小词频的统计数

cg: clip gredient的值

batch: 训练batch的大小

maxlen: RNN的最长时间长度

hidden: RNN的隐藏神经单元的大小

|model|score|dr|lr|cut|cg|batch|maxlen|hidden|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
|1|0.9787|0.5|5e-4|3|5|256|500|64|
|2|0.9793|0.5|5e-4|3|5|256|500|64|
|3|0.9780|0.5|5e-4|3|5|256|500|64|
|4|0.9788|0.5|5e-4|3|5|256|500|64|
|5|0.9782|0.5|5e-4|3|5|256|500|64|
|6|0.9782|0.5|5e-4|3|5|256|500|64|
|7|0.9782|0.5|5e-4|3|5|256|500|128|
|8|0.9787|0.5|5e-4|5|1|256|500|64|
|9|0.9801|0.5|5e-4|3|1|256|500|64|
|10|0.9790|0.5|5e-4|5|1|256|750|64|
|11|0.9795|0.3|5e-4|5|5|256|500|64|
|12|0.9785|0.3|5e-4|0|5|256|500|64|
|13|0.9783|0.3|1e-3|5|5|256|500|64|
|14|0.9781|0.3|2e-4|5|5|256|500|64|



  [1]: http://static.zybuluo.com/njuzrs/b9xuthpa3ccagkfur8f19s8t/image.png
  [2]: http://static.zybuluo.com/njuzrs/0fveo2j2o28v6k9fmkl19zfh/image.png
  [3]: http://static.zybuluo.com/njuzrs/3l91rn686m41i27ug4taclc0/image.png