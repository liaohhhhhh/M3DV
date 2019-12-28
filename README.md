# 代码结构与运行说明
## 代码结构
### 完整模块
* `main.py`为训练的主程序，包括模型初始化，数据集加载，训练过程
* `DenseNet.py`为神经网络结构代码
* `MyDataset.py`为自定义数据集类型，包括读取训练数据，读取测试数据与结果输出三个部分
* `Transform.py`为数据预处理部分，包括了对数据进行旋转，对称等操作

### 测试模块（供助教使用）
* `test.py`为测试文件，只需要打开这个文件并输入数据存放路径和模型路径即可生成`submission.csv`，运行方法如下
* 程序要求输入`datas_dir`与`--model_params_dir`两项。前面一项需要输入数据存放的目录。后一项需要输入模型参数的储存路径
#### Example
* `--datas_dir:./test`数据存放在文件夹`test`中
* `--model_params_dir:net_params0.631.pkl`模型文件名为`net_params0.631.pkl`

