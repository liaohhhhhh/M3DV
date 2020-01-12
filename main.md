# Main

这个文档是给你们入门用的, 文档内容包括但不限于:

- Notice: 包括代码规范, 代码管理, 工作日志, 怎么使用这份文档 (持续更新)
- Tasks: 用于发布每一周要做的工作, 是一个工作列表
- QA collection: 把你们遇到的问题和解决方案记录下来 (后续可能转为git issue的形式)
- Resource collection: 有用的paper和codes资源

后续可能会放到`github`上.

## Notice

### 环境 & 语言

基础环境我已经写进bashrc了, 基础环境的名字叫gyf

建议先做一下clone, 然后用自己的环境:

```
conda create -n yzm --clone gyf
```

### 代码规范

**Updated 2020.1.8**:

(入门级) 至少保证核心函数/类有功能级的注释, 例子:

```python
    def encode(self, boxes, labels, masks,input_size):
        '''
        (介绍功能)
        Encode target bounding boxes and class labels.
        We obey the Faster RCNN box coder:
          tx = (x - anchor_x) / anchor_w
          ty = (y - anchor_y) / anchor_h
          tw = log(w / anchor_w)
          th = log(h / anchor_h)
        (输入参数: (类型) 简介)
        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].
          masks: (tensor) object mask labels, sized [#obj,].
          input_size: (int/tuple) model input size of (w,h).
        (输出参数: (类型) 简介)
        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
          mask_targets:(tensor) encoded mask labels, sized [#anchors,].
        '''
       	inputs_size = ....
        
```

### 版本管理

**Updated 2020.1.8**

- 使用 `git` 进行版本管理, 入门教程见 Resource collection里的 Git 

- `commit`的规范基本参考自 [AngularJS](https://docs.google.com/document/d/1QrDFcIiPjSLDn3EL15IJygNPiHORgU1_OOAqWjiDU5Y/edit#heading=h.greljkmo14y0) , 做了一下自己的简化, 总结如下:

  - **Title**: `<type>: <subject> ` , 例子: `Update: The EAD2020 dataset is released` . 总结一下这次更新是什么type (主要是`Update`, `Debug`应该), 简单介绍做了什么(非必须)

  - **Body**: 分点列写, 主要写你改了什么文件, 做了什么(不需要写得很详细, 详细说明应该在注释里). 例子:

    ```
    Update: EAD2020 dataset is released
    <空一行>
    1. EADdet.py is created.
    2. main.py add interface to EADdet dataset.
    ```

### 工作日志

**Updated 2020.1.8**:

- 每个人的style都不太一样, **一个要点是工作日志要能够让自己知道某事某刻做了什么以及接下来要做什么**.

给个我自己的例子:

- [x] (9:25 ~ 11:00 ) (Signet) Evaluate the paper of the first reviewer; Think about how to include in the related work; Find the work that is similar

> **pre-分工**:
>
> - 我: 把两个实验补全 (pure clean + modulating factor), 找验证hard examples learning的方法; 针对第一个reviewer的意见看paper
> - 林少: 针对第二个reviewer的reply初稿; 
>
> **Summary of the suggestion from the first reviewer**:
>
> > **Trying to reply/ Practice writing**: 
> >
> > - (Related work) Though both datasets are partially annotated, the main problem solved in this paper is the **part-subject class missing problem (unverified part class problem)** (?) in OID Dataset. For example, the person class is annotated while the human-arm or human-nose is not. This paper solves this problem by ignoring the learning signal (false training samples)  for part classes proposals that are inside the subject class bounding boxes. While for our dataset, **the missing annotation is among instances in the same level** (?). **There is no subject-part pairs in our dataset.** We hope our method is complementary to the co-occur mechanism.
> > - (Replying concern 1): For down-weighting on the noisy examples, this is already using decoupling mechanism. While our baseline is set as not using decoupling mechanism.
> > - (Replying concern 2): The hard-example learning ability could be reflected by the A-recall: that our method shows a higher recall rate in annotated regions. Also, more experiments would be included. (With the modulating factor 1 on the clean examples and modulating factors 2 on noisy examples).
> > - (Replying concern 3): Noisy examples are that we know parts of them are in-correctly labeled, but we do not know which exactly is wrong (note that the yellow boxes are unavailable). 
> >
> > **Points**: ##这里是零散的记录阅读论文时候的一些小知识点
> >
> > < Needs to read some papers about solving the problem lies in OID Dataset.
> >
> >  < The framework is Chainer: 看起来是个弱化版的Pytorch.
> >
> > < It seems that cosine annealing makes performance boost.
> >
> > < Ignoring the learning signal method can be explored (if have time)
> >
> > **Problem-Solution keys**: ##这里是记录阅读论文的时候的问题－解决方案总结
> >
> > - part-subject: use co-occurrence loss (?) (ignore learning signals for classifying the part ) **Notes**: need to find the pairs by human...
> > - class-imbalance: train models exclusively on rare classes and ensembles them with common classes. (Q: how to identify the "rare"? <1000 images)
> > - large-scale: using **ChainerMN**(what is it?) and 512 GPUs (??? Is it a contribution?)
>
> **Thinking**: ##这里是一些思考
>
> - 方法不需要对比, 在related work中的noisy-annotation中提及.
> - 第三个concern reply的时候需要解释清楚: 我们所说的noisy examples指的是这一部分examples包含incorrectly labeled examples，但我们不知道到底是哪些出错.
> - 第一条和第二条concern合起来解决,  通过单独提升hard-example learning和单独关注noisy examples learning.
>
> **Exploring OID dataset**: ##写了一些接下来要做的事情
>
> - https://storage.googleapis.com/openimages/web/challenge2019.html
> - Further explore the first place solutions (methods and tricks)





## Tasks

### W1: 2020.1.10 ~ 2020.1.17

第一周主要是基础入门, 做好预先的工作准备 (如果需要更长的时间我会考虑)

- [ ] 如果对**Pytorch**不是很熟悉, 参考Resource collection里补充相关知识
- [ ] [EAD 2020主页](https://ead2020.grand-challenge.org/Home/)阅读, 注册, 了解要做的任务 (detection, segmentation)
- [ ] 收集与EAD2020相关的基础paper, 放到Resource collection. (简要总结文章的**Motivation**)
- [ ] (*) 阅读`./src`里的baseline代码(正在完善), copy到自己的文件夹魔改.
- [ ] (*) 阅读经典的object detection算法和对应Pytorch实现, 见Resource collection.



## QA collection

**说明**:

QA collection这里可以收集一些你们遇到的工程问题 / 关于paper理解的问题以及solution. 请在收集之前思考这个问题是不是一个合适的问题.

例子:

- [x] (Git) `git status` 命令显示的东西太多, 看不过来 (by gyf)

  > (gyf): 用 `git status -uno`, 只看追踪文件的status.

-----



## Resource collection

### Deep learning basic

- https://www.bilibili.com/video/av13260183?from=search&seid=1646745114568230241 这门课用来入门可以的 (机器学习的课程就不列了, 自己能找到)

### Pytorch basic

- https://github.com/chenyuntc/pytorch-book 比较不错的Pytorch入门项目, 基本上看到chapter 6就足够了.

### Git

- https://git-scm.com/doc 当然是参考官网了, 会用commit, pull, fetch, push, merge(分支管理) 

### Papers & Codes

- (Codes) https://github.com/open-mmlab/mmdetection 不太适合魔改, 但是用起来比较方便的object detection框架

- (Paper) one-stage object detection的经典论文 [Focal loss](https://arxiv.org/abs/1708.02002), Pytorch代码见https://github.com/kuangliu/pytorch-retinanet

- (Paper) two-stage经典论文 Faster RCNN, Pytorch代码: https://github.com/jwyang/faster-rcnn.pytorch

  



