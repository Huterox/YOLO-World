"""
    核心思想还是说做多尺度融合，在融合的时候对Image的feature vector 和 Text的feature vector 进行处理。
    这里主要注意以下四个点：
        1. Image的feature怎么提取
        2. Text的feature怎么提取
        3. 两个feature之间怎么融合
        4. Loss怎么计算，怎么对text&img进行评价
    其他的点，似乎比较中规中矩，没有太多的特别的点。

    一、对应image的feature提取，主要还是基于v8的blacknet进行处理
        但是这里注意的是，这里输出的将是三个尺度特性向量，对应三种尺寸维度
    二、对应text的提取，这部分的提取就比较灵性，直接基于Clip2预训练好的模型，把输出层拆掉，
        之后得到对齐的三个特征向量，这里是clip去适配到image的feature
    三、这就是我觉得很有意思的点了，这里面用来 残差+累加Cov1卷积+多头注意力机制
        这个也是论文当中提到的 Vision-Language PAN 
        然后有意思的是这里卷了两次，然后中间也有个多头注意力做了个融合，所以这块的融合就有点意思
        之后的话，我们是得到了五个输出的特征向量，之后这五个搞在一起。
    四、最后就是我们Loss的计算，这里的话完最后的融合的话，我们输出一个是文本描述的embed还有一个就是
        我们V8的输出，也就是正常标定。那么关于分类就是计算出预测的embed和我们一开始得到的embed通过
        clip这个在代码当中叫做Guide。计算Guide的相似度，做一个sort，然后任务它是对应的分类。最终完成
        zero-short在视觉的识别领域。
        This is called "YOLO-World: Real-Time Open-Vocabulary Object Detection" 这也意味着，LLM可以更好驱动
        视觉，更好的去控制设备，当然更加优秀的Agent机制也是可以更好地控制设备的，坐等开源~
"""
