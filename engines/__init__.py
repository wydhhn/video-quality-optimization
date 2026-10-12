"""分段引擎。

两种实现，统一接口 run(video_path, ...) -> (segments, meta)：
  · transnetv2   逐帧转场概率 + 阈值
  · omnishotcut  Shot-Query Transformer 直接预测切点
"""
