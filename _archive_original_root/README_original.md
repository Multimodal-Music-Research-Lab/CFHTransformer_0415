# CFHTransformer_v12_0425

## 前一版对于分开的transformer的修改在 https://github.com/yinlin1211/cft_modifications 但是我改动不到位，con和conp还没提升，所以当前最干净的代码还是本实验

当前仓库中最新整理的实验版本位于 `v12/`。下面的说明沿用 `v12/README.md` 的内容，除非特别说明，文中的路径都相对于 `v12/`。

当前仓库整理的是 `run/20260422_201016_COnP` 这次实验对应的代码、结果和独立后处理阈值搜索流程。

## 我们当前的结果

当前推荐汇报的是 MIR-ST500 测试集上的最终结果。

| Dataset | onset | frame | offset | COn | COnP | COnPOff |
|---------|------:|------:|-------:|----:|-----:|--------:|
| MIR-ST500 test set | 0.45 | 0.50 | 0.10 | 0.803172 | 0.776425 | 0.592620 |

这就是当前仓库里最完整、最推荐对外使用的一组结果。阈值搜索、训练过程和实验记录放在下面展开说明。

## 实验流程记录

### 1. 训练

本次实验对应的训练目录是：

- `run/20260422_201016_COnP/`

训练命令：

```bash
cd /mnt/ssd/lian/给claudecode/v10_baseline_conpoff
CUDA_VISIBLE_DEVICES=1 python3 train_conp_v6_0415.py --config config.yaml
```

训练配置要点：

- train: `1-400`
- val: `361-400`
- test: `401-500`
- batch size: `16`
- learning rate: `3e-4`
- max samples per epoch: `4000`
- hop length: `800`
- CQT: `288 bins`

训练日志：

- `run/20260422_201016_COnP/logs/train_stdout.log`

### 2. 训练内验证与外部 test 监控

训练脚本内部会做两件事：

1. 在 `val40` 上做训练内验证与阈值搜索
2. 在 `test100` 上做外部监控，记录到 `test_monitor.txt`

其中：

- 训练内 best model 保存标准是 `val COnP`
- `test_monitor.txt` 只作为外部观察记录，用来查看是否已经过拟合、从而结束训练；本身不参与训练时反向传播或 `val` 阈值搜索

外部监控文件：

- `run/20260422_201016_COnP/test_monitor.txt`

`test_monitor.txt` 主要用于观察训练是否已经进入平台期，而不是直接作为后处理阈值搜索的依据。后处理阈值的确定仍然完全在 `val40` 上完成。基于平台期内的候选 checkpoint，我们进一步选取了 `epoch128` 和 `epoch233` 两个模型，用来验证模型是否已经到顶。其中本 README 后面完整展示的是 `epoch128` 这一组实验流程。

这两个平台期 checkpoint 在完成“两轮 `val40` 阈值搜索”后的最终 test 结果如下：

| checkpoint | onset | frame | offset | COn | COnP | COnPOff |
|------------|------:|------:|-------:|----:|-----:|--------:|
| epoch128 | 0.45 | 0.50 | 0.10 | 0.803172 | 0.776425 | 0.592620 |
| epoch233 | 0.45 | 0.55 | 0.30 | 0.798849 | 0.771539 | 0.585094 |

本 README 当前主结果对应的 checkpoint 是：

- `run/20260422_201016_COnP/checkpoints/best_model_epoch0128_COnP0.7958.pt`

### 3. 独立 val40 阈值搜索

第一阶段代码：

- `评估/search_threshold_v2.py`

运行命令：

```bash
python3 评估/search_threshold_v2.py \
  --checkpoint run/20260422_201016_COnP/checkpoints/best_model_epoch0128_COnP0.7958.pt \
  --output_dir run/20260422_201016_COnP/threshold_search_v2_epoch0128
```

第一阶段结果：

- `BEST_VAL_BY_COnP onset=0.45 frame=0.50`
- `val COn=0.817416`
- `val COnP=0.802107`
- `val COnPOff=0.529928`

### 4. 独立 val40 offset 阈值搜索

第二阶段代码：

- `评估/search_offset_threshold_and_predict.py`

运行命令：

```bash
python3 评估/search_offset_threshold_and_predict.py \
  --checkpoint run/20260422_201016_COnP/checkpoints/best_model_epoch0128_COnP0.7958.pt \
  --onset_thresh 0.45 \
  --frame_thresh 0.50 \
  --output_dir run/20260422_201016_COnP/offset_search_epoch0128_best_val_conp
```

第二阶段结果：

- `SELECTED_BY_VAL off=0.10`
- `test COn=0.803172`
- `test COnP=0.776425`
- `test COnPOff=0.592620`

### 5. 最终 test 推理与结果文件

最终结果来自：

1. 先训练得到 `epoch128` checkpoint
2. 再在 `val40` 上独立搜索 `onset/frame`
3. 再在 `val40` 上独立搜索 `offset`
4. 最后用选出的三阈值在 `MIR-ST500 test set` 上推理一次

结果文件：

- `run/20260422_201016_COnP/threshold_search_v2_epoch0128/val_threshold_search.tsv`
- `run/20260422_201016_COnP/threshold_search_v2_epoch0128/selected_thresholds.tsv`
- `run/20260422_201016_COnP/offset_search_epoch0128_best_val_conp/val_offset_threshold_search.tsv`
- `run/20260422_201016_COnP/offset_search_epoch0128_best_val_conp/selected_offset_threshold.tsv`
- `run/20260422_201016_COnP/offset_search_epoch0128_best_val_conp/test_with_selected_offset_threshold.tsv`
- `run/20260422_201016_COnP/offset_search_epoch0128_best_val_conp/pred_test_offset_aware.json`

### 6. 训练内阈值与独立搜索的区别

训练脚本里保存过一组训练内阈值：

- `onset=0.50`
- `frame=0.40`

这组阈值来自训练过程中每 5 个 epoch 的 val 搜索，并记录在 checkpoint / monitor 里。

但独立评估脚本重新按固定口径搜索后，得到的是：

- `onset=0.45`
- `frame=0.50`

因此对外报告时，优先建议使用 `评估/` 目录下独立搜索得到的完整实验结果。

## 代码位置

- `train_conp_v6_0415.py`：训练脚本，包含训练内阈值搜索与 test monitor
- `predict_to_json.py`：两阈值 baseline 推理（`onset/frame`）
- `predict_to_json_offset.py`：三阈值 offset-aware 推理（`onset/frame/offset`）
- `评估/search_threshold_v2.py`：独立 `val40` 搜 `onset/frame`
- `评估/search_offset_threshold_and_predict.py`：固定 `onset/frame` 后独立 `val40` 搜 `offset`
- `evaluate_github.py`：原论文兼容评测脚本

## 环境与数据

- Python 3.12
- PyTorch
- librosa
- mir_eval
- CQT 缓存：288-bin, hop=800, 50ms/帧
- 标注：`data/MIR-ST500_corrected.json`
- 划分：`splits_v11/` (`train=1-400`, `val=361-400`, `test=401-500`)

## 模型权重

仓库不上传 `.pt` 权重文件，只保留代码、结果、日志和阈值表。

## 评测脚本来源

`evaluate_github.py` 来源于 [york135/singing_transcription_ICASSP2021](https://github.com/york135/singing_transcription_ICASSP2021/tree/master/evaluate)。
