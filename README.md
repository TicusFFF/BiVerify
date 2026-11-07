# BiVerify：用于数学自纠错的双向验证

**BiVerify: Closing the Loop Between Backward and Forward Verification for Mathematical Self-Correction**  
Rui Zou · Xin Zhao · Mengqi Wei · Ji-Rong Wen

数学答案即使满足题目给出的条件，推导过程也可能有错，或者没有找全题目要求的解。BiVerify 从两个方向验证同一份解答：先从答案出发，检查它是否满足原题条件，再按解题顺序检查关键推理是否正确、是否完整。

## 方法框架

[![BiVerify 双向验证框架](assets/framework.png)](assets/framework.png)

图左是反向验证，图右是正向验证。两个方向都可以调用数学工具。一个方向通过后，调度器切换到另一个方向；验证失败时，系统根据反馈修正解答或调整验证步骤，再次检查。只有同一份解答在两个方向上都通过，调度器才接受它。达到验证次数上限时，系统返回当前解答。

## 主要结果

[![论文表 1：各方法在五个数学题集上的准确率，按七种模型设置及其均值列出。](assets/main-results.png)](assets/main-results.png)

## 安装与环境配置

在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

设置 [`DEEPSEEK_API_KEY`](.env.example)。在 PowerShell 中使用 `$env:DEEPSEEK_API_KEY = "<your-key>"`，在 Bash 中使用 `export DEEPSEEK_API_KEY="<your-key>"`。示例的初答、验证、纠错和评分均调用 [DeepSeek 官方 API](https://api-docs.deepseek.com/)，请求 `deepseek-flash` 的非思考模式，并记录请求与返回的模型名。上表是论文中的实验结果，其中 DS-Flash 使用的是 `deepseek-v4-flash`。

## 快速开始

[`demo`](0-0_datasets/hard_test_raw/demo/test_top2_filt2.json) 包含两道数学题。和其他题集一样，通过 `--dataset` 指定。

### 1. 生成公共初答

```bash
python 0-1_initial_response/generate_initial_responses.py --dataset demo --output-dir runs/initial
```

这一步也会评价初答，将结果写入 `runs/initial/demo.initial-eval.json`。比较不同方法时，同一模型、同一道题共用一份初答。已有匹配的初答和版本记录时，可将下一步的 `--initial-dir` 指向它所在的目录。

### 2. 运行 BiVerify 并评分

```bash
python 0-3_all_model_test/run_plan_S_SC/S_SC_S4-2_BiVerify/run_biverify.py --dataset demo --initial-dir runs/initial --run
```

入口读取初答，启动本地 Python 工具，执行双向验证与纠错，再用第 1 步的评分标准评价最终回答。运行目录中的 `model_outputs/` 保存逐题验证过程和最终回答，`evaluation.json` 保存最终评分。参考答案仅用于评分。Python 工具会在本机执行模型生成的代码，请使用专用、低权限的环境。

要改跑 HMMT，将两条命令中的 `--dataset demo` 换成 `--dataset hmmt_feb_2025`。其他题集的来源和准备方法见[数据说明](0-0_datasets/README.md)。

## 引用

Rui Zou, Xin Zhao, Mengqi Wei, and Ji-Rong Wen. *BiVerify: Closing the Loop Between Backward and Forward Verification for Mathematical Self-Correction.*

[第三方代码、数据和许可证](THIRD_PARTY.md)
