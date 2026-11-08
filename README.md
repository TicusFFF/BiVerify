# BiVerify: Closing the Loop Between Backward and Forward Verification for Mathematical Self-Correction

English | [简体中文](README_zh-CN.md)

A mathematical answer may satisfy the given conditions while its reasoning is incorrect or some required solutions are missing. BiVerify verifies the same solution in two directions: it first checks the answer against the original conditions, then follows the solution steps to check whether the key reasoning is correct and complete.

## Framework

[![BiVerify bidirectional verification framework](assets/framework.png)](assets/framework.png)

The left side shows backward verification, and the right side shows forward verification. Both directions can use mathematical tools. When one direction passes, the scheduler switches to the other. When verification fails, the system uses the feedback to revise the solution or adjust the verification steps, then checks again. The scheduler accepts a solution only when the same solution passes both directions. When the verification limit is reached, the system returns the current solution.

## Main Results

[![Table 1 from the paper: Accuracy of each method on five mathematical reasoning datasets, reported for seven model configurations and their average.](assets/main-results.png)](assets/main-results.png)

## Installation and Configuration

Run the following command from the repository root:

```bash
python -m pip install -r requirements.txt
```

The example uses the official DeepSeek API. Set `DEEPSEEK_API_KEY`. In PowerShell, use `$env:DEEPSEEK_API_KEY = "<your-key>"`; in Bash, use `export DEEPSEEK_API_KEY="<your-key>"`.

## Quick Start

The included `demo` contains two mathematics problems. The following two steps call the model API.

### 1. Generate Shared Initial Responses

```bash
python 0-1_initial_response/generate_initial_responses.py --dataset demo --output-dir runs/initial
```

Initial responses are saved locally to `runs/initial/deepseek-flash_demo_first_response.jsonl`, and their scores are saved to `demo.initial-eval.json` in the same directory. When comparing methods, you can share this batch of initial responses. In step 2, set `--initial-dir` to the directory containing the initial responses and model version records.

### 2. Run BiVerify and Evaluate

```bash
python 0-3_all_model_test/run_plan_S_SC/S_SC_S4-2_BiVerify/run_biverify.py --dataset demo --initial-dir runs/initial --run
```

The program reads the initial responses, starts the local Python tool, and performs bidirectional verification and correction. By default, each run is saved to `runs/<UTC-timestamp>/`. The file `model_outputs/demo.json` contains per-problem reasoning and final responses, and `evaluation.json` contains the results from the shared evaluator. Reference answers are used only for evaluation. The Python tool executes model-generated code on your machine, so use a dedicated environment with restricted permissions.

To run HMMT, first prepare the dataset as described in `0-0_datasets/README.md`, then replace `--dataset demo` with `--dataset hmmt_feb_2025` in both commands.

Parts of the verification and correction code are adapted from ProgCo. See [attribution](THIRD_PARTY.md).

## Citation

arXiv link to be added.

```bibtex
@misc{zou2026biverify,
  title = {{BiVerify}: Closing the Loop Between Backward and Forward Verification for Mathematical Self-Correction},
  author = {Zou, Rui and Zhao, Xin and Wei, Mengqi and Wen, Ji-Rong},
  year = {2026}
}
```
