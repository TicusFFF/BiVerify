# 4-1_BiVerify Agent Prompt Mapping

## Principle

4-1_BiVerify reuses existing branches and prompts.

- Reverse branch is copied from `2-2-1-1_P4-5Py`.
- Forward branch is copied from `3-1_Forward`.
- The only new component is the bidirectional scheduler `S`.

## Reverse Branch

| Agent | Name | Source |
|---|---|---|
| A1 | Reverse Verification Flow Generator | `reverse_progco_infer_main.py::gen_verify_code_dict` |
| A2 | Reverse Verification Executor | `reverse_progco_infer_main.py::execute_verify_code_dict` |
| A3 | Reverse Failure Feedback Summarizer | `reverse_progco_infer_main.py::feedback_dict` |
| A4 | Reverse Verification Flow Refiner | `reverse_progco_infer_main.py::code_refine_dict` |
| A5 | Reverse Response Refiner | `reverse_progco_infer_main.py::refine_dict` |
| A6 | Reverse Answer Difference Controller | scheduler logic |

## Forward Branch

| Agent | Name | Source |
|---|---|---|
| A11 | Forward Verification Flow Generator | `forward_progco_infer_main.py::gen_verify_code_dict` |
| A12 | Forward Verification Executor | `forward_progco_infer_main.py::execute_verify_code_dict` |
| A13 | Forward Failure Feedback Summarizer | `forward_progco_infer_main.py::feedback_dict` |
| A14 | Forward Verification Flow Refiner | `forward_progco_infer_main.py::code_refine_dict` |
| A15 | Forward Response Refiner | `forward_progco_infer_main.py::refine_dict` |
| A16 | Forward Answer Difference Controller | scheduler logic |

## Shared

| Agent | Name | Source |
|---|---|---|
| A8 | Answer Extractor | `extract_answer.py` |
| A17 | ToolFlow Implementer | `math_hard_test_parallel_python_tool.py::branch_math_rewrite_dict` |
| S | BiVerify Scheduler | `progco_infer_main.py::ProgCoAgent.inference` |

## Scheduler Rule

The scheduler maintains one current candidate:

```text
C_t=(q, y, a)
```

The candidate is accepted only when the same `C_t` passes both directions:

```text
R_t=pass and F_t=pass
```

If the candidate changes, both `R_t` and `F_t` are reset to unknown.
