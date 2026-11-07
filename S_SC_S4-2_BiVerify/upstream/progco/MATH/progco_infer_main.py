import os
import sys
sys.path.append(os.environ["PROJECT_PATH"])

import time
import datetime
import random
import importlib.util
import inspect
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
from extract_answer import extract_answer
from utils.process_file import read_file, save_file


# ------------------------------------------------------------------------------------------------------------------------
# 4-1_BiVerify
#
# First principle:
#   Reuse existing branches and prompts.
#   Do not rewrite 2-2-1-1 reverse prompts.
#   Do not rewrite 3-1 forward prompts.
#
# Reverse branch:
#   copied from 2-2-1-1_P4-5Py as reverse_progco_infer_main.py
#
# Forward branch:
#   copied from 3-1_Forward as forward_progco_infer_main.py
#
# New component:
#   S / BiVerify Scheduler implemented by ProgCoAgent.inference
#
# S maintains one current candidate:
#   C_t = (q, y, a)
#
# Accept only when the same C_t passes both directions:
#   R_t = pass and F_t = pass
# ------------------------------------------------------------------------------------------------------------------------


def _load_local_module(file_name, module_name):
    current_dir = Path(__file__).resolve().parent
    file_path = current_dir / file_name
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load copied branches.
R = _load_local_module("reverse_progco_infer_main.py", "biverify_reverse_branch")
F = _load_local_module("forward_progco_infer_main.py", "biverify_forward_branch")


# Compatibility aliases for existing runner scripts.
# PythonToolLLMCodeExecutor currently imports MATH.progco_infer_main as progco
# and expects execute_verify_code_dict to exist.
# We set the default runtime to the forward executor; the scheduler can switch
# to reverse by passing execute_dict explicitly.
execute_verify_code_dict = F.execute_verify_code_dict
gen_verify_code_dict = F.gen_verify_code_dict
feedback_dict = F.feedback_dict
code_refine_dict = F.code_refine_dict
refine_dict = F.refine_dict

try:
    contrast_response_dict = F.contrast_response_dict
except Exception:
    contrast_response_dict = getattr(R, "contrast_response_dict", None)

# Keep class aliases for legacy runner compatibility.
Feebacker = F.Feebacker
VerifyCodeGenerator = F.VerifyCodeGenerator
CodeRefiner = F.CodeRefiner
Refiner = F.Refiner
ContrastResponse = F.ContrastResponse
PythonToolRequiredExecutor = F.PythonToolRequiredExecutor


# ------------------------------------------------------------------------------------------------------------------------
# Logger

class Logger:
    def __init__(self, args):
        self.args = args
        self.logs = []
        self.start_time = self.get_current_time()
        save_model_name = self.args.model
        self.log_path = f"MATH/logs/{save_model_name}/infered/{self.args.experiment_prefix}_{self.start_time}.json"

    def get_current_time(self):
        current_time = datetime.datetime.now()
        return current_time.strftime("%Y-%m-%d-%H:%M")

    def init_sample(self, sample_id, query):
        self.logs.append({"id": sample_id, "query": query, "turns": []})

    def update_turn(
        self,
        cur_turn,
        verify_code,
        response,
        result,
        execute_result,
        execute_content,
        feedback,
        reflection,
        code_feedback,
        revised_response=None,
        revised_result=None,
        a16_decision=None,
        next_verify_source=None,
        direction=None,
        candidate_version=None,
        R_status=None,
        F_status=None,
        scheduler_next=None,
    ):
        turn_log = {
            "cur_turn": cur_turn,
            "verify_code": verify_code,
            "response": response,
            "result": result,
            "execute_result": execute_result,
            "execute_content": execute_content,
            "feedback": feedback,
            "reflection": reflection,
            "code_feedback": code_feedback,
        }

        if revised_response is not None:
            turn_log["revised_response"] = revised_response
        if revised_result is not None:
            turn_log["revised_result"] = revised_result
        if a16_decision is not None:
            turn_log["a16_decision"] = a16_decision
        if next_verify_source is not None:
            turn_log["next_verify_source"] = next_verify_source

        if direction is not None:
            turn_log["direction"] = direction
        if candidate_version is not None:
            turn_log["candidate_version"] = candidate_version
        if R_status is not None:
            turn_log["R_status"] = R_status
        if F_status is not None:
            turn_log["F_status"] = F_status
        if scheduler_next is not None:
            turn_log["scheduler_next"] = scheduler_next

        self.logs[-1]["turns"].append(turn_log)

    def update_final_response(self, response, result):
        self.logs[-1]["final_response"] = response
        self.logs[-1]["final_result"] = result

    def update_insights(self, insights):
        self.logs[-1]["insights"] = insights

    def save_log(self):
        log_data = {
            "start_time": self.start_time,
            "end_time": self.get_current_time(),
            "args": self.args.__dict__,
            "result": None,
            "logs": self.logs,
        }
        save_file(self.log_path, log_data)


# ------------------------------------------------------------------------------------------------------------------------
# Helper functions

def _call_feedbacker(feedbacker, **kwargs):
    """
    Compatibility helper.
    2-2-1-1 and 3-1 may have slightly different Feebacker signatures.
    This function only passes supported keyword arguments.
    """
    sig = inspect.signature(feedbacker.__call__)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return feedbacker(**filtered)


def _call_code_refiner(code_refiner, **kwargs):
    """
    Compatibility helper.
    Some 3-1 versions have CodeRefiner.refine(response, result, verify_code, verify_process);
    older versions have CodeRefiner.refine(response, verify_code, verify_process).
    """
    sig = inspect.signature(code_refiner.refine)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return code_refiner.refine(**filtered)


def _safe_extract_answer(query, response, model):
    if response is None or not str(response).strip():
        return ""
    return extract_answer(query, response, model=model)


# ------------------------------------------------------------------------------------------------------------------------
# ProgCoAgent with BiVerify Scheduler

class ProgCoAgent:
    def __init__(self, args):
        self.dataset = args.dataset
        self.model = args.model
        self.max_cur_turn = args.max_cur_turn
        self.temperature = args.temperature

        # Disabled placeholder: the real executor is injected by
        # repro/scripts/math_hard_test_parallel_python_tool.py as
        # PythonToolLLMCodeExecutor.
        self.llm_code_executor = PythonToolRequiredExecutor(
            model=self.model,
            temperature=self.temperature,
        )

        self.logger = Logger(args)

        # Reuse the original initial response file convention.
        self.first_response_data = read_file(f"MATH/logs/{self.model}/all_random_500_first_response.jsonl")
        self.first_response_data = {item["id"]: item for item in self.first_response_data}

    def _init_agents(self, query):
        tool_runner = getattr(self.llm_code_executor, "run_refiner_with_tool_flow_loop", None)

        # Reverse branch copied from 2-2-1-1.
        self.R_verify_generator = R.VerifyCodeGenerator(
            model=self.model,
            temperature=self.temperature,
        )
        self.R_feedbacker = R.Feebacker(
            model=self.model,
            temperature=self.temperature,
        )
        self.R_response_refiner = R.Refiner(
            model=self.model,
            query=query,
            tool_runner=tool_runner,
        )
        self.R_code_refiner = R.CodeRefiner(
            model=self.model,
            query=query,
            temperatue=self.temperature,
            tool_runner=tool_runner,
        )

        # Forward branch copied from 3-1.
        self.F_verify_generator = F.VerifyCodeGenerator(
            model=self.model,
            temperature=self.temperature,
        )
        self.F_feedbacker = F.Feebacker(
            model=self.model,
            temperature=self.temperature,
        )
        self.F_response_refiner = F.Refiner(
            model=self.model,
            query=query,
            tool_runner=tool_runner,
        )
        self.F_code_refiner = F.CodeRefiner(
            model=self.model,
            query=query,
            temperatue=self.temperature,
            tool_runner=tool_runner,
        )

        # Kept for compatibility with existing logging; 4-1 scheduler does not rely on it.
        self.contraster = F.ContrastResponse(
            model=self.model,
            query=query,
            temperature=self.temperature,
        )

    def inference(self, sample):
        query = sample["query"]
        sample_id = sample["id"]

        self._init_agents(query)
        self.logger.init_sample(sample_id=sample_id, query=query)

        # S0: Initialize C_0=(q, y_0, a_0)
        current_response = self.first_response_data[sample_id]["response"]
        current_result = self.first_response_data[sample_id]["pred"]

        candidate_version = 0
        R_status = "unknown"
        F_status = "unknown"
        next_direction = "R"

        reverse_verify_code = None
        forward_verify_code = None

        final_response = current_response
        final_result = current_result

        cur_turn = 0

        while cur_turn < self.max_cur_turn + 1:
            print(
                f"cur_turn:{cur_turn}, candidate_version:{candidate_version}, "
                f"next:{next_direction}, R:{R_status}, F:{F_status}"
            )

            # S2 final accept: same C_t passes both.
            if R_status == "pass" and F_status == "pass":
                final_response = current_response
                final_result = current_result
                break

            # ----------------------------------------------------------------------------------------------------------------
            # S1-R: Reverse interface
            if next_direction == "R":
                verify_response = current_response
                verify_result = current_result

                if reverse_verify_code is None:
                    reverse_verify_code = self.R_verify_generator.gen_verify_code(query=query)

                used_reverse_verify_code = reverse_verify_code

                print("start reverse verification")
                execute_result, execute_content = self.llm_code_executor.execute(
                    query=query,
                    response=verify_response,
                    result=verify_result,
                    validate_response_fuc=used_reverse_verify_code,
                    execute_dict=R.execute_verify_code_dict,
                    caller="A2",
                )

                # S2-R: reverse pass
                if execute_result:
                    R_status = "pass"
                    feedback = "reverse verify passed"
                    scheduler_next = "Accept" if F_status == "pass" else "F"

                    self.logger.update_turn(
                        cur_turn,
                        used_reverse_verify_code,
                        verify_response,
                        verify_result,
                        execute_result,
                        execute_content,
                        feedback,
                        reflection="",
                        code_feedback="",
                        direction="R",
                        candidate_version=candidate_version,
                        R_status=R_status,
                        F_status=F_status,
                        scheduler_next=scheduler_next,
                    )

                    next_direction = "ACCEPT" if F_status == "pass" else "F"
                    cur_turn += 1
                    continue

                # S3-R: reverse fail
                R_status = "fail"
                print("reverse failed; start reverse feedback repair")

                reverse_feedback = _call_feedbacker(
                    self.R_feedbacker,
                    query=query,
                    response=verify_response,
                    execute_code=used_reverse_verify_code,
                    execute_result=execute_result,
                    execute_content=execute_content,
                )

                reverse_execute_content = execute_content

                reflection, revised_response = self.R_response_refiner.refine(
                    old_response=verify_response,
                    feedback=reverse_feedback,
                )

                if not revised_response or not str(revised_response).strip():
                    revised_response = verify_response

                revised_result = _safe_extract_answer(query, revised_response, model=self.model)

                # A6: answer difference controller for reverse branch.
                if revised_result != verify_result:
                    # S4: candidate update.
                    current_response = revised_response
                    current_result = revised_result
                    candidate_version += 1
                    R_status = "unknown"
                    F_status = "unknown"
                    reverse_verify_code = None
                    forward_verify_code = None

                    # Confirm by the same direction first.
                    next_direction = "R"
                    scheduler_next = "R"
                    answer_decision = "different"
                    code_feedback = ""

                else:
                    # S5-R: candidate unchanged; refine reverse verification flow.
                    code_feedback, reverse_verify_code = _call_code_refiner(
                        self.R_code_refiner,
                        response=verify_response,
                        verify_code=used_reverse_verify_code,
                        verify_process=reverse_execute_content,
                    )
                    next_direction = "R"
                    scheduler_next = "R"
                    answer_decision = "same"

                execute_content = getattr(self.llm_code_executor, "execute_content", execute_content)

                self.logger.update_turn(
                    cur_turn,
                    used_reverse_verify_code,
                    verify_response,
                    verify_result,
                    execute_result,
                    execute_content,
                    reverse_feedback,
                    reflection,
                    code_feedback,
                    revised_response=revised_response,
                    revised_result=revised_result,
                    a16_decision=answer_decision,
                    next_verify_source=scheduler_next,
                    direction="R",
                    candidate_version=candidate_version,
                    R_status=R_status,
                    F_status=F_status,
                    scheduler_next=scheduler_next,
                )

                cur_turn += 1
                continue

            # ----------------------------------------------------------------------------------------------------------------
            # S1-F: Forward interface
            if next_direction == "F":
                verify_response = current_response
                verify_result = current_result

                if forward_verify_code is None:
                    forward_verify_code = self.F_verify_generator.gen_verify_code(
                        query=query,
                        response=verify_response,
                        result=verify_result,
                    )

                used_forward_verify_code = forward_verify_code

                print("start forward verification")
                execute_result, execute_content = self.llm_code_executor.execute(
                    query=query,
                    response=verify_response,
                    result=verify_result,
                    validate_response_fuc=used_forward_verify_code,
                    execute_dict=F.execute_verify_code_dict,
                    caller="A12",
                )

                # S2-F: forward pass
                if execute_result:
                    F_status = "pass"
                    feedback = "forward verify passed"
                    scheduler_next = "Accept" if R_status == "pass" else "R"

                    self.logger.update_turn(
                        cur_turn,
                        used_forward_verify_code,
                        verify_response,
                        verify_result,
                        execute_result,
                        execute_content,
                        feedback,
                        reflection="",
                        code_feedback="",
                        direction="F",
                        candidate_version=candidate_version,
                        R_status=R_status,
                        F_status=F_status,
                        scheduler_next=scheduler_next,
                    )

                    next_direction = "ACCEPT" if R_status == "pass" else "R"
                    cur_turn += 1
                    continue

                # S3-F: forward fail
                F_status = "fail"
                print("forward failed; start forward feedback repair")

                forward_feedback = _call_feedbacker(
                    self.F_feedbacker,
                    query=query,
                    response=verify_response,
                    result=verify_result,
                    execute_code=used_forward_verify_code,
                    execute_result=execute_result,
                    execute_content=execute_content,
                )

                forward_execute_content = execute_content

                reflection, revised_response = self.F_response_refiner.refine(
                    old_response=verify_response,
                    feedback=forward_feedback,
                )

                if not revised_response or not str(revised_response).strip():
                    revised_response = verify_response

                revised_result = _safe_extract_answer(query, revised_response, model=self.model)

                # A16: answer difference controller for forward branch.
                if revised_result != verify_result:
                    # S4: candidate update.
                    current_response = revised_response
                    current_result = revised_result
                    candidate_version += 1
                    R_status = "unknown"
                    F_status = "unknown"
                    reverse_verify_code = None
                    forward_verify_code = None

                    # Confirm by the same direction first.
                    next_direction = "F"
                    scheduler_next = "F"
                    answer_decision = "different"
                    code_feedback = ""

                else:
                    # S5-F: answer unchanged.
                    # Follow 3-1 habit: adopt revised y' if it exists, then refine V_f.
                    text_changed = str(revised_response).strip() != str(verify_response).strip()
                    current_response = revised_response
                    current_result = verify_result

                    code_feedback, forward_verify_code = _call_code_refiner(
                        self.F_code_refiner,
                        response=revised_response,
                        result=revised_result,
                        verify_code=used_forward_verify_code,
                        verify_process=forward_execute_content,
                    )

                    if text_changed:
                        candidate_version += 1
                        # Since y changed, previous reverse state no longer safely applies.
                        R_status = "unknown"
                        F_status = "unknown"
                        reverse_verify_code = None

                    next_direction = "F"
                    scheduler_next = "F"
                    answer_decision = "same"

                execute_content = getattr(self.llm_code_executor, "execute_content", execute_content)

                self.logger.update_turn(
                    cur_turn,
                    used_forward_verify_code,
                    verify_response,
                    verify_result,
                    execute_result,
                    execute_content,
                    forward_feedback,
                    reflection,
                    code_feedback,
                    revised_response=revised_response,
                    revised_result=revised_result,
                    a16_decision=answer_decision,
                    next_verify_source=scheduler_next,
                    direction="F",
                    candidate_version=candidate_version,
                    R_status=R_status,
                    F_status=F_status,
                    scheduler_next=scheduler_next,
                )

                cur_turn += 1
                continue

            if next_direction == "ACCEPT":
                final_response = current_response
                final_result = current_result
                break

            # Safety fallback.
            next_direction = "R"
            cur_turn += 1

        # Use the latest current candidate as final output.
        # If the loop stops by turn limit immediately after a candidate update
        # or a second-side pass, this avoids falling back to a stale candidate.
        final_response = current_response
        final_result = current_result

        self.logger.update_final_response(final_response, final_result)
        try:
            self.logger.update_insights(self.contraster.memory)
        except Exception:
            self.logger.update_insights([])


# ------------------------------------------------------------------------------------------------------------------------
# Dataset

def read_dataset(start, end, shuffle=False):
    def format_sample(id, query, sample):
        return {"id": id, "query": query, "sample": sample}

    data = read_file("MATH/data/all_random_500.jsonl")
    data = [format_sample(id=dp["id"], query=dp["query"], sample=dp) for dp in data]
    if shuffle:
        random.seed(0)
        random.shuffle(data)
    if end == -1:
        return data[start:]
    return data[start:end]


# ------------------------------------------------------------------------------------------------------------------------
# Main Fuc

def main_fuc(args):
    agent = ProgCoAgent(args)
    data = read_dataset(start=args.start, end=args.end, shuffle=False)
    print("dataset size: ", len(data))
    for i, sample in enumerate(tqdm(data)):
        print("-" * 10)
        print("id:", sample["id"])
        agent.inference(sample)
        if i % 10 == 0:
            agent.logger.save_log()
    agent.logger.save_log()


# ------------------------------------------------------------------------------------------------------------------------
# Param

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="qdd_GPT4o")
    parser.add_argument("--max_cur_turn", type=int, default=3, help="Maximum self-correction rounds (default: 3)")
    parser.add_argument("--start", type=int, default=0, help="Starting index of the evaluation dataset slice (default: 0)")
    parser.add_argument("--end", type=int, default=-1, help="Ending index of the evaluation dataset slice (default: -1 for entire dataset)")
    parser.add_argument("--dataset", type=str, default="MATH")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    args.experiment_prefix = f"biverify_start-{args.start}_end-{args.end}"
    print(vars(args))
    return args


if __name__ == "__main__":
    args = parse_args()
    main_fuc(args)
