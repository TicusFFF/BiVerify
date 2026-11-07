import os
import sys
sys.path.append(os.environ["PROJECT_PATH"])
import re
import time
import datetime
import random
from tqdm import tqdm
from copy import deepcopy
from argparse import ArgumentParser
from extract_answer import extract_answer
from utils.call_llm import call_llm
from utils.process_file import read_file,save_file

# Agent / Prompt numbering for 3-1_Forward
#
# A11 / Agent11: Forward Verification Flow Generator
#     gen_verify_code_dict
#
# A12 / Agent12: Forward Verification Flow Executor
#     execute_verify_code_dict
#     + PythonToolLLMCodeExecutor.__init__ runtime prompt
#
# A13 / Agent13: Forward Failure Feedback Summarizer
#     feedback_dict
#
# A14 / Agent14: Forward Verification Flow Refiner with ToolFlow loop
#     code_refine_dict
#
# A15 / Agent15: Forward-Feedback Response Refiner with ToolFlow loop
#     refine_dict
#
# A16 / Agent16: Answer Difference Controller
#     implemented in ProgCoAgent.inference
#     It only compares a and a'.
#     If a' != a, update y and a, then regenerate V_f through A11.
#     If a' == a, update y to y' and refine the current V_f through A14.
#
# A17 / Agent17: ToolFlow Implementer
#     branch_math_rewrite_dict
#
# A8 / Agent8: Answer Extractor
#     extract_prompt

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

        # These fields are mainly for 3-1_Forward analysis.
        # They record the A15 output and the A16 routing decision.
        if revised_response is not None:
            turn_log["revised_response"] = revised_response
        if revised_result is not None:
            turn_log["revised_result"] = revised_result
        if a16_decision is not None:
            turn_log["a16_decision"] = a16_decision
        if next_verify_source is not None:
            turn_log["next_verify_source"] = next_verify_source

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
# ForwardVerifyFlowGenerator

# A11 / Agent11: Forward Verification Flow Generator
# Generate a forward mathematical verification flow for the current solution response.
gen_verify_code_dict = {
    "system_prompt": r"""You are an expert in mathematical verification. Given a problem, a current solution response, and its extracted final answer, generate a forward mathematical verification flow for that response.

[Forward Verification Introduction]
A forward mathematical verification flow starts from the original problem conditions and the current solution response. It checks whether the current solution can be derived correctly and completely from the problem statement.

The flow should inspect the current solution response, identify the key mathematical claims, and verify those claims from the original problem conditions. It should also verify that the extracted final answer matches the complete target object required by the problem.

[Requirements]
1. The verification flow should verify the current solution response in the forward direction.
2. The flow should take the current answer as its formal input, while using the current solution response to identify key intermediate claims that need verification.
3. After generating the verification flow name, first write the forward verification approach as comments, then generate the verification flow.
4. Keep a mathematical reasoning style. Prefer mathematical objects such as expressions, equations, sets, predicates, functions, recurrences, counts, sums, extrema, and solution sets.
5. Use mathematical-tool-level operations when appropriate: simplify, expand, factor, solve, count, sum, max, min, argmax, argmin, recurrence evaluation, DP state, set construction, and completeness checking.
6. Use natural mathematical definitions instead of implementation-level details.
7. Represent the full target required by the problem, especially for all-solutions, count, maximum, minimum, existence, uniqueness, probability, and infinitely-many problems.
8. If the solution response contains key intermediate mathematical claims, include checks for those claims when they are important for the correctness of the solution.
9. Do not over-fragment the solution into trivial steps. Verify only key claims and the final target object.
10. Define the mathematical object to be computed before asking for a result.
11. The flow may use mathematical notation and natural-language definitions; Python implementation is handled later by the tool-flow implementer.
12. Use ASCII mathematical notation for robustness:
    - x in S
    - CartesianProduct(A, B)
    - expr mod n = r
    - count(V)
    - max(V), min(V), sum(V)
    - V is nonempty
13. Output only the verification flow.""",

    "input_case_1": r"""Generate a forward verification flow for the following current solution.

[Problem]
At the candy store, a chocolate bar costs $c$ dollars and a vanilla bar costs $2$ dollars more than a chocolate bar. Jamie buys a chocolate bar and three vanilla bars, and Kevin buys five chocolate bars. How much money, in total, do Jamie and Kevin spend at the candy store in terms of $c$?

[Current Solution Response]
The vanilla bar costs c+2. Jamie spends c + 3(c+2) = 3c + 6 dollars. Kevin spends 5c dollars. Therefore, the total is 3c + 6 + 5c = 8c + 6.

[Current Answer]
8c + 6""",

    "output_case_1": r"""forward_verify_total_spent(total_spent):
    # Forward verification approach:
    # 1. Start from the original problem conditions and the current solution response.
    # 2. Define the vanilla-bar cost from the original condition.
    # 3. Check the current response's key intermediate claim for Jamie's cost.
    # 4. Check Kevin's cost from the original condition.
    # 5. Compute the full target total from the original conditions.
    # 6. Verify whether the current final answer equals the full target total.

    ans := total_spent

    v(c) := c + 2

    # Key intermediate claim in the current response:
    J_claim(c) := 3*c + 6
    J_true(c) := c + 3*v(c)
    Check simplify(J_true(c) - J_claim(c)) = 0.

    K_true(c) := 5*c

    T(c) := J_true(c) + K_true(c)

    D(c) := simplify(T(c) - ans)

    Forward verification passes iff both:
    1. simplify(J_true(c) - J_claim(c)) = 0 identically;
    2. D(c) = 0 identically.""",

    "input_case_2": r"""Generate a forward verification flow for the following current solution.

[Problem]
Solve $x^2 = 4$ over the real numbers.

[Current Solution Response]
Solving $x^2=4$ gives $x=2$. Therefore, the answer is $x=2$.

[Current Answer]
x = 2""",

    "output_case_2": r"""forward_verify_real_solution_set(answer):
    # Forward verification approach:
    # 1. Start from the original equation and the stated domain, the real numbers.
    # 2. Compute the complete real solution set of x^2 = 4.
    # 3. Interpret the current answer as the candidate solution set.
    # 4. Verify whether the candidate solution set equals the complete real solution set.
    # 5. This checks completeness, not only whether a listed value satisfies the equation.

    ans := answer

    S_true := solve(x^2 = 4 over real numbers)

    S_ans := solution set represented by ans

    Forward verification passes iff S_ans = S_true.""",

    "input_case_3": r"""Generate a forward verification flow for the following current solution.

[Problem]
The four-digit number $25AB$ is divisible by nine, with $A$ the tens digit and $B$ the units digit. How many different such four-digit numbers could $25AB$ represent?

[Current Solution Response]
For 25AB, the digit sum is 2+5+A+B = 7+A+B. We need A+B ≡ 2 mod 9. Each A from 0 to 9 gives one B, so there are 10 possibilities.

[Current Answer]
10""",

    "output_case_3": r"""forward_verify_divisible_by_nine_count(count):
    # Forward verification approach:
    # 1. Start from the full digit domain for A and B.
    # 2. Define the complete set of digit pairs that make 25AB divisible by 9.
    # 3. Compute the count of that complete set.
    # 4. Verify whether the current answer equals the complete count.
    # 5. This checks completeness of the count, not only the modular rule.

    ans := count

    D := {0,1,2,3,4,5,6,7,8,9}

    V := {(A,B) in CartesianProduct(D, D) :
        (2 + 5 + A + B) mod 9 = 0
    }

    N := count(V)

    Forward verification passes iff ans = N.""",

    "input_template": r"""Generate a forward verification flow for the following current solution.

[Problem]
{query}

[Current Solution Response]
{response}

[Current Answer]
{result}"""
}

class VerifyCodeGenerator:
    def __init__(self, model, temperature):
        self.init_messages = [
            {"role": "system", "content": gen_verify_code_dict["system_prompt"]},
            {"role": "user", "content": gen_verify_code_dict["input_case_1"]},
            {"role": "assistant", "content": gen_verify_code_dict["output_case_1"]},
            {"role": "user", "content": gen_verify_code_dict["input_case_2"]},
            {"role": "assistant", "content": gen_verify_code_dict["output_case_2"]},
            {"role": "user", "content": gen_verify_code_dict["input_case_3"]},
            {"role": "assistant", "content": gen_verify_code_dict["output_case_3"]},
        ]
        self.input_template = gen_verify_code_dict["input_template"]
        self.model = model
        self.temperature = temperature

    def gen_verify_code(self, query, response, result):
        messages = self.init_messages + [{
            "role": "user",
            "content": self.input_template.format(
                query=query,
                response=response,
                result=result,
            )
        }]
        response = call_llm(model=self.model, messages=messages, temperature=self.temperature)
        return response

# ------------------------------------------------------------------------------------------------------------------------
# ForwardVerificationExecutor

# A12 / Agent12: Forward Verification Flow Executor
# Execute forward mathematical verification flows.
execute_verify_code_dict = {
    "system_prompt": r"""You are a verification expert proficient in mathematical reasoning and tool-assisted verification.

Your task is to verify a given mathematical problem, current solution response, extracted final answer, and a forward mathematical verification flow.

The forward verification flow is mathematical reasoning. It may define expressions, sets, predicates, functions, recurrences, equations, inequalities, extrema, counts, sums, or solution sets.

A forward mathematical verification flow should:
1. Start from the original problem conditions and the current solution process.
2. Check whether important claims in the current solution can be derived from the problem conditions.
3. Check whether the final answer equals the complete target object required by the problem.
4. For all-solutions, count, maximum, minimum, probability, set, and completeness tasks, evaluate the full mathematical object.

Please proceed as follows:

1. Carefully read the problem, solution response, final answer, and forward verification flow.
2. Interpret the forward verification flow step by step.
3. Before executing, confirm that the flow includes the original conditions, intended domain, target mathematical object, and required completeness condition.
4. Evaluate simple explicit checks directly when reliable.
5. When a step requires reliable computation, symbolic manipulation, set evaluation, recurrence evaluation, dynamic programming, solving equations or inequalities, extrema computation, counting, summation, counterexample search, or completeness checking, write the corresponding local tool flow inside <tool_flow>...</tool_flow>.
6. The <tool_flow> should preserve the mathematical object from the forward verification flow. Use tool-level operations such as set construction, count, sum, max, min, argmax, argmin, solve, simplify, factor, recurrence evaluation, or completeness checking when appropriate.
7. For set, count, extremum, existence, uniqueness, or completeness tasks, evaluate the full mathematical object.
8. Use only assumptions, constants, variables, and conditions already stated in the problem, current solution response, or forward verification flow.
9. Use ASCII mathematical notation inside <tool_flow>:
   - x in S
   - CartesianProduct(A, B)
   - expr mod n = r
   - count(V)
   - max(V), min(V), sum(V)
   - V is nonempty
10. After each direct check or tool result, compare the value/object computed from the original problem conditions with the value/object claimed by the current solution or required by the current answer.
11. If the computed value/object is not mathematically equivalent to the claimed value/object, stop the forward verification immediately and return False. Do not continue to later checks after such a mismatch is found. If the check passes, continue to the next check and derive the final result.

Important:
- The final result must be either True or False.
- True means the current response passes the forward verification flow.
- False means the current response fails the forward verification flow.
- If the flow cannot be reliably confirmed, return False so that the system enters the feedback-and-revision loop.

Input information:

[Problem]
{query}

[Solution Process]
{response}

[Solution Answer]
{result}

[Forward Verification Flow]
{validate_response_fuc}

Please output exactly in the following format:

[Execution of Verification Code]
{Detailed step-by-step execution process}

[Verification Result]
<result>True</result> or <result>False</result>""",

    "input_case_1": r"""[Problem]
At the candy store, a chocolate bar costs $c$ dollars and a vanilla bar costs $2$ dollars more than a chocolate bar. Jamie buys a chocolate bar and three vanilla bars, and Kevin buys five chocolate bars. How much money, in total, do Jamie and Kevin spend at the candy store in terms of $c$?

[Solution Process]
The vanilla bar costs c+2. Jamie spends c + 3(c+2) = 3c + 6 dollars. Kevin spends 5c dollars. Therefore, the total is 3c + 6 + 5c = 8c + 6.

[Solution Answer]
8c + 6

[Forward Verification Flow]
forward_verify_total_spent(total_spent):
    # Forward verification approach:
    # 1. Start from the original problem conditions and the current solution response.
    # 2. Define the vanilla-bar cost from the original condition.
    # 3. Check the current response's key intermediate claim for Jamie's cost.
    # 4. Check Kevin's cost from the original condition.
    # 5. Compute the full target total from the original conditions.
    # 6. Verify whether the current final answer equals the full target total.

    ans := total_spent

    v(c) := c + 2

    J_claim(c) := 3*c + 6
    J_true(c) := c + 3*v(c)
    Check simplify(J_true(c) - J_claim(c)) = 0.

    K_true(c) := 5*c

    T(c) := J_true(c) + K_true(c)

    D(c) := simplify(T(c) - ans)

    Forward verification passes iff both:
    1. simplify(J_true(c) - J_claim(c)) = 0 identically;
    2. D(c) = 0 identically.""",

    "output_case_1": r"""[Execution of Verification Code]

The forward verification flow is interpreted with ans := 8c + 6.

1. From the problem condition, the vanilla bar cost is v(c) := c + 2.

2. The current response claims that Jamie spends J_claim(c) := 3c + 6.

3. From the original purchase condition, Jamie buys one chocolate bar and three vanilla bars, so:
   J_true(c) := c + 3*v(c) = c + 3*(c + 2).

4. To check this key intermediate claim, compute simplify(J_true(c) - J_claim(c)).

<tool_flow>
Let c be a symbolic variable.
v(c) := c + 2
J_true(c) := c + 3*v(c)
J_claim(c) := 3*c + 6
Return simplify(J_true(c) - J_claim(c)).
</tool_flow>

The execution result of PythonInterpreterTool is c, which is a nonzero expression.

5. Therefore, the current response's intermediate claim for Jamie's cost is not correct. The response says Jamie spends 3c + 6, but the correct expression is 4c + 6.

Since this checked key claim does not match the computed value, the current solution fails forward verification. No later checks need to be executed after this mismatch is found.

[Verification Result]
The final result of the execution is:
<result>False</result>""",

    "input_case_2": r"""[Problem]
Solve $x^2 = 4$ over the real numbers.

[Solution Process]
Solving $x^2=4$ gives $x=2$. Therefore, the answer is $x=2$.

[Solution Answer]
x = 2

[Forward Verification Flow]
forward_verify_real_solution_set(answer):
    # Forward verification approach:
    # 1. Start from the original equation and the stated domain, the real numbers.
    # 2. Compute the complete real solution set of x^2 = 4.
    # 3. Interpret the current answer as the candidate solution set.
    # 4. Verify whether the candidate solution set equals the complete real solution set.
    # 5. This checks completeness, not only whether a listed value satisfies the equation.

    ans := answer

    S_true := solve(x^2 = 4 over real numbers)

    S_ans := solution set represented by ans

    Forward verification passes iff S_ans = S_true.""",

    "output_case_2": r"""[Execution of Verification Code]

The forward verification flow is interpreted with ans := x = 2.

1. The original problem asks for solutions over the real numbers, so the target object is the complete real solution set of x^2 = 4.

2. Compute the complete real solution set.

<tool_flow>
Let x be a real symbolic variable.
Solve x^2 = 4 over the real numbers.
Return the complete solution set.
</tool_flow>

The execution result of PythonInterpreterTool gives the complete real solution set {-2, 2}.

3. The current answer represents only the candidate solution set {2}.

4. Since {2} is not equal to {-2, 2}, the current answer is incomplete.

Since this checked target object does not match the computed complete target object, the current solution fails forward verification. No later checks need to be executed after this mismatch is found.

[Verification Result]
The final result of the execution is:
<result>False</result>""",

    "input_template": r"""[Problem]
{query}

[Solution Process]
{response}

[Solution Answer]
{result}

[Forward Verification Flow]
{validate_response_fuc}""",

    "python_tool_input_template": r"""[Problem]
{query}

[Solution Process]
{response}

[Solution Answer]
{result}

[Forward Verification Flow]
{validate_response_fuc}

Execute the forward verification flow step by step. The forward verification flow is mathematical reasoning.
Use PythonInterpreterTool when a mathematical object needs reliable evaluation.
Wrap the mathematical tool flow with <tool_flow> and </tool_flow>.
The <tool_flow> should preserve the mathematical object from the forward verification flow, such as a set, count, sum, max, min, argmax, argmin, solve target, simplify target, recurrence, or completeness check.
For set, count, extremum, all-solutions, and completeness verification, evaluate the full mathematical object.
After each direct check or tool result, compare the computed value/object with the value/object claimed by the current solution or required by the current answer.
If the computed value/object is not mathematically equivalent to the claimed value/object, stop immediately and return <result>False</result>. Do not continue to later checks after such a mismatch is found.
Use ASCII mathematical notation: x in S, CartesianProduct(A, B), expr mod n = r, count(V), max(V), and V is nonempty.
End with <result>True</result> or <result>False</result>.""",

    "python_tool_system_suffix": r"""

You may use PythonInterpreterTool as a mathematical verification workbench.
It can support exact arithmetic, symbolic algebra, finite set construction, counting, summation, max/min or argmax/argmin over defined domains, recurrence or DP evaluation, equation solving, inequality checking, counterexample search, and completeness checks.
When such reliable computation is needed, write the local mathematical tool flow in <tool_flow>...</tool_flow>.
After a tool call succeeds, do not issue an equivalent <tool_flow> again. If it fails, revise the <tool_flow> based on the error before retrying.
The tool flow will be implemented as executable Python before PythonInterpreterTool execution.
A tool flow should preserve the mathematical object from the forward verification flow and evaluate the full set, count, extremum, recurrence, symbolic target, solution set, or completeness condition when required.
After each direct check or tool result, compare the computed value/object with the value/object claimed by the current solution or required by the current answer. If they are not mathematically equivalent, stop immediately and end with <result>False</result>.
Use ASCII mathematical notation in tool flows.
After passing a check, continue the forward verification and end with <result>True</result> or <result>False</result>."""
}


def split_content_result(response):
    if "[Verification Result]" in response:
        content,result=response.split("[Verification Result]")
        content=content.strip()
        result=result.strip()
        if "true" in result.lower():
            result=True
        else:
            result=False
        return content,result
    else:
        return response,True

def call_llm_with_tool_flow_loop(
    model,
    messages,
    temperature,
    tool_runner,
    caller,
    final_instruction,
):
    if tool_runner is None:
        return call_llm(
            model=model,
            messages=messages,
            temperature=temperature,
        )

    return tool_runner(
        model=model,
        messages=messages,
        temperature=temperature,
        caller=caller,
        final_instruction=final_instruction,
    )

class PythonToolRequiredExecutor:
    # 3-1_Forward must be run through the PythonTool test entrypoint.
    # The old pure-LLM simulated execution path is intentionally disabled to
    # avoid confusing it with the actual PythonInterpreterTool experiment.
    def __init__(self,model,temperature):
        raise RuntimeError(
            "Legacy pure-LLM verification execution is disabled for 3-1_Forward. "
            "Run 3-1_Forward through repro/scripts/math_hard_test_parallel_python_tool.py, "
            "which uses PythonToolLLMCodeExecutor and PythonInterpreterTool."
        )

    def execute(self,query,response,result,validate_response_fuc):
        raise RuntimeError("Legacy pure-LLM verification execution is disabled for 3-1_Forward.")

# ------------------------------------------------------------------------------------------------------------------------
# Forward Feedback Summarizer

# A13 / Agent13: Forward Failure Feedback Summarizer
# Convert forward verification execution process into natural-language feedback.
feedback_dict = {
    "input_case_1": r"""Extract key information from the failed execution process of the forward verification flow and convert it into natural language feedback for revising the solution.

[Problem]
At the candy store, a chocolate bar costs $c$ dollars and a vanilla bar costs $2$ dollars more than a chocolate bar. Jamie buys a chocolate bar and three vanilla bars, and Kevin buys five chocolate bars. How much money, in total, do Jamie and Kevin spend at the candy store in terms of $c$?

[Current Solution]
The vanilla bar costs c+2. Jamie spends c + 3(c+2) = 3c + 6 dollars. Kevin spends 5c dollars. Therefore, the total is 3c + 6 + 5c = 8c + 6.

[Current Answer]
8c + 6

[Forward Verification Flow]
forward_verify_total_spent(total_spent):
    ans := total_spent

    v(c) := c + 2

    J_claim(c) := 3*c + 6
    J_true(c) := c + 3*v(c)
    Check simplify(J_true(c) - J_claim(c)) = 0.

    K_true(c) := 5*c

    T(c) := J_true(c) + K_true(c)

    D(c) := simplify(T(c) - ans)

    Forward verification passes iff both:
    1. simplify(J_true(c) - J_claim(c)) = 0 identically;
    2. D(c) = 0 identically.

[Forward Verification Result]
False

[Forward Verification Execution Record]
The current answer is 8c + 6.

The forward verification flow checks the current response's key intermediate claim for Jamie's cost. The response claims:
J_claim(c) := 3c + 6.

From the problem, vanilla cost is v(c) := c + 2, and Jamie buys one chocolate bar and three vanilla bars, so:
J_true(c) := c + 3*v(c) = c + 3*(c + 2).

A local tool flow computes:
<tool_flow>
Let c be a symbolic variable.
v(c) := c + 2
J_true(c) := c + 3*v(c)
J_claim(c) := 3*c + 6
Return simplify(J_true(c) - J_claim(c)).
</tool_flow>

The execution result of PythonInterpreterTool is c, which is nonzero.

Therefore, the current response's intermediate claim for Jamie's cost is wrong. The current solution fails forward verification.

[Verification Result]
The final result of the execution is:
<result>False</result>""",

    "output_case_1": r"""The current solution contains an error in the forward computation of Jamie's cost. It claims that \(c + 3(c+2) = 3c + 6\), but the correct simplification is \(c + 3(c+2) = c + 3c + 6 = 4c + 6\). Because Jamie's cost is wrong, the final total \(8c+6\) is also wrong. The solution should recompute Jamie's cost and then recompute the total spent by Jamie and Kevin.""",

    "input_case_2": r"""Extract key information from the failed execution process of the forward verification flow and convert it into natural language feedback for revising the solution.

[Problem]
Solve $x^2 = 4$ over the real numbers.

[Current Solution]
Solving $x^2=4$ gives $x=2$. Therefore, the answer is $x=2$.

[Current Answer]
x = 2

[Forward Verification Flow]
forward_verify_real_solution_set(answer):
    ans := answer

    S_true := solve(x^2 = 4 over real numbers)

    S_ans := solution set represented by ans

    Forward verification passes iff S_ans = S_true.

[Forward Verification Result]
False

[Forward Verification Execution Record]
The current answer is x = 2.

The forward verification flow computes the complete real solution set of x^2 = 4:
<tool_flow>
Let x be a real symbolic variable.
Solve x^2 = 4 over the real numbers.
Return the complete solution set.
</tool_flow>

The execution result of PythonInterpreterTool gives {-2, 2}.

The current answer represents only {2}. Since {2} is not the complete solution set {-2, 2}, the current answer is incomplete.

[Verification Result]
The final result of the execution is:
<result>False</result>""",

    "output_case_2": r"""The current solution is incomplete. The problem asks for all real solutions of \(x^2=4\). A forward computation of the complete real solution set gives \(\{-2,2\}\), but the current answer only gives \(x=2\). The solution should include both real solutions.""",

    "input_template": r"""Extract key information from the failed execution process of the forward verification flow and convert it into natural language feedback for revising the solution.

[Problem]
{query}

[Current Solution]
{response}

[Current Answer]
{result}

[Forward Verification Flow]
{execute_code}

[Forward Verification Result]
{execute_result}

[Forward Verification Execution Record]
{execute_content}
"""
}


class Feebacker:
    def __init__(self, model, temperature):
        self.init_messages = [
            {"role": "user", "content": feedback_dict["input_case_1"]},
            {"role": "assistant", "content": feedback_dict["output_case_1"]},
            {"role": "user", "content": feedback_dict["input_case_2"]},
            {"role": "assistant", "content": feedback_dict["output_case_2"]},
        ]
        self.input_template = feedback_dict["input_template"]
        self.model = model
        self.temperature = temperature

    def __call__(self, query, response, result, execute_code, execute_result, execute_content):
        execute_content = execute_content.replace("[Execution of Verification Code]", "").strip()
        execute_content = execute_content.replace("[Execution of Verification Flow]", "").strip()

        input_content = self.input_template.format(
            query=query,
            response=response,
            result=result,
            execute_code=execute_code,
            execute_result=execute_result,
            execute_content=execute_content,
        )

        messages = self.init_messages + [{"role": "user", "content": input_content}]
        feedback = call_llm(model=self.model, messages=messages, temperature=self.temperature)
        feedback = feedback.strip()
        return feedback

# ------------------------------------------------------------------------------------------------------------------------
# Forward Verification Flow Refiner

# A14 / Agent14: Forward Verification Flow Refiner
# Reflect on and revise forward mathematical verification flows when needed.
code_refine_dict = {
    "system_prompt": r"""Your task is to reflect on whether a forward verification flow should be revised for the current latest solution.

[Forward Verification Introduction]
A forward mathematical verification flow starts from the original problem conditions and the current solution response. It checks whether the current solution can be derived correctly and completely from the problem statement.

A forward verification flow should:
1. Inspect the current solution response.
2. Identify important mathematical claims in the current response.
3. Verify those claims from the problem conditions.
4. Verify that the final answer equals the complete target object required by the problem.
5. Preserve domain and completeness requirements, such as all real solutions, all integer cases, full counts, global extrema, and complete sets.

[Important Context]
You are given:
- [Problem]: the original problem.
- [Current Solution]: the latest solution response currently kept by the system. This may be the revised response y' produced by the response refiner.
- [Current Answer]: the extracted answer from the latest solution response.
- [Initial Forward Verification Flow]: the previous forward verification flow V_f.
- [Forward Verification Process]: the previous failed execution record R_f.

The [Forward Verification Process] may have been produced before the latest solution response was adopted. Therefore:
- Use R_f as evidence about what went wrong in the previous forward verification.
- But revise the new forward verification flow for the current latest solution, not merely for the old response.
- The output V_f' should be suitable for verifying the current latest solution and current answer.

The error may come from:
1. the solution response;
2. the current answer;
3. the forward verification flow;
4. mismatch between an old forward verification flow and the latest revised solution.

You need to carefully analyze and complete the following tasks:

1. Reflect on the initial forward verification flow:
   - Treat the flow as computational mathematical reasoning.
   - Check whether the flow verifies the latest current solution in the forward direction.
   - Check whether the flow includes every original problem condition used by the solution.
   - Check whether the flow identifies key intermediate claims in the latest solution when they matter.
   - Check whether the flow defines the full mathematical target required by the problem: value, expression, set, predicate, function, recurrence, count, sum, maximum, minimum, solution set, or infinite family.
   - Check whether candidate values and intermediate claims are justified by complete expressions, sets, predicates, or extremum definitions when needed.
   - Check whether the intended domain is preserved, including positive/nonnegative, integer/real, finite/infinite, ordered/unordered, and all-solutions distinctions.
   - Check whether any tool-flow step preserves a full mathematical object, such as a full set, count, max/min, recurrence, solve target, symbolic expression, or completeness check.
   - If the latest solution changed the reasoning process while keeping the same answer, revise the flow so that it verifies the latest solution rather than the old one.

2. Provide a new forward verification flow:
   - If the initial forward verification flow is still correct for the latest current solution, repeat it in the new forward verification flow section.
   - If errors, mismatches, or inadequacies are found, revise it into a stronger forward mathematical verification flow.
   - Keep the style close to mathematical-tool calls, using expressions, sets, predicates, functions, recurrences, equations, inequalities, extrema, counts, sums, and solution sets.
   - Use mathematical-tool-level expressions such as count(V), max(V), min(V), sum(V), solve(...), simplify(...), factor(...), and recurrence evaluation.
   - Use ASCII mathematical notation: x in S, CartesianProduct(A, B), expr mod n = r, count(V), max(V), and V is nonempty.

When reliable computation is needed to diagnose the forward verification flow, write a local <tool_flow>...</tool_flow> in the [Reflection] section.
After a tool call succeeds, do not issue an equivalent <tool_flow> again. If it fails, revise the <tool_flow> based on the error before retrying.

The <tool_flow> should preserve the mathematical object related to the forward verification flow, such as a full set, count, maximum, minimum, recurrence, symbolic expression, or completeness condition.

After the tool result is returned, continue the reflection and produce the final output in the required format:
[Reflection]
...
[New Forward Verification Flow]
...

Please strictly output your analysis and revision according to the following format:

[Reflection]
{Detailed reflection process}

[New Forward Verification Flow]
{Complete forward verification flow based on the reflection}""",

    "input_case_1": r"""Reflect on the initial forward verification flow and provide a new forward verification flow for the latest current solution.

[Problem]
At the candy store, a chocolate bar costs $c$ dollars and a vanilla bar costs $2$ dollars more than a chocolate bar. Jamie buys a chocolate bar and three vanilla bars, and Kevin buys five chocolate bars. How much money, in total, do Jamie and Kevin spend at the candy store in terms of $c$?

[Current Solution]
The chocolate bar costs \(c\), and the vanilla bar costs \(c+2\). Jamie buys one chocolate bar and three vanilla bars, so Jamie spends \(c+3(c+2)=4c+6\). Kevin buys five chocolate bars, so Kevin spends \(5c\). Therefore, the total is \((4c+6)+5c=9c+6\). Thus the answer is \(9c+6\).

[Current Answer]
9c + 6

[Initial Forward Verification Flow]
forward_verify_total_spent(total_spent):
    # Forward verification approach:
    # 1. Start from the original problem conditions and the current solution response.
    # 2. Check the current response's key intermediate claim for Jamie's cost.
    # 3. Compute the full target total from the original conditions.
    # 4. Verify whether the current final answer equals the full target total.

    ans := total_spent

    v(c) := c + 2

    J_claim(c) := 3*c + 6
    J_true(c) := c + 3*v(c)
    Check simplify(J_true(c) - J_claim(c)) = 0.

    K_true(c) := 5*c

    T(c) := J_true(c) + K_true(c)

    D(c) := simplify(T(c) - ans)

    Forward verification passes iff both:
    1. simplify(J_true(c) - J_claim(c)) = 0 identically;
    2. D(c) = 0 identically.

[Forward Verification Process]
The previous execution failed because the flow checked J_claim(c) := 3c + 6, while the forward computation from the original conditions gives J_true(c) := c + 3(c+2) = 4c + 6. The tool result for simplify(J_true(c) - J_claim(c)) was c, so the previous flow returned False.

[Verification Result]
<result>False</result>""",

    "output_case_1": r"""[Reflection]
The latest current solution is different from the claim encoded in the initial forward verification flow. The latest solution correctly states that Jamie spends \(c+3(c+2)=4c+6\), while the initial verification flow still checks the old intermediate claim \(J_claim(c)=3c+6\).

The previous failure record is useful evidence, but the new forward verification flow should verify the latest solution. Therefore, the intermediate claim for Jamie's cost should be updated to \(J_claim(c)=4c+6\). The final target total should still be computed from the original problem conditions and compared with the current answer \(9c+6\).

[New Forward Verification Flow]
forward_verify_total_spent(total_spent):
    # Forward verification approach:
    # 1. Start from the original problem conditions and the latest current solution.
    # 2. Define the vanilla-bar cost from the original condition.
    # 3. Check the latest current response's key intermediate claim for Jamie's cost.
    # 4. Check Kevin's cost from the original condition.
    # 5. Compute the full target total from the original conditions.
    # 6. Verify whether the current final answer equals the full target total.

    ans := total_spent

    v(c) := c + 2

    J_claim(c) := 4*c + 6
    J_true(c) := c + 3*v(c)
    Check simplify(J_true(c) - J_claim(c)) = 0.

    K_claim(c) := 5*c
    K_true(c) := 5*c
    Check simplify(K_true(c) - K_claim(c)) = 0.

    T(c) := J_true(c) + K_true(c)

    D(c) := simplify(T(c) - ans)

    Forward verification passes iff all:
    1. simplify(J_true(c) - J_claim(c)) = 0 identically;
    2. simplify(K_true(c) - K_claim(c)) = 0 identically;
    3. D(c) = 0 identically.""",

    "input_case_2": r"""Reflect on the initial forward verification flow and provide a new forward verification flow for the latest current solution.

[Problem]
Amber, Micah, and Ahito ran 52 miles in total. Amber ran 8 miles. Micah ran 3.5 times what Amber ran. How many miles did Ahito run?

[Current Solution]
Amber ran 8 miles. Micah ran \(3.5 \times 8 = 28\) miles. Since the three runners ran 52 miles in total, Ahito ran \(52 - 8 - 28 = 16\) miles. Thus the answer is \(\boxed{16}\).

[Current Answer]
16

[Initial Forward Verification Flow]
forward_verify_ahito_miles(ahito_miles):
    # Forward verification approach:
    # 1. Start from the original distance conditions.
    # 2. Compute Micah's miles from Amber's miles.
    # 3. Compute Ahito's miles from the total-distance condition.
    # 4. Verify whether the current answer equals the computed value.

    ans := ahito_miles
    total := 52
    amber := 8
    micah := 3.5 * amber

    H := total - amber - micah

    Forward verification passes iff ans = H.

[Forward Verification Process]
The flow computes micah = 3.5*8 = 28 and H = 52 - 8 - 28 = 16. Since ans=16, the verification result is True.

[Verification Result]
<result>True</result>""",

    "output_case_2": r"""[Reflection]
The initial forward verification flow accurately checks the latest solution in the forward direction. It starts from the original conditions, computes Micah's distance from Amber's distance, then computes Ahito's distance from the total. It compares the computed value with the current answer. No revision is needed.

[New Forward Verification Flow]
forward_verify_ahito_miles(ahito_miles):
    # Forward verification approach:
    # 1. Start from the original distance conditions.
    # 2. Compute Micah's miles from Amber's miles.
    # 3. Compute Ahito's miles from the total-distance condition.
    # 4. Verify whether the current answer equals the computed value.

    ans := ahito_miles
    total := 52
    amber := 8
    micah := 3.5 * amber

    H := total - amber - micah

    Forward verification passes iff ans = H.""",

    "input_template": r"""Reflect on the initial forward verification flow and provide a new forward verification flow for the latest current solution.
(Note: If the forward verification flow is correct for the latest current solution, repeat it in the new forward verification flow section.)

[Problem]
{query}

[Current Solution]
{response}

[Current Answer]
{result}

[Initial Forward Verification Flow]
{verify_code}

[Forward Verification Process]
{verify_process}
"""
}


class CodeRefiner:
    def __init__(self, model, query, temperatue, tool_runner=None):
        self.init_messages = [
            {"role": "system", "content": code_refine_dict["system_prompt"]},
            {"role": "user", "content": code_refine_dict["input_case_1"]},
            {"role": "assistant", "content": code_refine_dict["output_case_1"]},
            {"role": "user", "content": code_refine_dict["input_case_2"]},
            {"role": "assistant", "content": code_refine_dict["output_case_2"]},
        ]
        self.template = code_refine_dict["input_template"]
        self.model = model
        self.query = query
        self.memory = []
        self.temperatue = temperatue
        self.tool_runner = tool_runner

    def extract_new_solution(self, new_response):
        if "[New Forward Verification Flow]" in new_response:
            return (
                new_response.split("[New Forward Verification Flow]")[0].strip(),
                new_response.split("[New Forward Verification Flow]")[1].strip(),
            )
        if "[New Verification Flow]" in new_response:
            return (
                new_response.split("[New Verification Flow]")[0].strip(),
                new_response.split("[New Verification Flow]")[1].strip(),
            )
        if "[New Verification Code]" in new_response:
            return (
                new_response.split("[New Verification Code]")[0].strip(),
                new_response.split("[New Verification Code]")[1].strip(),
            )
        return "", new_response

    def refine(self, response, result, verify_code, verify_process):
        verify_process = verify_process.replace("[Execution of Verification Code]", "").strip()
        verify_process = verify_process.replace("[Execution of Verification Flow]", "").strip()

        input_context = self.template.format(
            query=self.query,
            response=response,
            result=result,
            verify_code=verify_code,
            verify_process=verify_process,
        )
        messages = self.init_messages + [{"role": "user", "content": input_context}]

        final_instruction = (
            "Please now produce the final output in the required format:\n\n"
            "[Reflection]\n"
            "{Detailed reflection process}\n\n"
            "[New Forward Verification Flow]\n"
            "{Complete forward verification flow based on the reflection}"
        )


        output = call_llm_with_tool_flow_loop(
            model=self.model,
            messages=messages,
            temperature=self.temperatue,
            tool_runner=self.tool_runner,
            caller="A14",
            final_instruction=final_instruction,
        )

        output = output.strip()
        reflection, new_code = self.extract_new_solution(output)
        if "```python" in new_code:
            new_code = new_code.replace("```python", "")
        if "```" in new_code:
            new_code = new_code.replace("```", "")

        return reflection, new_code


# ------------------------------------------------------------------------------------------------------------------------
# Forward Feedback Response Refiner

# A15 / Agent15: Response Refiner
# Reflect on and revise the solution response using forward verification feedback.
refine_dict = {
    "system_prompt": r"""Your task is to revise a mathematical solution according to forward verification feedback.

Given a problem [Problem], a current solution response [Initial Solution], and feedback from a forward verification flow [Feedback], carefully compare the solution and the feedback.

Forward verification checks whether the current solution can be derived correctly and completely from the original problem conditions. It may point out an incorrect intermediate claim, wrong arithmetic, invalid formula, missing case, incomplete solution set, wrong domain, wrong count, or wrong final target object.

You need to complete the following tasks:

1. Reflect on the initial solution:
   - Compare the solution and the feedback.
   - Examine the solution's forward reasoning step by step.
   - Decide whether the feedback reveals an error in the solution or a possible weakness in the forward verification flow.
   - If the feedback is correct, revise the solution.
   - If the solution is correct and the feedback is caused by a weak verification flow, keep the solution and state why.

2. Provide a new solution:
   - If the initial solution is correct, use the original solution process.
   - If errors or inadequacies are found, rewrite the solution so that it correctly derives the answer from the original problem conditions.
   - The new solution must be a complete solution response.
   - The final answer must be clearly stated.

When reliable constructive computation is needed to revise the solution, you may write one local <tool_flow>...</tool_flow> in the [Reflection] section.
After a tool call succeeds, do not issue an equivalent <tool_flow> again. If it fails, revise the <tool_flow> based on the error before retrying.

The <tool_flow> should compute a candidate answer or a key mathematical object from the original problem conditions and the forward verification feedback.

Use ASCII mathematical notation in constructive tool flows:
- x in S
- CartesianProduct(A, B)
- expr mod n = r
- count(V)
- max(V), min(V), sum(V)
- V is nonempty

Tool-flow usage example for constructive solution repair:
<tool_flow>
Let c be a symbolic variable.
Compute T(c) = c + 3*(c + 2) + 5*c.
Return simplify(T(c)).
</tool_flow>

After the tool result is returned, continue the reflection and produce the final output in the required format:
[Reflection]
...
[New Solution]
...

Note:
- Always provide a new solution section.
- If the feedback does not require a mathematical correction, do not merely paraphrase or restyle the solution. Keep the original mathematical solution unchanged and explicitly state that no mathematical revision is needed.
- Do not change the final answer unless the feedback shows a mathematical mismatch in a checked key claim or in the complete target object.
- When the feedback identifies an incorrect intermediate claim, do not only patch that single line. Re-derive all downstream steps that depend on that claim and make sure the final answer follows from the corrected reasoning.
- If you revise the solution, make the mathematical change explicit. Do not output a cosmetically different solution when the mathematical content is unchanged.

Please strictly output your analysis and revision according to the following format:

[Reflection]
{{Detailed reflection process}}

[New Solution]
{{Complete solution process based on the reflection}}""",

    "input_case_1": r"""Reflect on the initial solution and provide a new solution.

[Problem]
At the candy store, a chocolate bar costs $c$ dollars and a vanilla bar costs $2$ dollars more than a chocolate bar. Jamie buys a chocolate bar and three vanilla bars, and Kevin buys five chocolate bars. How much money, in total, do Jamie and Kevin spend at the candy store in terms of $c$?

[Initial Solution]
The vanilla bar costs c+2. Jamie spends c + 3(c+2) = 3c + 6 dollars. Kevin spends 5c dollars. Therefore, the total is 3c + 6 + 5c = 8c + 6.

[Feedback]
The current solution contains an error in the forward computation of Jamie's cost. It claims that c + 3(c+2) = 3c + 6, but the correct simplification is c + 3(c+2) = c + 3c + 6 = 4c + 6. Because Jamie's cost is wrong, the final total 8c+6 is also wrong. The solution should recompute Jamie's cost and then recompute the total spent by Jamie and Kevin.""",

    "output_case_1": r"""[Reflection]
The feedback is correct. The solution incorrectly simplifies Jamie's cost. Since Jamie buys one chocolate bar and three vanilla bars, and each vanilla bar costs \(c+2\), Jamie's cost is:
\[
c + 3(c+2) = c + 3c + 6 = 4c + 6.
\]
Kevin's cost is \(5c\). Therefore the total is:
\[
(4c+6) + 5c = 9c + 6.
\]
So the final answer should be revised from \(8c+6\) to \(9c+6\).

[New Solution]
The chocolate bar costs \(c\) dollars, and the vanilla bar costs \(c+2\) dollars.

Jamie buys one chocolate bar and three vanilla bars, so Jamie spends
\[
c + 3(c+2) = c + 3c + 6 = 4c + 6.
\]

Kevin buys five chocolate bars, so Kevin spends
\[
5c.
\]

Therefore, the total amount spent by Jamie and Kevin is
\[
(4c+6) + 5c = 9c + 6.
\]

Thus, the total amount spent is
\[
\boxed{9c+6}.
\]""",

    "input_case_2": r"""Reflect on the initial solution and provide a new solution.

[Problem]
Solve $x^2 = 4$ over the real numbers.

[Initial Solution]
Solving $x^2=4$ gives $x=2$. Therefore, the answer is $x=2$.

[Feedback]
The current solution is incomplete. The problem asks for all real solutions of \(x^2=4\). A forward computation of the complete real solution set gives \(\{-2,2\}\), but the current answer only gives \(x=2\). The solution should include both real solutions.""",

    "output_case_2": r"""[Reflection]
The feedback is correct. The equation \(x^2=4\) over the real numbers has two solutions, because both \(2^2=4\) and \((-2)^2=4\). The initial solution only gives \(x=2\), so it omits the solution \(x=-2\).

[New Solution]
We solve
\[
x^2 = 4.
\]
Taking square roots over the real numbers gives
\[
x = \pm 2.
\]
Thus the complete set of real solutions is
\[
\boxed{x=-2,\ 2}.
\]""",

    "input_template": r"""Reflect on the initial solution and provide a new solution.

[Problem]
{query}

[Initial Solution]
(Remember, feedback is not always correct.)
{old_response}

[Feedback]
{feedback}
"""
}


class Refiner:
    def __init__(self,model,query,tool_runner=None):
        self.init_messages=[
            {"role":"system","content":refine_dict["system_prompt"]},
            {"role":"user","content":refine_dict["input_case_1"]},
            {"role":"assistant","content":refine_dict["output_case_1"]},
            {"role":"user","content":refine_dict["input_case_2"]},
            {"role":"assistant","content":refine_dict["output_case_2"]},
        ]
        self.template = refine_dict["input_template"]
        self.model = model
        self.query = query
        self.temperature=0
        self.tool_runner = tool_runner

    def extract_new_solution(self,new_response):
        if "[New Solution]" not in new_response:
            return "",new_response
        else:
            return new_response.split("[New Solution]")[0].strip(),new_response.split("[New Solution]")[1].strip()

    def refine(self, old_response,feedback):
        input_context = self.template.format(query=self.query, old_response=old_response,feedback=feedback)
        messages=self.init_messages+[{"role":"user","content":input_context}]

        final_instruction = (
            "Please now produce the final output in the required format:\n\n"
            "[Reflection]\n"
            "{Detailed reflection process}\n\n"
            "[New Solution]\n"
            "{Complete solution process based on the reflection}"
        )

        last_feedback = ""
        last_new_response = ""
        for attempt in range(1, 5):
            response = call_llm_with_tool_flow_loop(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                tool_runner=self.tool_runner,
                caller="A15",
                final_instruction=final_instruction,
            )

            new_response = str(response or "").strip()
            feedback,new_response=self.extract_new_solution(new_response)
            last_feedback = feedback
            last_new_response = new_response
            if new_response.strip():
                return feedback,new_response

            print(f"[Refiner] empty new_response attempt {attempt}/4, model={self.model}", flush=True)
            if attempt < 4:
                time.sleep(3 * attempt)

        return last_feedback,last_new_response

# ------------------------------------------------------------------------------------------------------------------------
# Contrast and Regenerate

# ContrastResponse class is kept unchanged.
# The 3-1_Forward main flow no longer relies on it as A16.
contrast_response_dict= {
    "system_prompt": "You are an expert at comparing and extracting key points.\n\nTask: Analyze the differences between two solutions and extract key points\n\nBackground: For the same problem, two solutions have provided different answers. We need to analyze these differences in depth to identify the key aspects of the problem.\n\nSteps:\n1. Carefully read the problem and both solutions\n2. Ignore surface differences in expression, focus on substantial differences in content and method\n3. Compare the core ideas, key steps, and final results of both solutions\n4. Summarize the essence of the problem reflected by these differences and the key points to note when solving\n\nOutput requirements:\n1. Concisely list 1-3 key points\n2. Each point should be specific and helpful for regenerating a better solution\n\nOutput format:\n[Comparative Analysis Process]\n{{Your comparative analysis process}}\n\n[Core Differences in Solutions]\n{{Summarize the differences in solutions based on the comparative analysis process, answer in bullet points}}\n\n[Key Points to Note When Solving the Problem]\n{{Summarize the key points to note when solving the problem based on the differences in solutions, answer in bullet points}}",
    "input_template": "Based on the following information, please analyze the differences and extract key points:\n(The key points should be detailed, not vague.)\n\n[Problem]\n{query}\n\n[Solution 1]\n{response_1}\n\n[Solution 2]\n{response_2}\n\n",
    "input_case_1": "Based on the following information, please analyze the differences and extract key points:\n(The key points should be detailed, not vague.)\n\n[Problem]\nEvaluate $\\sqrt{3\\cdot6\\cdot10\\cdot15\\cdot21\\cdot28}$.\n\n[Solution 1]\nTo evaluate \\(\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28}\\), we start by factoring each of the numbers:\n\n- \\(3\\) is already a prime number.\n- \\(6 = 2 \\cdot 3\\)\n- \\(10 = 2 \\cdot 5\\)\n- \\(15 = 3 \\cdot 5\\)\n- \\(21 = 3 \\cdot 7\\)\n- \\(28 = 2^2 \\cdot 7\\)\n\nNow, let's write the product \\(3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28\\) using these factorizations:\n\n\\[\n3 \\cdot (2 \\cdot 3) \\cdot (2 \\cdot 5) \\cdot (3 \\cdot 5) \\cdot (3 \\cdot 7) \\cdot (2^2 \\cdot 7)\n\\]\n\nCombine the factors:\n\n- The factor of \\(2\\) appears \\(1 + 1 + 2 = 4\\) times.\n- The factor of \\(3\\) appears \\(1 + 1 + 1 + 1 = 4\\) times.\n- The factor of \\(5\\) appears \\(1 + 1 = 2\\) times.\n- The factor of \\(7\\) appears \\(1 + 1 = 2\\) times.\n\nThus, the product can be rewritten as:\n\n\\[\n2^4 \\cdot 3^4 \\cdot 5^2 \\cdot 7^2\n\\]\n\nNow, take the square root of the entire expression:\n\n\\[\n\\sqrt{2^4 \\cdot 3^4 \\cdot 5^2 \\cdot 7^2} = \\sqrt{(2^2 \\cdot 3^2 \\cdot 5 \\cdot 7)^2}\n\\]\n\nThe square root of a square is the base itself:\n\n\\[\n2^2 \\cdot 3^2 \\cdot 5 \\cdot 7 = 4 \\cdot 9 \\cdot 5 \\cdot 7\n\\]\n\nCalculate the product:\n\n- \\(4 \\cdot 9 = 36\\)\n- \\(36 \\cdot 5 = 180\\)\n- \\(180 \\cdot 7 = 1260\\)\n\nTherefore, the value of \\(\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28}\\) is \\(\\boxed{1260}\\).\n\n[Solution 2]\nTo evaluate \\(\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28}\\), we start by factoring each number into its prime factors:\n\n\\[\n3 = 3\n\\]\n\n\\[\n6 = 2 \\times 3\n\\]\n\n\\[\n10 = 2 \\times 5\n\\]\n\n\\[\n15 = 3 \\times 5\n\\]\n\n\\[\n21 = 3 \\times 7\n\\]\n\n\\[\n28 = 2^2 \\times 7\n\\]\n\nNow, we multiply all these factorizations together:\n\n\\[\n3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28 = 3 \\times (2 \\times 3) \\times (2 \\times 5) \\times (3 \\times 5) \\times (3 \\times 7) \\times (2^2 \\times 7)\n\\]\n\nCombine the factors:\n\n\\[\n= 2^3 \\times 3^4 \\times 5^2 \\times 7^2\n\\]\n\nNow, take the square root of the product:\n\n\\[\n\\sqrt{2^3 \\times 3^4 \\times 5^2 \\times 7^2} = \\sqrt{2^3} \\times \\sqrt{3^4} \\times \\sqrt{5^2} \\times \\sqrt{7^2}\n\\]\n\nCalculate each square root:\n\n\\[\n\\sqrt{2^3} = \\sqrt{8} = 2^{3/2} = 2 \\sqrt{2}\n\\]\n\n\\[\n\\sqrt{3^4} = \\sqrt{81} = 3^2 = 9\n\\]\n\n\\[\n\\sqrt{5^2} = 5\n\\]\n\n\\[\n\\sqrt{7^2} = 7\n\\]\n\nCombine these results:\n\n\\[\n\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28} = (2 \\sqrt{2}) \\times 9 \\times 5 \\times 7\n\\]\n\nCalculate the product of the integers:\n\n\\[\n9 \\times 5 = 45\n\\]\n\n\\[\n45 \\times 7 = 315\n\\]\n\nThus, the expression becomes:\n\n\\[\n2 \\sqrt{2} \\times 315 = 630 \\sqrt{2}\n\\]\n\nTherefore, the value of \\(\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28}\\) is \\(\\boxed{630}\\).",
    "output_case_1": "[Comparative Analysis Process]\nI analyzed both solutions to identify the differences in their approaches and calculations:\n\n1. Both solutions correctly factorized each number into its prime components.\n2. Both solutions attempted to simplify the expression under the square root by combining the prime factors.\n3. The key difference lies in the handling of the square root:\n   - Solution 1 correctly identified that the product under the square root is a perfect square and simplified it directly to an integer.\n   - Solution 2 incorrectly calculated the square root of \\(2^3\\) as \\(2\\sqrt{2}\\) instead of simplifying the entire expression as a perfect square.\n\n[Core Differences in Solutions]\n1. Handling of the square root:\n   - Solution 1: Correctly identified the expression as a perfect square and simplified it to an integer.\n   - Solution 2: Incorrectly handled the square root of \\(2^3\\) and did not recognize the entire expression as a perfect square.\n2. Final result:\n   - Solution 1: Correctly calculated the result as 1260.\n   - Solution 2: Incorrectly calculated the result as \\(630\\sqrt{2}\\).\n\n[Key Points to Note When Solving the Problem]\n1. Verify whether the expression under the square root is a perfect square by checking the exponents of all prime factors.\n2. Simplify the expression under the square root completely before calculating the final result to ensure accuracy.\n3. Recognize that if the product of factors results in a perfect square, the square root should yield an integer, not an expression involving radicals.",
    "input_case_2": "Based on the following information, please analyze the differences and extract key points:\n(The key points should be detailed, not vague.)\n\n[Problem]\nThe trisectors of angles $B$ and $C$ of scalene triangle $ABC$ meet at points $P$ and $Q$, as shown. Angle $A$ measures 39 degrees and angle $QBP$ measures 14 degrees. What is the measure of angle $BPC$?\n\n[asy]unitsize(2cm);\nlabel(\"$B$\",(0,0),W);\n\nlabel(\"$A$\",(1.2,1.5),N);\nlabel(\"$C$\",(1,0),E);\nlabel(\"$Q$\",(.8,.6),N);\nlabel(\"$P$\",(.7,.2),N);\n\ndraw((0,0)--(1.2,1.5)--(1,0)--cycle,linewidth(1));\ndraw((0,0)--(.8,.6)--(1,0),linewidth(1));\ndraw((0,0)--(.7,.2)--(1,0),linewidth(1));\n[/asy]\n\n[Solution 1]\nTo solve for the measure of angle \\( BPC \\), we start by analyzing the given information about the triangle and the angles.\n\nWe know:\n- \\( \\angle A = 39^\\circ \\).\n- \\( \\angle QBP = 14^\\circ \\).\n\nSince \\( \\triangle ABC \\) is a triangle, the sum of its angles is \\( 180^\\circ \\). Therefore, we can find the sum of angles \\( B \\) and \\( C \\):\n\n\\[\n\\angle B + \\angle C = 180^\\circ - \\angle A = 180^\\circ - 39^\\circ = 141^\\circ.\n\\]\n\nThe problem involves trisectors of angles \\( B \\) and \\( C \\). Let the angles at \\( B \\) be trisected into \\( x, x, x \\) and at \\( C \\) into \\( y, y, y \\). Thus, we have:\n\n\\[\n3x = \\angle B \\quad \\text{and} \\quad 3y = \\angle C.\n\\]\n\nFrom the equation \\( \\angle B + \\angle C = 141^\\circ \\), we have:\n\n\\[\n3x + 3y = 141^\\circ \\quad \\Rightarrow \\quad x + y = 47^\\circ.\n\\]\n\nThe angle \\( \\angle QBP = 14^\\circ \\) is given. Since \\( Q \\) and \\( P \\) are points on the trisectors, \\( \\angle QBP \\) is the angle between the first trisector of \\( \\angle B \\) and the second trisector of \\( \\angle C \\). Therefore, we have:\n\n\\[\n\\angle QBP = x + y = 47^\\circ.\n\\]\n\nHowever, this contradicts the given \\( \\angle QBP = 14^\\circ \\). Therefore, we need to reconsider the trisector configuration. The correct interpretation is that \\( \\angle QBP = x + y - 33^\\circ \\) (since \\( x + y = 47^\\circ \\) and \\( \\angle QBP = 14^\\circ \\)).\n\nNow, we need to find \\( \\angle BPC \\). The angle \\( \\angle BPC \\) is the angle between the second trisector of \\( \\angle B \\) and the second trisector of \\( \\angle C \\). Therefore, it is:\n\n\\[\n\\angle BPC = 180^\\circ - (x + y) = 180^\\circ - 47^\\circ = 133^\\circ.\n\\]\n\nThus, the measure of angle \\( BPC \\) is \\(\\boxed{133^\\circ}\\).\n\n[Solution 2]\nTo solve this problem, we need to find the measure of angle \\( BPC \\) in the given triangle configuration. We are given that angle \\( A \\) measures 39 degrees and angle \\( QBP \\) measures 14 degrees. The points \\( P \\) and \\( Q \\) are the intersections of the trisectors of angles \\( B \\) and \\( C \\).\n\nFirst, let's denote the angles of triangle \\( ABC \\) as follows:\n- \\( \\angle A = 39^\\circ \\)\n- Let \\( \\angle B = x \\)\n- Let \\( \\angle C = y \\)\n\nSince the sum of angles in a triangle is 180 degrees, we have:\n\\[\nx + y + 39^\\circ = 180^\\circ\n\\]\n\\[\nx + y = 141^\\circ\n\\]\n\nNext, consider the trisectors of angles \\( B \\) and \\( C \\). Since \\( P \\) and \\( Q \\) are points where the trisectors meet, we can express the angles at \\( B \\) and \\( C \\) in terms of their trisectors:\n- The trisectors of \\( \\angle B \\) divide it into three equal parts, so each part is \\( \\frac{x}{3} \\).\n- The trisectors of \\( \\angle C \\) divide it into three equal parts, so each part is \\( \\frac{y}{3} \\).\n\nThe angle \\( QBP \\) is given as 14 degrees. This angle is formed by one trisector of \\( \\angle B \\) and one trisector of \\( \\angle C \\). Therefore, we can express \\( \\angle QBP \\) as:\n\\[\n\\angle QBP = \\frac{x}{3} + \\frac{y}{3} = 14^\\circ\n\\]\n\nNow, we have two equations:\n1. \\( x + y = 141^\\circ \\)\n2. \\( \\frac{x}{3} + \\frac{y}{3} = 14^\\circ \\)\n\nFrom the second equation, multiply through by 3 to clear the fractions:\n\\[\nx + y = 42^\\circ\n\\]\n\nHowever, this seems inconsistent with the first equation. Let's re-evaluate the setup. The correct interpretation should be:\n- The angle \\( QBP \\) is actually the angle between the trisectors, not the sum of the trisectors.\n\nThus, the correct setup should be:\n\\[\n\\frac{x}{3} + \\frac{y}{3} = 14^\\circ\n\\]\n\nNow, solve the system of equations:\n1. \\( x + y = 141^\\circ \\)\n2. \\( \\frac{x}{3} + \\frac{y}{3} = 14^\\circ \\)\n\nMultiply the second equation by 3:\n\\[\nx + y = 42^\\circ\n\\]\n\nThis is incorrect, so let's correct the interpretation:\n- The angle \\( QBP \\) is the angle between the trisectors, not the sum of the trisectors.\n\nRe-evaluate:\n- The angle \\( BPC \\) is the external angle formed by the trisectors.\n\nThe correct interpretation is:\n- The angle \\( BPC \\) is the external angle to \\( \\angle QBP \\).\n\nThus, the measure of angle \\( BPC \\) is:\n\\[\n180^\\circ - 14^\\circ = 166^\\circ\n\\]\n\nTherefore, the measure of angle \\( BPC \\) is \\(\\boxed{166^\\circ}\\).",
    "output_case_2": "[Comparative Analysis Process]\nI analyzed both solutions to identify the differences in their approaches and calculations:\n\n1. Both solutions correctly identified the given angles and the relationship between the angles in triangle \\(ABC\\).\n2. Both solutions attempted to use the properties of trisectors to find the measure of angle \\(BPC\\).\n3. The key difference lies in the interpretation and calculation of the angles formed by the trisectors:\n   - Solution 1 attempted to use the sum of the trisected angles and derived a contradiction, leading to a reevaluation and a final angle of \\(133^\\circ\\).\n   - Solution 2 initially misinterpreted the relationship between the trisectors and the given angle \\(QBP\\), leading to an incorrect intermediate step, but eventually corrected the interpretation to find the external angle \\(BPC\\) as \\(166^\\circ\\).\n\n[Core Differences in Solutions]\n1. Interpretation of angle \\(QBP\\):\n   - Solution 1: Initially assumed \\(QBP\\) was the sum of trisected angles, leading to a contradiction and reevaluation.\n   - Solution 2: Initially miscalculated the sum of trisected angles, but corrected to find the external angle.\n2. Calculation of angle \\(BPC\\):\n   - Solution 1: Concluded \\(BPC\\) as \\(133^\\circ\\) after reevaluation.\n   - Solution 2: Concluded \\(BPC\\) as \\(166^\\circ\\) by considering it as an external angle.\n\n[Key Points to Note When Solving the Problem]\n1. Carefully interpret the geometric configuration and relationships between angles, especially when dealing with trisectors or other subdivisions.\n2. Verify the consistency of derived equations with given conditions to avoid contradictions.\n3. Consider alternative geometric interpretations, such as external angles, when initial calculations lead to inconsistencies.",
    "input_case_3": "Based on the following information, please analyze the differences and extract key points:\n(The key points should be detailed, not vague.)\n\n[Problem]\nWhich of the following is closest to 67.4?\n\nA.      67.332\nB.      67.473\nC.      67.526\nD.      67.445\nE.      67.346\n\nExpress your answer as A, B, C, D, or E.\n\n[Solution 1]\nB. 67.473\n\n[Solution 2]\nTo determine which option is closest to 67.4, we need to calculate the absolute difference between 67.4 and each of the given options. Let's do that:\n\nA. |67.4 - 67.332| = 0.068\nB. |67.4 - 67.473| = 0.073\nC. |67.4 - 67.526| = 0.126\nD. |67.4 - 67.445| = 0.045\nE. |67.4 - 67.346| = 0.054\n\nNow, let's compare these differences:\n\n- A: 0.068\n- B: 0.073\n- C: 0.126\n- D: 0.045\n- E: 0.054\n\nThe smallest difference is 0.045, which corresponds to option D (67.445).\n\nTherefore, the option closest to 67.4 is D.",
    "output_case_3": "[Comparative Analysis Process]\nI analyzed both solutions to identify the differences in their approaches and calculations:\n\n1. Solution 1 directly selected option B without showing any calculations or reasoning.\n2. Solution 2 provided a detailed calculation of the absolute differences between 67.4 and each of the given options, then identified the smallest difference to determine the closest value.\n\n[Core Differences in Solutions]\n1. Methodology:\n- Solution 1: Provided an answer without any supporting calculations or reasoning.\n- Solution 2: Used a systematic approach by calculating the absolute differences and comparing them to find the closest value.\n2. Accuracy:\n- Solution 1: Incorrectly identified option B as the closest value.\n- Solution 2: Correctly identified option D as the closest value based on the smallest absolute difference.\n\n[Key Points to Note When Solving the Problem]\n1. Always calculate the absolute differences when determining which number is closest to a given value to ensure accuracy.\n2. Compare the calculated differences to identify the smallest one, which indicates the closest value.\n3. Provide clear reasoning and calculations to support the chosen answer, ensuring transparency and correctness.",
    "gen_response_with_insights_template": "Answer the following question:\n\n[Problem]\n{query}\n\n[Key Points to Note When Solving the Problem]\n{insights}\n\n[Solution]\n{{Your solution}}\n"
}

class ContrastResponse:
    def __init__(self,model,query,temperature=0) -> None:
        self.init_messages=[
            {"role":"system","content":contrast_response_dict["system_prompt"]},
            {"role":"user","content":contrast_response_dict["input_case_1"]},
            {"role":"assistant","content":contrast_response_dict["output_case_1"]},
            {"role":"user","content":contrast_response_dict["input_case_2"]},
            {"role":"assistant","content":contrast_response_dict["output_case_2"]},
            {"role":"user","content":contrast_response_dict["input_case_3"]},
            {"role":"assistant","content":contrast_response_dict["output_case_3"]},
        ]
        self.contrast_response_template=contrast_response_dict["input_template"]
        self.gen_response_with_insights_template=contrast_response_dict["gen_response_with_insights_template"]
        self.model=model
        self.temperature=temperature
        self.query=query
        self.memory=[]

    def extract_insights(self,output):
        if "[Key Points to Note When Solving the Problem]" in output:
            insights=output.split("[Key Points to Note When Solving the Problem]")[-1].strip()
        else:
            insights=output
        pattern = r'\d+\.\s*(.*)'
        matches = re.findall(pattern, insights)
        self.memory = []  
        for match in matches:
            self.memory.append(match)

    def contrast_response(self,response_1,response_2):
        responses=[response_1,response_2]
        random.shuffle(responses)
        input_content=self.contrast_response_template.format(query=self.query,response_1=responses[0],response_2=responses[1])
        messages=self.init_messages+[{"role":"user","content":input_content}]
        output=call_llm(model=self.model,messages=messages,temperature=self.temperature)
        self.extract_insights(output)
        return output

    def gen_response_with_insights(self):
        insights_str=""
        for i,insight in enumerate(self.memory):
            insights_str+=f"{i+1}. {insight}\n"
        input_content=self.gen_response_with_insights_template.format(query=self.query,insights=insights_str)
        new_response=call_llm(model=self.model,messages=[{"role":"user","content":input_content}],temperature=self.temperature)
        new_response=new_response.strip()
        return new_response
    
    def __call__(self,response_1,response_2):
        contrast_result=self.contrast_response(response_1,response_2)
        new_response=self.gen_response_with_insights()
        return contrast_result,new_response

# ------------------------------------------------------------------------------------------------------------------------
# Dataset

def read_dataset(start,end,shuffle=False):
    def format_sample(id,query,sample):
        return {"id":id,"query":query,"sample":sample}
    data=read_file("MATH/data/all_random_500.jsonl")
    data=[format_sample(id=dp["id"],query=dp["query"],sample=dp) for dp in data]
    if shuffle:
        random.seed(0)
        random.shuffle(data)
    if end==-1:
        return data[start:]
    else:
        return data[start:end]


# ------------------------------------------------------------------------------------------------------------------------
# ProgCo Inference

def get_first_response(model,query,temperature):
    """Get the initial response"""
    messages=[{"role":"user","content":query}]
    response=call_llm(model=model,messages=messages,temperature=temperature)
    return response

class ProgCoAgent:
    def __init__(self,args):
        self.dataset = args.dataset
        self.model=args.model
        self.max_cur_turn=args.max_cur_turn
        self.temperature=args.temperature
        # Disabled placeholder: the real BranchMath executor is injected by
        # repro/scripts/math_hard_test_parallel_python_tool.py as
        # PythonToolLLMCodeExecutor.
        self.llm_code_executor = PythonToolRequiredExecutor(model=self.model,temperature=self.temperature)
        self.feedbacker = Feebacker(model=args.model,temperature=self.temperature)
        self.logger=Logger(args)
        # Reuse the original response
        self.first_response_data=read_file(f"MATH/logs/{self.model}/all_random_500_first_response.jsonl")
        self.first_response_data={item["id"]: item for item in self.first_response_data}


    def inference(self, sample):
        cur_turn = 0
        query = sample["query"]
        tool_runner = getattr(self.llm_code_executor, "run_refiner_with_tool_flow_loop", None)

        self.response_refiner = Refiner(
            model=self.model,
            query=query,
            tool_runner=tool_runner,
        )
        self.code_refiner = CodeRefiner(
            model=self.model,
            query=query,
            temperatue=self.temperature,
            tool_runner=tool_runner,
        )

        # Keep ContrastResponse class unchanged, but A16 in 3-1_Forward is implemented
        # as a simple answer-difference controller in this inference loop.
        self.contraster = ContrastResponse(model=self.model, query=query, temperature=self.temperature)

        new_response = None
        new_verify_code = None

        # If True, the next turn should use A14's refined V_f' directly,
        # instead of regenerating a new V_f through A11.
        use_refined_verify_code_next = False

        self.logger.init_sample(sample_id=sample["id"], query=query)

        while cur_turn < self.max_cur_turn + 1:
            print(f"cur_turn:{cur_turn}")
            print("start get response")

            if cur_turn == 0:
                # Reuse the initial response from the first-response file.
                response = self.first_response_data[sample["id"]]["response"]
                result = self.first_response_data[sample["id"]]["pred"]

                # A11: Generate the forward verification flow for the current response.
                verify_code_generator = VerifyCodeGenerator(model=self.model, temperature=self.temperature)
                verify_code = verify_code_generator.gen_verify_code(
                    query=query,
                    response=response,
                    result=result,
                )

            else:
                # Use the response produced by A15 in the previous failed turn.
                if new_response is not None and str(new_response).strip():
                    response = new_response

                result = extract_answer(query, response, model=self.model)

                if (
                    use_refined_verify_code_next
                    and new_verify_code is not None
                    and str(new_verify_code).strip()
                ):
                    # A16 same-answer branch:
                    # A15 produced y', a'=a.
                    # A14 refined the current V_f into V_f' for y'.
                    # Now use V_f' directly in A12.
                    verify_code = new_verify_code
                    use_refined_verify_code_next = False
                else:
                    # A16 different-answer branch or normal new turn:
                    # regenerate V_f through A11 because the answer changed
                    # or because no refined V_f' is available.
                    verify_code_generator = VerifyCodeGenerator(model=self.model, temperature=self.temperature)
                    verify_code = verify_code_generator.gen_verify_code(
                        query=query,
                        response=response,
                        result=result,
                    )
                    use_refined_verify_code_next = False

            # A12: Perform the forward verification.
            print("start execute forward verify flow")
            execute_result, execute_content = self.llm_code_executor.execute(
                query=query,
                response=response,
                result=result,
                validate_response_fuc=verify_code,
            )

            if execute_result:  # Forward verification passed
                feedback = "forward verify passed"
                self.logger.update_turn(
                    cur_turn,
                    verify_code,
                    response,
                    result,
                    execute_result,
                    execute_content,
                    feedback,
                    reflection="",
                    code_feedback="",
                )
                cur_turn += 1
                break

            print("start forward-feedback refine")

            # A13: Generate forward feedback from A12 failed execution.
            feedback_summary = self.feedbacker(
                query=query,
                response=response,
                result=result,
                execute_code=verify_code,
                execute_result=execute_result,
                execute_content=execute_content,
            )

            a12_execute_content = execute_content

            # A15: Revise the response using forward feedback.
            reflection, revised_response = self.response_refiner.refine(
                old_response=response,
                feedback=feedback_summary,
            )

            # Safety fallback: if A15 fails to produce a nonempty new solution,
            # keep the current response.
            if not revised_response or not str(revised_response).strip():
                revised_response = response

            # A8: Extract a' from y'.
            revised_result = extract_answer(query, revised_response, model=self.model)

            # A16: Answer difference controller.
            # It only compares a and a'.
            if revised_result != result:
                # Different answer:
                # update y and a, then regenerate V_f through A11 in the next turn.
                new_response = revised_response
                new_verify_code = None
                use_refined_verify_code_next = False
                code_feedback = ""
                a16_decision = "different"
                next_verify_source = "A11"

            else:
                # Same answer:
                # keep the revised response y', but do not go back to A11.
                # Instead, ask A14 to revise the current V_f for the latest y' and a'.
                new_response = revised_response

                code_feedback, new_verify_code = self.code_refiner.refine(
                    response=revised_response,
                    result=revised_result,
                    verify_code=verify_code,
                    verify_process=a12_execute_content,
                )

                use_refined_verify_code_next = (
                    new_verify_code is not None
                    and str(new_verify_code).strip() != ""
                )

                a16_decision = "same"
                next_verify_source = "A14" if use_refined_verify_code_next else "A11_fallback"

            # Refresh execute_content because A14/A15 may have appended PythonTool events
            # through the shared executor.
            execute_content = getattr(self.llm_code_executor, "execute_content", execute_content)

            self.logger.update_turn(
                cur_turn,
                verify_code,
                response,
                result,
                execute_result,
                execute_content,
                feedback_summary,
                reflection,
                code_feedback,
                revised_response=revised_response,
                revised_result=revised_result,
                a16_decision=a16_decision,
                next_verify_source=next_verify_source,
            )

            cur_turn += 1

        # If the loop stops after a failed turn because max_cur_turn is reached,
        # prefer the latest revised response when available.
        # This preserves the behavior of a verification-driven refiner rather than
        # enforcing "only last verified response can be final".
        if new_response is not None and str(new_response).strip():
            final_response = new_response
            final_result = extract_answer(query, final_response, model=self.model)
        else:
            final_response = response
            final_result = result

        self.logger.update_final_response(final_response, final_result)
        self.logger.update_insights(self.contraster.memory)


# ------------------------------------------------------------------------------------------------------------------------
# Main Fuc

def main_fuc(args):
    agent=ProgCoAgent(args)
    data=read_dataset(start=args.start,end=args.end,shuffle=False)
    print("dataset size: ",len(data))
    for i,sample in enumerate(tqdm(data)):
        print("-"*10)
        print("id:",sample["id"])
        agent.inference(sample)
        if i%10==0:
            agent.logger.save_log()
    agent.logger.save_log()

# ------------------------------------------------------------------------------------------------------------------------
# Param

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default="qdd_GPT4o")
    parser.add_argument('--max_cur_turn', type=int, default=3,help='Maximum self-correction rounds (default: 3)')
    parser.add_argument('--start', type=int, default=0,help='Starting index of the evaluation dataset slice (default: 0)')
    parser.add_argument('--end', type=int, default=-1,help='Ending index of the evaluation dataset slice (default: -1 for entire dataset)')
    parser.add_argument('--dataset', type=str, default="MATH")
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()
    args.experiment_prefix = f"forward_start-{args.start}_end-{args.end}"
    print(vars(args))  # Print all parameters
    return args
                

if __name__ == "__main__":
    args = parse_args()
    main_fuc(args)
