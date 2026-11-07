# 4-1_BiVerify 提示词

本文档是 `4-1_BiVerify_prompts.md` 的中文翻译版。

`4-1_BiVerify` 本身不重写分支提示词。调度器 `S` 没有单独的 LLM 提示词；它是在 `upstream/progco/MATH/progco_infer_main.py::ProgCoAgent.inference` 中实现的确定性控制逻辑。

反向分支提示词逐字复用 `2-2-1-1_P4-5Py`，正向分支提示词逐字复用 `3-1_Forward`。下面的中文内容分别对应这两套源码提示词的中文翻译。

## 提示词索引

1. 反向分支 `gen_verify_code_dict`（A1：反向验证流程生成器）
2. 反向分支 `execute_verify_code_dict`（A2：反向验证流程执行器）
3. 反向分支 `feedback_dict`（A3：反向失败反馈总结器）
4. 反向分支 `code_refine_dict`（A4：反向验证流程修订器）
5. 反向分支 `refine_dict`（A5：反向反馈回复修订器）
6. 反向分支 `contrast_response_dict`（对比 / 洞察智能体）
7. 正向分支 `gen_verify_code_dict`（A11：正向验证流程生成器）
8. 正向分支 `execute_verify_code_dict`（A12：正向验证流程执行器）
9. 正向分支 `feedback_dict`（A13：正向失败反馈总结器）
10. 正向分支 `code_refine_dict`（A14：正向验证流程修订器）
11. 正向分支 `refine_dict`（A15：正向反馈回复修订器）
12. 正向分支 `contrast_response_dict`（正向分支对比 / 洞察智能体）
13. `branch_math_rewrite_dict`（A17：ToolFlow 实现器）
14. `extract_prompt`（A8：答案抽取器）
15. 重复成功 Tool Flow 的运行时后续消息

## 反向分支提示词

来源：`upstream/progco/MATH/reverse_progco_infer_main.py`
## 1. `gen_verify_code_dict`

````python
gen_verify_code_dict = {
    "system_prompt": r"""你是一名反向推理验证专家。给定一个问题，你需要为该问题生成一个反向数学验证流程。

[反向推理介绍]
* 反向推理是一种从结果出发、反向验证问题的思考方法。具体而言，它包括：

1. 从给定答案开始，而不是从问题的初始条件开始。
2. 假设该答案正确，然后基于这个假设和问题中的已知条件进行反向推导。
3. 通过反向推导，检查问题中的其他已知条件或约束是否能够被满足。
4. 如果反向推导的结果能够满足所有条件，则原答案可以被认为是正确的。

[要求]
1. 验证流程必须从候选答案开始，并将其与原问题进行核验。
2. 验证流程接收一个输入，即答案，并推导出 True/False 验证结果。
3. 生成验证流程名称后，先用注释写出反向分析验证思路，再生成验证流程。
4. 保持数学推理风格。优先使用集合、谓词、函数、递推、方程、不等式、极值、计数、求和和解集等数学对象。
5. 在合适时使用数学工具级操作：max、min、argmax、argmin、count、sum、solve、simplify、factor、recurrence、DP state 和集合构造。
6. 使用自然数学定义，而不是实现层面的细节。
7. 用完整数学对象表示完整验证目标，尤其是所有取值、计数、最大值、最小值、存在性、唯一性和无穷多解问题。
8. 在要求结果之前，先定义需要计算的数学对象。
9. 流程可以使用数学记号和自然语言定义；Python 实现稍后由工具流实现器处理。
10. 为了稳健性，使用 ASCII 数学记号：
    - x in S
    - CartesianProduct(A, B)
    - expr mod n = r
    - count(V)
    - max(V), min(V), sum(V)
    - V is nonempty
11. 只输出验证流程。""",

    "input_case_1": r"""为下列问题的答案生成一个验证流程：
一个直角三角形的一条直角边长为 9 米。另两条边的长度是连续整数米数。这个三角形的周长是多少米？""",

    "output_case_1": r"""verify_triangle_perimeter(perimeter):
    # 反向验证思路：
    # 1. 从候选周长开始。
    # 2. 用该周长表示另一条直角边和斜边。
    # 3. 定义满足原题整数、连续边长与勾股条件的边长选择集合。
    # 4. 当且仅当该集合非空时，候选周长通过验证。

    ans := perimeter
    leg1 := 9
    S := ans - leg1
    h(l) := S - l

    V(ans) := {l in positive integers :
        h(l) in positive integers
        and abs(h(l) - l) = 1
        and leg1^2 + l^2 = h(l)^2
    }

    Since l + h(l) = S by construction, every l in V(ans) gives a right triangle with perimeter ans.

    Verification passes iff V(ans) is nonempty.""",

    "input_case_2": r"""为下列问题的答案生成一个验证流程（**验证流程应从输入答案开始，并使用反向推理验证答案的正确性。**）：
下列哪个数最小？你的答案应为最小数前面的字母。

$\bullet$ A. $14.45001$ 四舍五入到最接近的十分位。

$\bullet$ B. $14.54999$ 四舍五入到最接近的十分位。

$\bullet$ C. $14.44999$ 四舍五入到最接近的十分位。""",

    "output_case_2": r"""verify_smallest_number_choice(choice):
    # 反向验证思路：
    # 1. 从候选选项开始。
    # 2. 定义每个选项的四舍五入值。
    # 3. 定义具有最小四舍五入值的选项集合。
    # 4. 当且仅当候选选项属于该集合时，通过验证。

    ans := choice
    O := {A, B, C}

    r(A) := round_to_nearest_tenth(14.45001)
    r(B) := round_to_nearest_tenth(14.54999)
    r(C) := round_to_nearest_tenth(14.44999)

    m := min {r(o) for o in O}
    V := {o in O : r(o) = m}

    Verification passes iff ans in V.""",

    "input_case_3": r"""为下列问题的答案生成一个验证流程（**验证流程应从输入答案开始，并使用反向推理验证答案的正确性。**）：
单词 ``COPYRIGHT'' 的字母可以排列成多少种不同方式？""",

    "output_case_3": r"""verify_arrangement_count(arrangement_count):
    # 反向验证思路：
    # 1. 从候选排列数开始。
    # 2. 验证 COPYRIGHT 有 9 个不同字母。
    # 3. 将排列数定义为不同字母个数的阶乘。
    # 4. 当且仅当候选计数等于该阶乘值时，通过验证。

    ans := arrangement_count
    word := COPYRIGHT
    L := set of letters in word

    Check count(L) = 9 and every letter of word occurs exactly once.

    N := 9!

    Verification passes iff ans = N.""",

    "input_template": r"""为下列问题的答案生成一个验证流程（**验证流程应从输入答案开始，并使用反向推理验证答案的正确性。**）：
{query}"""
}
````

## 2. `execute_verify_code_dict`

````python
execute_verify_code_dict = {
    "system_prompt": r"""你是一名精通数学推理和工具辅助验证的验证专家。

你的任务是验证给定数学问题及其解答。一名数据标注者已经为该问题写出了一个数学验证流程。

该验证流程是数学推理。它可能会定义集合、谓词、函数、递推、方程、不等式、极值、计数、求和或解集。

请按如下步骤进行：

1. 仔细阅读给定问题、解题过程、答案和验证流程。
2. 逐步解释验证流程。
3. 执行前，确认该流程包含原始条件、预期定义域、目标数学对象以及任何必要的完备性条件。
4. 对可靠的简单显式检查直接求值。
5. 当某一步需要可靠计算、符号操作、集合求值、递推求值、动态规划、解方程或不等式、极值计算、计数、求和、反例搜索或完备性检查时，在 <tool_flow>...</tool_flow> 中写出相应的本地工具流。
6. <tool_flow> 应保留验证流程中的数学对象。适当时使用集合构造、count、sum、max、min、argmax、argmin、solve、simplify、factor、递推求值或空集检查等工具级操作。
7. 对集合、计数、极值、存在性、唯一性或完备性任务，求值完整数学对象。
8. 只使用验证流程中已经给出的假设、常量、变量和条件。
9. 在 <tool_flow> 内使用 ASCII 数学记号：
   - x in S
   - CartesianProduct(A, B)
   - expr mod n = r
   - count(V)
   - max(V), min(V), sum(V)
   - V is nonempty
10. 工具结果返回后，继续验证并推导最终验证结果。

输入信息：

[Problem]
{query}

[Solution Process]
{response}

[Solution Answer]
{result}

[Verification Flow]
{validate_response_fuc}

请严格按以下格式输出：

[Execution of Verification Code]
{Detailed step-by-step execution process}

[Verification Result]
{True or False}""",

    "input_case_1": r"""[Problem]
乘积 $ab = 1200$，$a$ 是整数，$b$ 是奇整数。$b$ 的最大可能值是多少？

[Solution Process]
为求在 \(ab = 1200\)、\(a\) 为整数且 \(b\) 为奇整数时 \(b\) 的最大可能值，我们先分解 1200。

首先求 1200 的质因数分解：
\[
1200 = 12 \times 100 = (2^2 \times 3) \times (2^2 \times 5^2) = 2^4 \times 3 \times 5^2
\]

因为 \(b\) 是奇整数，所以它不能含有任何因子 2。因此，\(b\) 只能由 1200 的奇质因子组成，即 3 和 5。

去掉因子 2 后，1200 的奇数部分为：
\[
3 \times 5^2 = 3 \times 25 = 75
\]

因此，\(b\) 的最大可能值是 75。我们检查当 \(b=75\) 时 \(a\) 是否为整数。

计算 \(b=75\) 时的 \(a\)：
\[
a = \frac{1200}{b} = \frac{1200}{75}
\]

执行除法：
\[
1200 \div 75 = 16
\]

因为 \(a=16\) 是整数，所以条件满足。因此，\(b\) 的最大可能值为 \boxed{75}。

[Solution Answer]
75

[Verification Flow]
verify_largest_odd_b(b):
    # 反向验证思路：
    # 1. 从 b 的候选值开始。
    # 2. 定义 1200 的所有正奇约数集合。
    # 3. 当且仅当候选答案等于该集合最大值时，通过验证。

    ans := b
    N := 1200

    V := {d in positive integers :
        d divides N
        and d is odd
    }

    M := max(V)

    Verification passes iff ans = M.""",

    "output_case_1": r"""[Execution of Verification Code]

1. 将验证流程解释为 ans := 75 且 N := 1200。

2. 该流程定义了完整有效集合：
   V := {d in positive integers : d divides 1200 and d is odd}.

3. 然后定义 M := max(V)，并要求 ans = M。这正确表示了原问题，因为 b 必须是 1200 的奇整数约数，而题目询问 b 的最大可能值。

4. 计算 V 和 max(V) 是有限约数计算，因此我通过本地工具流使用 PythonInterpreterTool。

<tool_flow>
Evaluate V = {d in positive integers : d divides 1200 and d is odd}.
Return V and max(V).
</tool_flow>

PythonInterpreterTool 的执行结果给出 max(V) = 75。

5. 因为 ans = 75 等于 max(V)，验证通过。

[Verification Result]
执行的最终结果是：
<result>True</result>""",

    "input_case_2": r"""[Problem]
在糖果店，一块巧克力棒价格为 $c$ 美元，一块香草棒比巧克力棒贵 $2$ 美元。Jamie 买了一块巧克力棒和三块香草棒，Kevin 买了五块巧克力棒。用 $c$ 表示，Jamie 和 Kevin 在糖果店一共花了多少钱？

[Solution Process]
Jamie 一共花费 $c + 3(c + 2) = 3c + 6$ 美元。
Kevin 一共花费 $5c$ 美元。
Jamie 和 Kevin 的总花费为 $3c + 6 + 5c = 8c + 6$ 美元。

[Solution Answer]
8c + 6

[Verification Flow]
verify_total_spent(total_spent):
    # 反向验证思路：
    # 1. 从候选总金额表达式开始。
    # 2. 根据原始购买条件定义真实总金额表达式。
    # 3. 当且仅当符号差恒等于零时，候选答案通过验证。

    ans := total_spent

    v(c) := c + 2
    J(c) := c + 3*v(c)
    K(c) := 5*c
    T(c) := J(c) + K(c)

    D(c) := simplify(T(c) - ans)

    Verification passes iff D(c) = 0 identically.""",

    "output_case_2": r"""[Execution of Verification Code]

将验证流程解释为 ans := 8c + 6。

1. 香草棒价格为 v(c) := c + 2。

2. Jamie 的花费为 J(c) := c + 3*v(c) = c + 3*(c + 2) = 4c + 6。

3. Kevin 的花费为 K(c) := 5*c。

4. 真实总额为 T(c) := J(c) + K(c) = 9c + 6。

5. 该流程定义 D(c) := simplify(T(c) - ans)。这是合适的符号化简工具级操作。

<tool_flow>
Let c be a symbolic variable.
ans := 8*c + 6
v(c) := c + 2
J(c) := c + 3*v(c)
K(c) := 5*c
T(c) := J(c) + K(c)
Return simplify(T(c) - ans).
</tool_flow>

PythonInterpreterTool 的执行结果为 c，这是一个非零表达式。

6. 因此 T(c) 与 ans 不同。候选解答未通过验证。

[Verification Result]
执行的最终结果是：
<result>False</result>""",

    "input_template": r"""[Problem]
{query}

[Solution Process]
{response}

[Solution Answer]
{result}

[Verification Flow]
{validate_response_fuc}""",

    "python_tool_input_template": r"""[Problem]
{query}

[Solution Process]
{response}

[Solution Answer]
{result}

[Verification Flow]
{validate_response_fuc}

逐步执行验证流程。该验证流程是数学推理。
当某个数学对象需要可靠求值时，使用 PythonInterpreterTool。
用 <tool_flow> 和 </tool_flow> 包裹数学工具流。
<tool_flow> 应保留验证流程中的数学对象，例如集合、计数、求和、最大值、最小值、argmax、argmin、求解目标、化简目标、递推或空集检查。
对于集合、计数、极值和完备性验证，求值完整数学对象。
使用 ASCII 数学记号：x in S, CartesianProduct(A, B), expr mod n = r, count(V), max(V), and V is nonempty.
以 <result>True</result> 或 <result>False</result> 结束。""",

    "python_tool_system_suffix": r"""

你可以将 PythonInterpreterTool 用作数学验证工作台。
它可以支持精确算术、符号代数、有限集合构造、计数、求和、定义域上的 max/min 或 argmax/argmin、递推或 DP 求值、方程求解、不等式检查、反例搜索和完备性检查。
当需要这类可靠计算时，在 <tool_flow>...</tool_flow> 中写出本地数学工具流。
工具调用成功后，不得再次进行等价调用；调用失败时，应根据错误信息调整工具流后再重试。
该工具流会在 PythonInterpreterTool 执行前被实现为可执行 Python。
工具流应保留验证流程中的数学对象，并在需要时求值完整集合、计数、极值、递推、符号目标或完备性条件。
在工具流中使用 ASCII 数学记号。
工具结果返回后，继续验证，并以 <result>True</result> 或 <result>False</result> 结束。"""
}
````

## 3. `feedback_dict`

````python
feedback_dict = {
    "input_case_1": r"""从验证流程的执行过程中提取关键信息，并将其转换为自然语言形式。

[Problem]
在直角三角形 $EAD$ 中，直角在 $A$，$AE=4$ 单位，$AB=BC=CD$，且三角形 $ABE$ 的面积为 6 平方单位。线段 $CE$ 的长度是多少？将答案表示为四舍五入到最接近十分位的小数。

[Execution of Verification Code]
预测的线段 \( CE \) 长度为 9.4。

验证流程根据面积条件重建 AB：
因为 area_ABE = 6 且 AE = 4，所以 AB = 2*6/4 = 3。

因为 AB = BC = CD，所以 AD = 3*AB = 9。

流程根据原始直角三角形定义所需长度：
CE := sqrt(AE^2 + AD^2).

该计算由本地工具流求值：
<tool_flow>
Evaluate CE = sqrt(4^2 + 9^2).
Return CE exactly and rounded to the nearest tenth.
</tool_flow>

PythonInterpreterTool 的执行结果为 sqrt(97)，四舍五入到最接近十分位为 9.8。

因为 9.8 与候选值 9.4 不同，所以验证失败。

[Verification Result]
执行的最终结果是：
<result>False</result>""",

    "output_case_1": r"""验证流程表明，给定长度 9.4 与从原题条件推出的值不同。由三角形 ABE 的面积和 AE = 4 可得 AB = 3。因为 AB = BC = CD，所以 AD = 9。因此 CE = sqrt(4^2 + 9^2) = sqrt(97)，四舍五入到最接近十分位为 9.8。由于 9.8 与 9.4 不同，原答案未通过验证。""",

    "input_case_2": r"""从验证流程的执行过程中提取关键信息，并将其转换为自然语言形式。

[Problem]
计算 $\sqrt{5\cdot10\cdot14\cdot21\cdot15\cdot20}$。

[Execution of Verification Code]
预测答案为 \(1050\sqrt{2}\)。

验证流程定义根号下的乘积：
P := 5*10*14*21*15*20.

本地工具流对该乘积求值：
<tool_flow>
Evaluate P = 5*10*14*21*15*20.
Return P.
</tool_flow>

PythonInterpreterTool 的执行结果为 4410000。

验证流程将候选答案的平方与 P 比较：
<tool_flow>
Let ans = 1050*sqrt(2).
Return simplify(ans^2).
</tool_flow>

PythonInterpreterTool 的执行结果为 2205000。

因为 2205000 与 4410000 不同，候选答案未通过平方根验证。

[Verification Result]
执行的最终结果是：
<result>False</result>""",

    "output_case_2": r"""验证流程表明，预测答案 \(1050\sqrt{2}\) 与原始平方根表达式不一致。根号下的精确乘积为 4410000，而预测答案的平方为 2205000。由于这两个值不同，预测答案未通过验证。""",

    "input_template": r"""从验证流程的执行过程中提取关键信息，并将其转换为自然语言形式。

[Problem]
{query}

[Execution of Verification Code]
{execute_content}
"""
}
````

## 4. `code_refine_dict`

````python
code_refine_dict = {
    "system_prompt": r"""你的任务是反思一个验证流程是否正确。

[反向推理介绍]
* 反向推理是一种从结果出发、反向验证问题的思考方法。具体而言，它包括：

1. 从给定答案开始，而不是从问题的初始条件开始。
2. 假设该答案正确，然后基于该假设和问题中的已知条件进行反向推导。
3. 通过反向推导，检查问题中的其他已知条件或约束是否能够被满足。
4. 如果反向推导结果能够满足所有条件，则原答案可以被认为是正确的。

给定一个问题 [Problem]，一名反向推理验证专家为它生成了数学验证流程 [Initial Verification Flow]。然而，该验证流程在验证解答 [Solution] 时未通过。

错误可能来自解答，也可能来自验证流程。

你需要仔细分析并完成以下任务：

1. 反思初始验证流程：
   - 将该流程视为计算性数学推理。
   - 检查该流程是否包含原问题的每个条件。
   - 检查该流程是否定义了问题所需的完整数学对象：集合、谓词、函数、递推、计数、求和、最大值、最小值、解集或无穷族。
   - 检查候选值是否由完整集合、谓词或极值定义来证明。
   - 检查预期定义域是否被保留，包括正/非负、整数/实数、有限/无限以及有序/无序的区别。
   - 检查任何 tool-flow 步骤是否保留完整数学对象，例如完整集合、计数、max/min、argmin、递推、求解目标或完备性检查。

2. 给出新的验证流程：
   - 如果初始验证流程正确，则在新验证流程部分重复它。
   - 如果发现错误或不足，则将其修订为更强的数学验证流程。
   - 保持接近数学工具调用的风格，使用集合、谓词、函数、递推、方程、不等式、极值、计数、求和和解集。
   - 使用数学工具级表达式，例如 count(V)、max(V)、min(V)、sum(V)、solve(...)、simplify(...) 和递推求值。
   - 使用 ASCII 数学记号：x in S, CartesianProduct(A, B), expr mod n = r, count(V), max(V), and V is nonempty.

当需要可靠计算来诊断验证流程时，在 [Reflection] 部分写一个本地 <tool_flow>...</tool_flow>。
工具调用成功后，不得再次进行等价调用；调用失败时，应根据错误信息调整工具流后再重试。

<tool_flow> 应保留与验证流程相关的数学对象，例如完整集合、计数、最大值、最小值、递推、符号表达式或完备性条件。

用于验证流程诊断的 tool-flow 示例：
<tool_flow>
Evaluate V = {(A,B) in CartesianProduct({0,1,2,3,4,5,6,7,8,9}, {0,1,2,3,4,5,6,7,8,9}) :
    (2 + 5 + A + B) mod 9 = 0}.
Return V and count(V).
</tool_flow>

工具结果返回后，继续反思，并按要求格式生成最终输出：
[Reflection]
...
[New Verification Flow]
...

请严格按照以下格式输出你的分析和修订：

[Reflection]
{Detailed reflection process}

[New Verification Flow]
{Complete verification flow based on the reflection}""",

    "input_case_1": r"""反思初始验证流程，并给出新的验证流程。
（注意：如果验证流程正确，你也需要在新验证流程部分 [New Verification Flow] 中逐字重复初始验证流程。）

[Problem]
四位数 $25AB$ 能被 9 整除，其中 $A$ 是十位数字，$B$ 是个位数字。这样的四位数 $25AB$ 可以表示多少个不同数字？

[Initial Verification Flow]
verify_divisible_by_nine_count(count):
    # 反向验证思路：
    # 1. 从候选计数开始。
    # 2. 定义完整数字对定义域。
    # 3. 使用能被 9 整除的规则定义有效数字对。
    # 4. 当且仅当候选计数等于该有效集合大小时，通过验证。

    ans := count
    D := {0,1,2,3,4,5,6,7,8,9}

    V := {(A,B) in CartesianProduct(D, D) :
        (2 + 5 + A + B) mod 9 = 0
    }

    N := count(V)

    Verification passes iff ans = N.

[Solution]
为解决这个问题，我们需要使用 9 的整除规则。一个数能被 9 整除，当且仅当其数字和能被 9 整除。

对于四位数 \(25AB\)，数字和为：
\[
2 + 5 + A + B = 7 + A + B
\]

我们要求这个和能被 9 整除。因此需要：

\[
7 + A + B \equiv 0 \pmod{9}
\]

这意味着：

\[
A + B \equiv -7 \equiv 2 \pmod{9}
\]

因此需要：

\[
A + B \equiv 2 \pmod{9}
\]

\(A\) 从 0 到 9 的每个取值都会给出唯一对应的 \(B\)，所以共有 10 个有效数对 \((A, B)\)。

因此，有 \(\boxed{10}\) 个不同的四位数 \(25AB\) 能被 9 整除。

[Verification Process]
验证流程被解释为 ans := 10。

它定义 D = {0,1,2,3,4,5,6,7,8,9}，并定义 V = {(A,B) in CartesianProduct(D, D) : (2 + 5 + A + B) mod 9 = 0}。

<tool_flow>
Evaluate V = {(A,B) in CartesianProduct({0,1,2,3,4,5,6,7,8,9}, {0,1,2,3,4,5,6,7,8,9}) :
    (2 + 5 + A + B) mod 9 = 0}.
Return V and count(V).
</tool_flow>

工具结果给出 count(V) = 11。

由于候选答案为 10，但 count(V) = 11，验证失败。

[Verification Result]
执行的最终结果是：
<result>False</result>""",

    "output_case_1": r"""[Reflection]
审查初始验证流程后，它正确表示了原问题。

1. 它从候选计数开始。
2. 它定义了完整数字对定义域 CartesianProduct(D, D)，其中 A 和 B 都是 0 到 9 的数字。
3. 它使用数字 25AB 的原始 9 整除条件定义了有效集合 V。
4. 它定义 N := count(V)，并将候选答案与 N 比较。

该流程定义了完整的有效数字对集合，因此检查了完备性。失败来自解答，因为完整计数是 11，而解答声称为 10。

[New Verification Flow]
verify_divisible_by_nine_count(count):
    # 反向验证思路：
    # 1. 从候选计数开始。
    # 2. 定义完整数字对定义域。
    # 3. 使用能被 9 整除的规则定义有效数字对。
    # 4. 当且仅当候选计数等于该有效集合大小时，通过验证。

    ans := count
    D := {0,1,2,3,4,5,6,7,8,9}

    V := {(A,B) in CartesianProduct(D, D) :
        (2 + 5 + A + B) mod 9 = 0
    }

    N := count(V)

    Verification passes iff ans = N.""",

    "input_case_2": r"""反思初始验证流程，并给出新的验证流程。

[Problem]
Amber、Micah 和 Ahito 总共跑了 52 英里。Amber 跑了 8 英里。Micah 跑的距离是 Amber 的 3.5 倍。Ahito 跑了多少英里？

[Initial Verification Flow]
verify_ahito_miles(ahito_miles):
    # 反向验证思路：
    # 1. 从 Ahito 跑的候选英里数开始。
    # 2. 根据 Amber 的英里数定义 Micah 的英里数。
    # 3. 定义总距离条件强制得到的值。
    # 4. 当且仅当候选答案等于该强制值时，通过验证。

    ans := ahito_miles
    total := 52
    amber := 8
    micah := 3.5 * amber

    H := total - amber - micah

    Verification passes iff ans = H.

[Solution]
Micah 跑了 8 x 3.5 = <<8*3.5=28>>28 英里。
所以 Ahito 跑了 52 - 8 - 28 = <<52-8-28=16>>16 英里。答案：\boxed{16}。

[Verification Process]
验证流程被解释为 ans := 16。

它定义 micah := 3.5 * 8 = 28，H := 52 - 8 - 28 = 16。

因为 ans = H，验证结果为 True。""",

    "output_case_2": r"""[Reflection]
初始验证流程准确检查了 Ahito 英里数的解答是否正确。

1. 它从候选答案开始。
2. 它正确编码了总距离、Amber 的距离以及 Micah 与 Amber 距离之间的关系。
3. 它定义了由总距离条件强制得到的值 H。
4. 它将候选答案与 H 比较。

对于 ans = 16，有 H = 52 - 8 - 3.5*8 = 16。验证流程完整，无需修订。

[New Verification Flow]
verify_ahito_miles(ahito_miles):
    # 反向验证思路：
    # 1. 从 Ahito 跑的候选英里数开始。
    # 2. 根据 Amber 的英里数定义 Micah 的英里数。
    # 3. 定义总距离条件强制得到的值。
    # 4. 当且仅当候选答案等于该强制值时，通过验证。

    ans := ahito_miles
    total := 52
    amber := 8
    micah := 3.5 * amber

    H := total - amber - micah

    Verification passes iff ans = H.""",

    "input_template": r"""反思初始验证流程，并给出新的验证流程。
（注意：如果验证流程正确，你也需要在新验证流程部分 [New Verification Flow] 中逐字重复初始验证流程。）

[Problem]
{query}

[Initial Verification Flow]
{verify_code}

[Solution]
{response}

[Verification Process]
{verify_process}
"""
}
````

## 5. `refine_dict`

````python
refine_dict={
    "system_prompt": r"""你的任务是反思一个解答是否正确。

给定一个问题 [Problem]，一名反向推理验证专家为该问题生成了数学验证流程 [Initial Verification Flow]。该验证流程为解答 [Solution] 生成了反馈。

给定一个问题、对应解答以及来自验证流程的反馈，请仔细比较解答和反馈。

你需要仔细分析并完成以下任务：
1. 反思初始解题过程：
   - 详细比较解答与反馈。
   - 逐步检查解题思路的逻辑性与准确性。
   - 考虑反馈是否支持修订解答，或是否暴露了验证流程本身的弱点。
   - 当验证流程不完整、过窄或检查了错误的数学对象时，验证反馈可能是不完整的。
   - 如果发现错误或不足，给出详细反馈和改进建议。
   - 如果解答正确，而反馈源于较弱的验证流程，则说明该解答正确。
2. 给出新的解答：
   - 如果初始解答正确，则使用原解题过程。
   - 如果发现错误或不足，则基于你的反思修订解答。

当需要可靠的构造性计算来修订解答时，在 [Reflection] 部分写一个本地 <tool_flow>...</tool_flow>。
工具调用成功后，不得再次进行等价调用；调用失败时，应根据错误信息调整工具流后再重试。

<tool_flow> 应根据原问题条件和验证反馈计算候选答案或关键数学对象。

在构造性工具流中使用 ASCII 数学记号：
- x in S
- CartesianProduct(A, B)
- expr mod n = r
- count(V)
- max(V), min(V), sum(V)
- V is nonempty

用于构造性解答修复的 tool-flow 示例：
<tool_flow>
Let c be a symbolic variable.
Compute T(c) = c + 3*(c + 2) + 5*c.
Return simplify(T(c)).
</tool_flow>

工具结果返回后，继续反思，并按要求格式生成最终输出：
[Reflection]
...
[New Solution]
...

注意：
- 总是提供 new solution 部分。

请严格按照以下格式输出你的分析和修订：

[Reflection]
{{Detailed reflection process}}

[New Solution]
{{Complete solution process based on the reflection}}""",

    "input_case_1": r"""反思初始解答，并给出新的解答。

[Problem]
计算 $\sqrt{5\cdot10\cdot14\cdot21\cdot15\cdot20}$。

[Initial Solution]
（记住，反馈并不总是正确的。）
为计算 \(\sqrt{5 \cdot 10 \cdot 14 \cdot 21 \cdot 15 \cdot 20}\)，我们先化简根号内的表达式。可将每个数分解为质因数：

\[
\begin{align*}
5 &= 5, \
10 &= 2 \cdot 5, \
14 &= 2 \cdot 7, \
21 &= 3 \cdot 7, \
15 &= 3 \cdot 5, \
20 &= 2^2 \cdot 5.
\end{align*}
\]

将这些分解代入乘积：

\[
5 \cdot 10 \cdot 14 \cdot 21 \cdot 15 \cdot 20
= 5 \cdot (2 \cdot 5) \cdot (2 \cdot 7) \cdot (3 \cdot 7) \cdot (3 \cdot 5) \cdot (2^2 \cdot 5).
\]

合并所有因子：

\[
= 2^3 \cdot 3^2 \cdot 5^4 \cdot 7^2.
\]

现在对整个表达式开平方：

\[
\sqrt{2^3 \cdot 3^2 \cdot 5^4 \cdot 7^2}.
\]

积的平方根等于平方根的乘积：

\[
= \sqrt{2^3} \cdot \sqrt{3^2} \cdot \sqrt{5^4} \cdot \sqrt{7^2}.
\]

分别化简：

\[
\sqrt{2^3} = \sqrt{2^2 \cdot 2} = 2 \sqrt{2},
\]
\[
\sqrt{3^2} = 3,
\]
\[
\sqrt{5^4} = 5^2 = 25,
\]
\[
\sqrt{7^2} = 7.
\]

合并结果：

\[
= (2 \sqrt{2}) \cdot 3 \cdot 25 \cdot 7.
\]

计算整数部分：

\[
3 \cdot 25 = 75,\quad
75 \cdot 7 = 525,\quad
2 \cdot 525 = 1050.
\]

因此表达式化简为：

\[
1050 \sqrt{2}.
\]

所以 \(\sqrt{5 \cdot 10 \cdot 14 \cdot 21 \cdot 15 \cdot 20}\) 的值为 \(\boxed{1050 \sqrt{2}}\)。

[Feedback]
问题要求计算表达式 \(\sqrt{5 \cdot 10 \cdot 14 \cdot 21 \cdot 15 \cdot 20}\)。

1. **计算这些数的乘积：**
   - 乘积为 \(5 \times 10 \times 14 \times 21 \times 15 \times 20\)。
   - 使用计算工具，乘积为 4,410,000。

2. **将给定答案平方：**
   - 预测答案为 \(1050\sqrt{2}\)。
   - 为验证，平方该表达式：\((1050\sqrt{2})^2 = 1050^2 \times 2\)。
   - 使用计算工具，平方值为 2,205,000。

3. **验证：**
   - 比较平方答案 2,205,000 与计算出的乘积 4,410,000。
   - 因为 2,205,000 不等于 4,410,000，所以验证失败。

**结论：**
预测答案 \(1050\sqrt{2}\) 不正确，因为其平方与这些数的乘积不匹配。验证过程返回结果 "False"。""",

    "output_case_1": r"""[Reflection]
初始解答尝试通过将每个数分解为质因数来化简表达式 \(\sqrt{5 \cdot 10 \cdot 14 \cdot 21 \cdot 15 \cdot 20}\)。整体方法合适，但因子计数和最终化简存在错误。

1. 质因数分解为：
   - \(5 = 5\)
   - \(10 = 2 \cdot 5\)
   - \(14 = 2 \cdot 7\)
   - \(21 = 3 \cdot 7\)
   - \(15 = 3 \cdot 5\)
   - \(20 = 2^2 \cdot 5\)

2. 合并这些因子得到：
   \[
   5 \cdot 10 \cdot 14 \cdot 21 \cdot 15 \cdot 20
   = 2^4 \cdot 3^2 \cdot 5^4 \cdot 7^2.
   \]

3. 因此，
   \[
   \sqrt{5 \cdot 10 \cdot 14 \cdot 21 \cdot 15 \cdot 20}
   = \sqrt{2^4 \cdot 3^2 \cdot 5^4 \cdot 7^2}
   = 2^2 \cdot 3 \cdot 5^2 \cdot 7.
   \]

4. 乘积为：
   \[
   2^2 \cdot 3 \cdot 5^2 \cdot 7
   = 4 \cdot 3 \cdot 25 \cdot 7
   = 2100.
   \]

反馈正确指出候选答案 \(1050\sqrt{2}\) 未通过验证，因为它的平方与根号下的乘积不匹配。解答应修订为整数值 \(2100\)。

[New Solution]
计算
\[
\sqrt{5\cdot10\cdot14\cdot21\cdot15\cdot20},
\]
先将各项分解为质因数：
\[
5 = 5,\quad
10 = 2\cdot5,\quad
14 = 2\cdot7,\quad
21 = 3\cdot7,\quad
15 = 3\cdot5,\quad
20 = 2^2\cdot5.
\]

因此，
\[
5\cdot10\cdot14\cdot21\cdot15\cdot20
= 2^4\cdot3^2\cdot5^4\cdot7^2.
\]

开平方得到
\[
\sqrt{2^4\cdot3^2\cdot5^4\cdot7^2}
= 2^2\cdot3\cdot5^2\cdot7.
\]

计算：
\[
2^2\cdot3\cdot5^2\cdot7
= 4\cdot3\cdot25\cdot7
= 2100.
\]

因此，
\[
\sqrt{5\cdot10\cdot14\cdot21\cdot15\cdot20}
= \boxed{2100}.
\]""",

    "input_case_2": r"""反思初始解答，并给出新的解答。

[Problem]
梯形 $ABCD$ 的周长是多少？ [asy]
size(3inch, 1.5inch);
pair a=(0,0), b=(18,24), c=(68,24), d=(75,0), f=(68,0), e=(18,0);
draw(a--b--c--d--cycle);
draw(b--e);
draw(shift(0,2)*e--shift(2,2)*e--shift(2,0)*e);
label("30", (9,12), W);
label("50", (43,24), N);
label("25", (71.5, 12), E);
label("24", (18, 12), E);
label("$A$", a, SW);
label("$B$", b, N);
label("$C$", c, N);
label("$D$", d, SE);
label("$E$", e, S);
[/asy]

[Initial Solution]
（记住，反馈并不总是正确的。）
为求梯形 \(ABCD\) 的周长，需要确定四条边 \(AB\)、\(BC\)、\(CD\) 和 \(DA\) 的长度。

已知：
- \(AB = 30\)
- \(BC = 50\)
- \(CD = 25\)
- \(BE = 24\)

因为 \(BE\) 垂直于 \(AE\) 和 \(ED\)，所以 \(AE\) 和 \(ED\) 是水平线段。我们可以用勾股定理求 \(AE\) 和 \(ED\)。

先计算 \(AE\)：
\[
AE = \sqrt{AB^2 - BE^2} = \sqrt{30^2 - 24^2} = \sqrt{900 - 576} = \sqrt{324} = 18
\]

接着计算 \(ED\)：
由于 \(ED\) 是从 \(E\) 到 \(D\) 的剩余水平线段，且 \(AD = AE + ED\)，需要先求 \(AD\)。

从 \(A\) 到 \(D\) 的总水平距离是 \(AE\) 和 \(ED\) 的和。因为 \(AD\) 是梯形的底边，可用 \(AD = AE + ED\)。

现在计算 \(AD\)：
\[
AD = \sqrt{CD^2 - BC^2} = \sqrt{25^2 - 50^2} = \sqrt{625 - 2500} = \sqrt{0} = 0
\]

这个计算显然不正确，因为 \(AD\) 不可能为零。重新审视问题：

因为 \(BE\) 垂直于 \(AE\) 和 \(ED\)，且 \(BE = 24\)，从 \(A\) 到 \(D\) 的水平距离为 \(AE\) 和 \(ED\) 的和。

已知 \(AE = 18\)，并且从 \(A\) 到 \(D\) 的总水平距离为 \(75\)（因为 \(D\) 在 x 轴上），所以：
\[
ED = 75 - 18 = 57
\]

现在计算梯形 \(ABCD\) 的周长：
\[
\text{Perimeter} = AB + BC + CD + DA = 30 + 50 + 25 + 75 = 180
\]

因此，梯形 \(ABCD\) 的周长是 \(\boxed{180}\)。""",

    "output_case_2": r"""[Reflection]
初始解答试图通过求和梯形各边来计算周长。然而，解答和给出的反馈中都存在对梯形尺寸与构型的基本误解。

仔细审查问题后，可以看到一些长度命名和使用上存在混淆。错误在于对题意的解释以及计算步骤的呈现。具体问题如下：

1. 初始解答错误地试图通过误用勾股定理计算 \(AD\) 和 \(ED\)，从而导致错误。
2. 在描述周长时，也似乎误解了一些边的位置。
3. 反馈虽然批评了解答，但仍然用错误的边长计算了周长。

必须重新判断并正确识别所有边长，然后直接求和得到梯形 \(ABCD\) 的周长。

[New Solution]
1. 根据题图明确识别梯形各边：
    - \(AB = 30\)（由图中标注可知）
    - \(BC = 50\)
    - \(CD = 25\)
    - \(DA = 75\)（由水平排列可知 \(AD = 18 + 57 = 75\)。）

2. 将所有边相加计算周长：
   \[
   \text{Perimeter} = AB + BC + CD + DA = 30 + 50 + 25 + 75
   \]
   \[
   = 180
   \]

因此，梯形 \(ABCD\) 的周长为 \(\boxed{180}\)。""",

    "input_template": r"""反思初始解答，并给出新的解答。

[Problem]
{query}

[Initial Solution]
（记住，反馈并不总是正确的。）
{old_response}

[Feedback]
{feedback}
"""
}
````

## 6. `contrast_response_dict`

````python
contrast_response_dict= {
    "system_prompt": r"""你是一名擅长比较并提取要点的专家。

任务：分析两个解答之间的差异，并提取关键要点。

背景：对于同一道题，两个解答给出了不同答案。我们需要深入分析这些差异，以识别该问题的关键方面。

步骤：
1. 仔细阅读题目和两个解答。
2. 忽略表达层面的表面差异，关注内容和方法上的实质差异。
3. 比较两个解答的核心思路、关键步骤和最终结果。
4. 总结这些差异反映出的问题本质，以及解题时需要注意的关键点。

输出要求：
1. 简洁列出 1-3 个关键点。
2. 每个点都应具体，并有助于重新生成更好的解答。

输出格式：
[Comparative Analysis Process]
{{Your comparative analysis process}}

[Core Differences in Solutions]
{{Summarize the differences in solutions based on the comparative analysis process, answer in bullet points}}

[Key Points to Note When Solving the Problem]
{{Summarize the key points to note when solving the problem based on the differences in solutions, answer in bullet points}}""",

    "input_template": r"""基于以下信息，请分析差异并提取关键点：
（关键点应具体，不要含糊。）

[Problem]
{query}

[Solution 1]
{response_1}

[Solution 2]
{response_2}

""",

    "input_case_1": r"""基于以下信息，请分析差异并提取关键点：
（关键点应具体，不要含糊。）

[Problem]
计算 $\sqrt{3\cdot6\cdot10\cdot15\cdot21\cdot28}$。

[Solution 1]
为计算 \(\sqrt{3 \cdot 6 \cdot 10 \cdot 15 \cdot 21 \cdot 28}\)，我们先分解每个数：

- \(3\) 本身是质数。
- \(6 = 2 \cdot 3\)
- \(10 = 2 \cdot 5\)
- \(15 = 3 \cdot 5\)
- \(21 = 3 \cdot 7\)
- \(28 = 2^2 \cdot 7\)

于是乘积可写为：
\[
3 \cdot (2 \cdot 3) \cdot (2 \cdot 5) \cdot (3 \cdot 5) \cdot (3 \cdot 7) \cdot (2^2 \cdot 7)
\]

合并因子：

- 因子 \(2\) 出现 \(1 + 1 + 2 = 4\) 次。
- 因子 \(3\) 出现 \(1 + 1 + 1 + 1 = 4\) 次。
- 因子 \(5\) 出现 \(1 + 1 = 2\) 次。
- 因子 \(7\) 出现 \(1 + 1 = 2\) 次。

因此乘积可重写为：
\[
2^4 \cdot 3^4 \cdot 5^2 \cdot 7^2
\]

对整个表达式开平方：
\[
\sqrt{2^4 \cdot 3^4 \cdot 5^2 \cdot 7^2}
= \sqrt{(2^2 \cdot 3^2 \cdot 5 \cdot 7)^2}
\]

一个平方数的平方根就是底数本身：
\[
2^2 \cdot 3^2 \cdot 5 \cdot 7 = 4 \cdot 9 \cdot 5 \cdot 7
\]

计算乘积：
- \(4 \cdot 9 = 36\)
- \(36 \cdot 5 = 180\)
- \(180 \cdot 7 = 1260\)

因此，\(\sqrt{3 \cdot 6 \cdot 10 \cdot 15 \cdot 21 \cdot 28}\) 的值为 \(\boxed{1260}\)。

[Solution 2]
为计算 \(\sqrt{3 \cdot 6 \cdot 10 \cdot 15 \cdot 21 \cdot 28}\)，先将每个数分解为质因数：

\[
3 = 3,\quad
6 = 2 \times 3,\quad
10 = 2 \times 5,\quad
15 = 3 \times 5,\quad
21 = 3 \times 7,\quad
28 = 2^2 \times 7
\]

将所有分解相乘：
\[
3 \cdot 6 \cdot 10 \cdot 15 \cdot 21 \cdot 28
= 3 \times (2 \times 3) \times (2 \times 5) \times (3 \times 5) \times (3 \times 7) \times (2^2 \times 7)
\]

合并因子：
\[
= 2^3 \times 3^4 \times 5^2 \times 7^2
\]

然后对乘积开平方：
\[
\sqrt{2^3 \times 3^4 \times 5^2 \times 7^2}
= \sqrt{2^3} \times \sqrt{3^4} \times \sqrt{5^2} \times \sqrt{7^2}
\]

分别计算每个平方根：
\[
\sqrt{2^3} = \sqrt{8} = 2^{3/2} = 2 \sqrt{2},\quad
\sqrt{3^4} = 9,\quad
\sqrt{5^2} = 5,\quad
\sqrt{7^2} = 7
\]

合并结果：
\[
\sqrt{3 \cdot 6 \cdot 10 \cdot 15 \cdot 21 \cdot 28}
= (2 \sqrt{2}) \times 9 \times 5 \times 7
\]

整数部分为 \(9 \times 5 = 45\)，\(45 \times 7 = 315\)，所以表达式变为：
\[
2 \sqrt{2} \times 315 = 630 \sqrt{2}
\]

因此，\(\sqrt{3 \cdot 6 \cdot 10 \cdot 15 \cdot 21 \cdot 28}\) 的值为 \(\boxed{630}\)。""",

    "output_case_1": r"""[Comparative Analysis Process]
我分析了两个解答，以识别它们在方法和计算上的差异：

1. 两个解答都正确地把每个数分解为质因数。
2. 两个解答都试图通过合并质因数来化简根号下的表达式。
3. 关键差异在于对平方根的处理：
   - 解答 1 正确识别出根号下的乘积是完全平方数，并直接化简为整数。
   - 解答 2 错误地将 \(\sqrt{2^3}\) 计算为 \(2\sqrt{2}\)，而没有把整个表达式作为完全平方数整体化简。

[Core Differences in Solutions]
1. 平方根处理：
   - 解答 1：正确识别表达式是完全平方数，并化简为整数。
   - 解答 2：错误处理了 \(\sqrt{2^3}\)，且没有识别整个表达式是完全平方数。
2. 最终结果：
   - 解答 1：正确计算结果为 1260。
   - 解答 2：错误计算结果为 \(630\sqrt{2}\)。

[Key Points to Note When Solving the Problem]
1. 通过检查所有质因数指数，确认根号下表达式是否为完全平方数。
2. 在计算最终结果前，完整化简根号下表达式以保证准确性。
3. 如果因子乘积得到完全平方数，则平方根应为整数，而不是含根式的表达式。""",

    "input_case_2": r"""基于以下信息，请分析差异并提取关键点：
（关键点应具体，不要含糊。）

[Problem]
不等边三角形 $ABC$ 中，角 $B$ 和角 $C$ 的三等分线如图相交于点 $P$ 和 $Q$。角 $A$ 为 39 度，角 $QBP$ 为 14 度。角 $BPC$ 的度数是多少？

[asy]unitsize(2cm);
label("$B$",(0,0),W);

label("$A$",(1.2,1.5),N);
label("$C$",(1,0),E);
label("$Q$",(.8,.6),N);
label("$P$",(.7,.2),N);

draw((0,0)--(1.2,1.5)--(1,0)--cycle,linewidth(1));
draw((0,0)--(.8,.6)--(1,0),linewidth(1));
draw((0,0)--(.7,.2)--(1,0),linewidth(1));
[/asy]

[Solution 1]
为求角 \(BPC\)，先分析给定的三角形和角度信息。

已知：
- \(\angle A = 39^\circ\)。
- \(\angle QBP = 14^\circ\)。

由于 \(\triangle ABC\) 是三角形，其内角和为 \(180^\circ\)。因此：
\[
\angle B + \angle C = 180^\circ - 39^\circ = 141^\circ.
\]

题目涉及角 \(B\) 和角 \(C\) 的三等分线。设 \(B\) 处三等分角为 \(x,x,x\)，\(C\) 处三等分角为 \(y,y,y\)，则：
\[
3x = \angle B,\quad 3y = \angle C.
\]

由 \(3x+3y=141^\circ\) 得：
\[
x+y=47^\circ.
\]

给定 \(\angle QBP=14^\circ\)。由于 \(Q\) 和 \(P\) 位于三等分线上，\(\angle QBP\) 是角 \(B\) 的第一条三等分线与角 \(C\) 的第二条三等分线之间的角。因此：
\[
\angle QBP=x+y=47^\circ.
\]

然而这与给定的 \(\angle QBP=14^\circ\) 矛盾，所以需要重新考虑三等分线构型。正确解释是 \(\angle QBP=x+y-33^\circ\)（因为 \(x+y=47^\circ\)，而 \(\angle QBP=14^\circ\)）。

现在求 \(\angle BPC\)。角 \(\angle BPC\) 是角 \(B\) 第二条三等分线和角 \(C\) 第二条三等分线之间的角，因此：
\[
\angle BPC=180^\circ-(x+y)=180^\circ-47^\circ=133^\circ.
\]

因此，角 \(BPC\) 的度数为 \(\boxed{133^\circ}\)。

[Solution 2]
为解此题，需要求给定构型中角 \(BPC\) 的度数。已知角 \(A\) 为 39 度，角 \(QBP\) 为 14 度。点 \(P\) 和 \(Q\) 是角 \(B\) 与角 \(C\) 的三等分线交点。

设三角形 \(ABC\) 的角为：
- \(\angle A = 39^\circ\)
- \(\angle B = x\)
- \(\angle C = y\)

三角形内角和为 180 度，因此：
\[
x+y+39^\circ=180^\circ,\quad x+y=141^\circ.
\]

三等分线将 \(\angle B\) 分为三份，每份 \(\frac{x}{3}\)，将 \(\angle C\) 分为三份，每份 \(\frac{y}{3}\)。

给定 \(\angle QBP=14^\circ\)。该角由一条 \(B\) 的三等分线和一条 \(C\) 的三等分线形成，因此写作：
\[
\angle QBP=\frac{x}{3}+\frac{y}{3}=14^\circ.
\]

于是得到 \(x+y=42^\circ\)，这与 \(x+y=141^\circ\) 不一致。重新评估后，认为 \(\angle QBP\) 是三等分线之间的角，而不是三等分角之和。

最终将 \(\angle BPC\) 视为 \(\angle QBP\) 的外角：
\[
180^\circ-14^\circ=166^\circ.
\]

因此，角 \(BPC\) 的度数为 \(\boxed{166^\circ}\)。""",

    "output_case_2": r"""[Comparative Analysis Process]
我分析了两个解答，以识别它们在方法和计算上的差异：

1. 两个解答都正确识别了给定角以及三角形 \(ABC\) 中角之间的关系。
2. 两个解答都尝试使用三等分线性质求 \(\angle BPC\)。
3. 关键差异在于对三等分线形成角的解释和计算：
   - 解答 1 尝试使用三等分角之和，并导出矛盾，随后重新评估，得到最终角 \(133^\circ\)。
   - 解答 2 起初也误解三等分线和给定角 \(QBP\) 的关系，导致错误中间步骤，但最终改用外角解释，得到 \(\angle BPC=166^\circ\)。

[Core Differences in Solutions]
1. 对角 \(QBP\) 的解释：
   - 解答 1：起初假设 \(QBP\) 是三等分角之和，产生矛盾后重新评估。
   - 解答 2：起初误算三等分角之和，后来改为外角解释。
2. 对角 \(BPC\) 的计算：
   - 解答 1：重新评估后认为 \(BPC=133^\circ\)。
   - 解答 2：把 \(BPC\) 作为外角，认为 \(BPC=166^\circ\)。

[Key Points to Note When Solving the Problem]
1. 仔细解释几何构型和角之间的关系，尤其是涉及三等分线或其他分割线时。
2. 验证推导出的方程是否与给定条件一致，以避免矛盾。
3. 当初始计算产生不一致时，应考虑其他几何解释，例如外角关系。""",

    "input_case_3": r"""基于以下信息，请分析差异并提取关键点：
（关键点应具体，不要含糊。）

[Problem]
下列哪个数最接近 67.4？

A.      67.332
B.      67.473
C.      67.526
D.      67.445
E.      67.346

请将答案表示为 A、B、C、D 或 E。

[Solution 1]
B. 67.473

[Solution 2]
为了判断哪个选项最接近 67.4，需要计算 67.4 与每个选项的绝对差：

A. |67.4 - 67.332| = 0.068
B. |67.4 - 67.473| = 0.073
C. |67.4 - 67.526| = 0.126
D. |67.4 - 67.445| = 0.045
E. |67.4 - 67.346| = 0.054

比较这些差值：

- A: 0.068
- B: 0.073
- C: 0.126
- D: 0.045
- E: 0.054

最小差值为 0.045，对应选项 D（67.445）。

因此，最接近 67.4 的选项是 D。""",

    "output_case_3": r"""[Comparative Analysis Process]
我分析了两个解答，以识别它们在方法和计算上的差异：

1. 解答 1 直接选择了选项 B，没有给出任何计算或推理。
2. 解答 2 详细计算了 67.4 与每个选项之间的绝对差，并通过最小差值确定最接近的数。

[Core Differences in Solutions]
1. 方法：
- 解答 1：只给出答案，没有支持性的计算或推理。
- 解答 2：采用系统方法，计算绝对差并比较以寻找最接近的值。
2. 准确性：
- 解答 1：错误地认为选项 B 最接近。
- 解答 2：基于最小绝对差，正确识别选项 D 为最接近值。

[Key Points to Note When Solving the Problem]
1. 判断哪个数最接近给定值时，应计算绝对差以确保准确性。
2. 比较所有差值，找出最小差值，该选项即为最接近值。
3. 给出清晰推理和计算来支持所选答案，保证透明性和正确性。""",

    "gen_response_with_insights_template": r"""回答下列问题：

[Problem]
{query}

[Key Points to Note When Solving the Problem]
{insights}

[Solution]
{{Your solution}}
"""
}
````
## 正向分支提示词

来源：`upstream/progco/MATH/forward_progco_infer_main.py`
## 7. `gen_verify_code_dict`（A11：正向验证流程生成器）

来源：`upstream/progco/MATH/progco_infer_main.py`

````python
gen_verify_code_dict = {
    "system_prompt": r"""你是数学验证专家。给定一道题目、当前解答回复以及从中抽取出的最终答案，请为该回复生成一个正向数学验证流程。

[正向验证简介]
正向数学验证流程从原题条件和当前解答回复出发，检查当前解答是否能从题目陈述中被正确且完整地推出。

该流程应当检查当前解答回复，识别关键数学主张，并从原始题目条件出发验证这些主张。它还应验证抽取出的最终答案是否匹配题目要求的完整目标对象。

[要求]
1. 验证流程应当沿正向方向验证当前解答回复。
2. 流程应将当前答案作为形式输入，同时使用当前解答回复来识别需要验证的关键中间主张。
3. 生成验证流程名称后，先用注释写出正向验证思路，再生成验证流程。
4. 保持数学推理风格。优先使用表达式、方程、集合、谓词、函数、递推、计数、求和、极值和解集等数学对象。
5. 适当使用数学工具级操作：simplify、expand、factor、solve、count、sum、max、min、argmax、argmin、递推求值、DP 状态、集合构造和完备性检查。
6. 使用自然的数学定义，而不是实现层细节。
7. 表示题目要求的完整目标，尤其是所有解、计数、最大值、最小值、存在性、唯一性、概率以及无限多个对象的问题。
8. 如果解答回复包含关键中间数学主张，并且这些主张对解答正确性重要，则包含对这些主张的检查。
9. 不要把解答过度拆成琐碎步骤。只验证关键主张和最终目标对象。
10. 在请求结果前，先定义要计算的数学对象。
11. 流程可以使用数学记号和自然语言定义；Python 实现稍后由工具流实现器处理。
12. 为了稳健性，使用 ASCII 数学记号：
    - x in S
    - CartesianProduct(A, B)
    - expr mod n = r
    - count(V)
    - max(V), min(V), sum(V)
    - V is nonempty
13. 只输出验证流程。""",

    "input_case_1": r"""为下面的当前解答生成一个正向验证流程。

[问题]
在糖果店，一块巧克力棒的价格是 $c$ 美元，一块香草棒比巧克力棒贵 $2$ 美元。Jamie 买了一块巧克力棒和三块香草棒，Kevin 买了五块巧克力棒。用 $c$ 表示，Jamie 和 Kevin 在糖果店一共花了多少钱？

[当前解答回复]
香草棒的价格是 c+2。Jamie 花费 c + 3(c+2) = 3c + 6 美元。Kevin 花费 5c 美元。因此，总花费是 3c + 6 + 5c = 8c + 6。

[当前答案]
8c + 6""",

    "output_case_1": r"""forward_verify_total_spent(total_spent):
    # 正向验证思路：
    # 1. 从原始题目条件和当前解答回复出发。
    # 2. 根据原始条件定义香草棒价格。
    # 3. 检查当前回复中关于 Jamie 花费的关键中间主张。
    # 4. 根据原始条件检查 Kevin 的花费。
    # 5. 从原始条件计算完整目标总花费。
    # 6. 验证当前最终答案是否等于完整目标总花费。

    ans := total_spent

    v(c) := c + 2

    # 当前回复中的关键中间主张：
    J_claim(c) := 3*c + 6
    J_true(c) := c + 3*v(c)
    Check simplify(J_true(c) - J_claim(c)) = 0.

    K_true(c) := 5*c

    T(c) := J_true(c) + K_true(c)

    D(c) := simplify(T(c) - ans)

    正向验证通过当且仅当二者都成立：
    1. simplify(J_true(c) - J_claim(c)) = 0 identically;
    2. D(c) = 0 identically.""",

    "input_case_2": r"""为下面的当前解答生成一个正向验证流程。

[问题]
在实数范围内解 $x^2 = 4$。

[当前解答回复]
解 $x^2=4$ 得到 $x=2$。因此，答案是 $x=2$。

[当前答案]
x = 2""",

    "output_case_2": r"""forward_verify_real_solution_set(answer):
    # 正向验证思路：
    # 1. 从原始方程和给定定义域（实数）出发。
    # 2. 计算 x^2 = 4 的完整实数解集。
    # 3. 将当前答案解释为候选解集。
    # 4. 验证候选解集是否等于完整实数解集。
    # 5. 这检查的是完备性，而不只是列出的某个值是否满足方程。

    ans := answer

    S_true := solve(x^2 = 4 over real numbers)

    S_ans := solution set represented by ans

    正向验证通过当且仅当 S_ans = S_true.""",

    "input_case_3": r"""为下面的当前解答生成一个正向验证流程。

[问题]
四位数 $25AB$ 能被九整除，其中 $A$ 是十位数字，$B$ 是个位数字。这样的四位数 $25AB$ 可能表示多少个不同的数？

[当前解答回复]
对于 25AB，数字和是 2+5+A+B = 7+A+B。我们需要 A+B ≡ 2 mod 9。A 从 0 到 9 的每个取值都有一个 B，所以共有 10 种可能。

[当前答案]
10""",

    "output_case_3": r"""forward_verify_divisible_by_nine_count(count):
    # 正向验证思路：
    # 1. 从 A 和 B 的完整数字定义域出发。
    # 2. 定义使 25AB 能被 9 整除的完整数字对集合。
    # 3. 计算该完整集合的大小。
    # 4. 验证当前答案是否等于完整计数。
    # 5. 这检查的是计数完备性，而不只是模运算规则。

    ans := count

    D := {0,1,2,3,4,5,6,7,8,9}

    V := {(A,B) in CartesianProduct(D, D) :
        (2 + 5 + A + B) mod 9 = 0
    }

    N := count(V)

    正向验证通过当且仅当 ans = N.""",

    "input_template": r"""为下面的当前解答生成一个正向验证流程。

[问题]
{query}

[当前解答回复]
{response}

[当前答案]
{result}"""
}
````

## 8. `execute_verify_code_dict`（A12：正向验证流程执行器）

来源：`upstream/progco/MATH/progco_infer_main.py`

````python
execute_verify_code_dict = {
    "system_prompt": r"""你是精通数学推理和工具辅助验证的验证专家。

你的任务是验证给定的数学题、当前解答回复、抽取出的最终答案，以及一个正向数学验证流程。

正向验证流程是数学推理。它可以定义表达式、集合、谓词、函数、递推、方程、不等式、极值、计数、求和或解集。

正向数学验证流程应当：
1. 从原始题目条件和当前解答过程出发。
2. 检查当前解答中的重要主张是否能从题目条件推出。
3. 检查最终答案是否等于题目要求的完整目标对象。
4. 对所有解、计数、最大值、最小值、概率、集合和完备性任务，求值完整数学对象。

请按如下步骤进行：

1. 仔细阅读题目、解答回复、最终答案和正向验证流程。
2. 逐步解释正向验证流程。
3. 在执行前，确认流程包含原始条件、预期定义域、目标数学对象以及所需的完备性条件。
4. 对可靠的简单显式检查，直接求值。
5. 当某一步需要可靠计算、符号变形、集合求值、递推求值、动态规划、解方程或不等式、极值计算、计数、求和、反例搜索或完备性检查时，在 <tool_flow>...</tool_flow> 中写出对应的本地工具流。
6. <tool_flow> 应保留正向验证流程中的数学对象。适当使用集合构造、count、sum、max、min、argmax、argmin、solve、simplify、factor、递推求值或完备性检查等工具级操作。
7. 对集合、计数、极值、存在性、唯一性或完备性任务，求值完整数学对象。
8. 只使用题目、当前解答回复或正向验证流程中已经陈述的假设、常数、变量和条件。
9. 在 <tool_flow> 内使用 ASCII 数学记号：
   - x in S
   - CartesianProduct(A, B)
   - expr mod n = r
   - count(V)
   - max(V), min(V), sum(V)
   - V is nonempty
10. 在每次直接检查或工具结果之后，比较从原始题目条件计算出的值/对象与当前解答声称的值/对象或当前答案所要求的值/对象。
11. 如果计算出的值/对象与声称的值/对象在数学上不等价，立即停止正向验证并返回 False。发现这种不一致后，不要继续后续检查。如果检查通过，则继续下一项检查并推出最终结果。

重要：
- 最终结果必须是 True 或 False。
- True 表示当前回复通过正向验证流程。
- False 表示当前回复未通过正向验证流程。
- 如果无法可靠确认流程，则返回 False，使系统进入反馈和修订循环。

输入信息：

[问题]
{query}

[解答过程]
{response}

[解答答案]
{result}

[正向验证流程]
{validate_response_fuc}

请严格按以下格式输出：

[验证代码执行过程]
{详细的逐步执行过程}

[验证结果]
<result>True</result> 或 <result>False</result>""",

    "input_case_1": r"""[问题]
在糖果店，一块巧克力棒的价格是 $c$ 美元，一块香草棒比巧克力棒贵 $2$ 美元。Jamie 买了一块巧克力棒和三块香草棒，Kevin 买了五块巧克力棒。用 $c$ 表示，Jamie 和 Kevin 在糖果店一共花了多少钱？

[解答过程]
香草棒的价格是 c+2。Jamie 花费 c + 3(c+2) = 3c + 6 美元。Kevin 花费 5c 美元。因此，总花费是 3c + 6 + 5c = 8c + 6。

[解答答案]
8c + 6

[正向验证流程]
forward_verify_total_spent(total_spent):
    # 正向验证思路：
    # 1. 从原始题目条件和当前解答回复出发。
    # 2. 根据原始条件定义香草棒价格。
    # 3. 检查当前回复中关于 Jamie 花费的关键中间主张。
    # 4. 根据原始条件检查 Kevin 的花费。
    # 5. 从原始条件计算完整目标总花费。
    # 6. 验证当前最终答案是否等于完整目标总花费。

    ans := total_spent

    v(c) := c + 2

    J_claim(c) := 3*c + 6
    J_true(c) := c + 3*v(c)
    Check simplify(J_true(c) - J_claim(c)) = 0.

    K_true(c) := 5*c

    T(c) := J_true(c) + K_true(c)

    D(c) := simplify(T(c) - ans)

    正向验证通过当且仅当二者都成立：
    1. simplify(J_true(c) - J_claim(c)) = 0 identically;
    2. D(c) = 0 identically.""",

    "output_case_1": r"""[验证代码执行过程]

正向验证流程被解释为 ans := 8c + 6。

1. 根据题目条件，香草棒价格为 v(c) := c + 2。

2. 当前回复声称 Jamie 花费 J_claim(c) := 3c + 6。

3. 根据原始购买条件，Jamie 买了一块巧克力棒和三块香草棒，所以：
   J_true(c) := c + 3*v(c) = c + 3*(c + 2)。

4. 为检查这个关键中间主张，计算 simplify(J_true(c) - J_claim(c))。

<tool_flow>
Let c be a symbolic variable.
v(c) := c + 2
J_true(c) := c + 3*v(c)
J_claim(c) := 3*c + 6
Return simplify(J_true(c) - J_claim(c)).
</tool_flow>

PythonInterpreterTool 的执行结果是 c，这是一个非零表达式。

5. 因此，当前回复关于 Jamie 花费的中间主张不正确。回复称 Jamie 花费 3c + 6，但正确表达式是 4c + 6。

由于这个被检查的关键主张与计算出的值不匹配，当前解答未通过正向验证。发现这种不一致后，不需要执行后续检查。

[验证结果]
执行的最终结果是：
<result>False</result>""",

    "input_case_2": r"""[问题]
在实数范围内解 $x^2 = 4$。

[解答过程]
解 $x^2=4$ 得到 $x=2$。因此，答案是 $x=2$。

[解答答案]
x = 2

[正向验证流程]
forward_verify_real_solution_set(answer):
    # 正向验证思路：
    # 1. 从原始方程和给定定义域（实数）出发。
    # 2. 计算 x^2 = 4 的完整实数解集。
    # 3. 将当前答案解释为候选解集。
    # 4. 验证候选解集是否等于完整实数解集。
    # 5. 这检查的是完备性，而不只是列出的某个值是否满足方程。

    ans := answer

    S_true := solve(x^2 = 4 over real numbers)

    S_ans := solution set represented by ans

    正向验证通过当且仅当 S_ans = S_true.""",

    "output_case_2": r"""[验证代码执行过程]

正向验证流程被解释为 ans := x = 2。

1. 原题要求在实数范围内求解，所以目标对象是 x^2 = 4 的完整实数解集。

2. 计算完整实数解集。

<tool_flow>
Let x be a real symbolic variable.
Solve x^2 = 4 over the real numbers.
Return the complete solution set.
</tool_flow>

PythonInterpreterTool 的执行结果给出完整实数解集 {-2, 2}。

3. 当前答案只表示候选解集 {2}。

4. 由于 {2} 不等于 {-2, 2}，当前答案不完整。

由于这个被检查的目标对象与计算出的完整目标对象不匹配，当前解答未通过正向验证。发现这种不一致后，不需要执行后续检查。

[验证结果]
执行的最终结果是：
<result>False</result>""",

    "input_template": r"""[问题]
{query}

[解答过程]
{response}

[解答答案]
{result}

[正向验证流程]
{validate_response_fuc}""",

    "python_tool_input_template": r"""[问题]
{query}

[解答过程]
{response}

[解答答案]
{result}

[正向验证流程]
{validate_response_fuc}

逐步执行正向验证流程。正向验证流程是数学推理。
当某个数学对象需要可靠求值时，使用 PythonInterpreterTool。
用 <tool_flow> 和 </tool_flow> 包裹数学工具流。
<tool_flow> 应保留正向验证流程中的数学对象，例如集合、计数、求和、最大值、最小值、argmax、argmin、solve 目标、simplify 目标、递推或完备性检查。
对于集合、计数、极值、所有解和完备性验证，求值完整数学对象。
在每次直接检查或工具结果之后，比较计算出的值/对象与当前解答声称的值/对象或当前答案所要求的值/对象。
如果计算出的值/对象与声称的值/对象在数学上不等价，立即停止并返回 <result>False</result>。发现这种不一致后，不要继续后续检查。
使用 ASCII 数学记号：x in S, CartesianProduct(A, B), expr mod n = r, count(V), max(V), and V is nonempty。
以 <result>True</result> 或 <result>False</result> 结束。""",

    "python_tool_system_suffix": r"""

你可以使用 PythonInterpreterTool 作为数学验证工作台。
它可以支持精确算术、符号代数、有限集合构造、计数、求和、在定义域上的 max/min 或 argmax/argmin、递推或 DP 求值、方程求解、不等式检查、反例搜索和完备性检查。
当需要此类可靠计算时，在 <tool_flow>...</tool_flow> 中写出本地数学工具流。
工具调用成功后，不得再次进行等价调用；调用失败时，应根据错误信息调整工具流后再重试。
工具流将在 PythonInterpreterTool 执行前被实现为可执行 Python。
工具流应保留正向验证流程中的数学对象，并在需要时求值完整集合、计数、极值、递推、符号目标、解集或完备性条件。
在每次直接检查或工具结果之后，比较计算出的值/对象与当前解答声称的值/对象或当前答案所要求的值/对象。如果它们在数学上不等价，立即停止并以 <result>False</result> 结束。
在工具流中使用 ASCII 数学记号。
检查通过后，继续正向验证，并以 <result>True</result> 或 <result>False</result> 结束。"""
}
````

## 9. `feedback_dict`（A13：正向失败反馈总结器）

来源：`upstream/progco/MATH/progco_infer_main.py`

````python
feedback_dict = {
    "input_case_1": r"""从正向验证流程的失败执行过程中提取关键信息，并将其转换为用于修订解答的自然语言反馈。

[问题]
在糖果店，一块巧克力棒的价格是 $c$ 美元，一块香草棒比巧克力棒贵 $2$ 美元。Jamie 买了一块巧克力棒和三块香草棒，Kevin 买了五块巧克力棒。用 $c$ 表示，Jamie 和 Kevin 在糖果店一共花了多少钱？

[当前解答]
香草棒的价格是 c+2。Jamie 花费 c + 3(c+2) = 3c + 6 美元。Kevin 花费 5c 美元。因此，总花费是 3c + 6 + 5c = 8c + 6。

[当前答案]
8c + 6

[正向验证流程]
forward_verify_total_spent(total_spent):
    ans := total_spent

    v(c) := c + 2

    J_claim(c) := 3*c + 6
    J_true(c) := c + 3*v(c)
    Check simplify(J_true(c) - J_claim(c)) = 0.

    K_true(c) := 5*c

    T(c) := J_true(c) + K_true(c)

    D(c) := simplify(T(c) - ans)

    正向验证通过当且仅当二者都成立：
    1. simplify(J_true(c) - J_claim(c)) = 0 identically;
    2. D(c) = 0 identically.

[正向验证结果]
False

[正向验证执行记录]
当前答案是 8c + 6。

正向验证流程检查当前回复中关于 Jamie 花费的关键中间主张。回复声称：
J_claim(c) := 3c + 6。

根据题目，香草棒价格是 v(c) := c + 2，并且 Jamie 买了一块巧克力棒和三块香草棒，所以：
J_true(c) := c + 3*v(c) = c + 3*(c + 2)。

一个本地工具流计算：
<tool_flow>
Let c be a symbolic variable.
v(c) := c + 2
J_true(c) := c + 3*v(c)
J_claim(c) := 3*c + 6
Return simplify(J_true(c) - J_claim(c)).
</tool_flow>

PythonInterpreterTool 的执行结果是 c，这是非零的。

因此，当前回复关于 Jamie 花费的中间主张是错误的。当前解答未通过正向验证。

[验证结果]
执行的最终结果是：
<result>False</result>""",

    "output_case_1": r"""当前解答在 Jamie 花费的正向计算中包含错误。它声称 \(c + 3(c+2) = 3c + 6\)，但正确化简是 \(c + 3(c+2) = c + 3c + 6 = 4c + 6\)。因为 Jamie 的花费错误，最终总额 \(8c+6\) 也错误。解答应重新计算 Jamie 的花费，然后重新计算 Jamie 和 Kevin 的总花费。""",

    "input_case_2": r"""从正向验证流程的失败执行过程中提取关键信息，并将其转换为用于修订解答的自然语言反馈。

[问题]
在实数范围内解 $x^2 = 4$。

[当前解答]
解 $x^2=4$ 得到 $x=2$。因此，答案是 $x=2$。

[当前答案]
x = 2

[正向验证流程]
forward_verify_real_solution_set(answer):
    ans := answer

    S_true := solve(x^2 = 4 over real numbers)

    S_ans := solution set represented by ans

    正向验证通过当且仅当 S_ans = S_true.

[正向验证结果]
False

[正向验证执行记录]
当前答案是 x = 2。

正向验证流程计算 x^2 = 4 的完整实数解集：
<tool_flow>
Let x be a real symbolic variable.
Solve x^2 = 4 over the real numbers.
Return the complete solution set.
</tool_flow>

PythonInterpreterTool 的执行结果给出 {-2, 2}。

当前答案只表示 {2}。由于 {2} 不是完整解集 {-2, 2}，当前答案不完整。

[验证结果]
执行的最终结果是：
<result>False</result>""",

    "output_case_2": r"""当前解答不完整。题目要求给出 \(x^2=4\) 的所有实数解。对完整实数解集的正向计算给出 \(\{-2,2\}\)，但当前答案只给出 \(x=2\)。解答应包含两个实数解。""",

    "input_template": r"""从正向验证流程的失败执行过程中提取关键信息，并将其转换为用于修订解答的自然语言反馈。

[问题]
{query}

[当前解答]
{response}

[当前答案]
{result}

[正向验证流程]
{execute_code}

[正向验证结果]
{execute_result}

[正向验证执行记录]
{execute_content}
"""
}
````

## 10. `code_refine_dict`（A14：正向验证流程修订器）

来源：`upstream/progco/MATH/progco_infer_main.py`

````python
code_refine_dict = {
    "system_prompt": r"""你的任务是反思当前最新解答对应的正向验证流程是否需要修订。

[正向验证简介]
正向数学验证流程从原题条件和当前解答回复出发，检查当前解答是否能从题目陈述中被正确且完整地推出。

正向验证流程应当：
1. 检查当前解答回复。
2. 识别当前回复中的重要数学主张。
3. 从题目条件验证这些主张。
4. 验证最终答案是否等于题目要求的完整目标对象。
5. 保留定义域和完备性要求，例如所有实数解、所有整数情形、完整计数、全局极值和完整集合。

[重要上下文]
你将得到：
- [问题]：原始题目。
- [当前解答]：系统当前保留的最新解答回复。这可能是回复修订器产生的修订回复 y'。
- [当前答案]：从最新解答回复中抽取出的答案。
- [初始正向验证流程]：先前的正向验证流程 V_f。
- [正向验证过程]：先前失败的执行记录 R_f。

[正向验证过程] 可能是在采用最新解答回复之前产生的。因此：
- 使用 R_f 作为前一次正向验证哪里出错的证据。
- 但要为当前最新解答修订新的正向验证流程，而不只是为旧回复修订。
- 输出 V_f' 应适合验证当前最新解答和当前答案。

错误可能来自：
1. 解答回复；
2. 当前答案；
3. 正向验证流程；
4. 旧正向验证流程与最新修订解答之间的不匹配。

你需要仔细分析并完成以下任务：

1. 反思初始正向验证流程：
   - 将该流程视为计算性数学推理。
   - 检查该流程是否沿正向方向验证最新当前解答。
   - 检查该流程是否包含解答使用的每个原始题目条件。
   - 检查该流程是否在重要时识别最新解答中的关键中间主张。
   - 检查该流程是否定义题目要求的完整数学目标：值、表达式、集合、谓词、函数、递推、计数、求和、最大值、最小值、解集或无限族。
   - 检查候选值和中间主张在需要时是否由完整表达式、集合、谓词或极值定义支撑。
   - 检查预期定义域是否被保留，包括正/非负、整数/实数、有限/无限、有序/无序以及所有解的区别。
   - 检查任一工具流步骤是否保留完整数学对象，例如完整集合、计数、max/min、递推、solve 目标、符号表达式或完备性检查。
   - 如果最新解答在保持相同答案的同时改变了推理过程，请修订流程，使其验证最新解答而不是旧解答。

2. 提供一个新的正向验证流程：
   - 如果初始正向验证流程对最新当前解答仍然正确，则在新的正向验证流程部分重复它。
   - 如果发现错误、不匹配或不足，则将其修订为更强的正向数学验证流程。
   - 风格应接近数学工具调用，使用表达式、集合、谓词、函数、递推、方程、不等式、极值、计数、求和和解集。
   - 使用数学工具级表达式，例如 count(V)、max(V)、min(V)、sum(V)、solve(...)、simplify(...)、factor(...) 和递推求值。
   - 使用 ASCII 数学记号：x in S, CartesianProduct(A, B), expr mod n = r, count(V), max(V), and V is nonempty。

当需要可靠计算来诊断正向验证流程时，在 [Reflection] 部分写一个本地 <tool_flow>...</tool_flow>。
工具调用成功后，不得再次进行等价调用；调用失败时，应根据错误信息调整工具流后再重试。

<tool_flow> 应保留与正向验证流程相关的数学对象，例如完整集合、计数、最大值、最小值、递推、符号表达式或完备性条件。

工具结果返回后，继续反思，并按要求格式生成最终输出：
[Reflection]
...
[New Forward Verification Flow]
...

请严格按以下格式输出你的分析和修订：

[Reflection]
{详细反思过程}

[New Forward Verification Flow]
{基于反思得到的完整正向验证流程}""",

    "input_case_1": r"""反思初始正向验证流程，并为最新当前解答提供一个新的正向验证流程。

[问题]
在糖果店，一块巧克力棒的价格是 $c$ 美元，一块香草棒比巧克力棒贵 $2$ 美元。Jamie 买了一块巧克力棒和三块香草棒，Kevin 买了五块巧克力棒。用 $c$ 表示，Jamie 和 Kevin 在糖果店一共花了多少钱？

[当前解答]
巧克力棒价格为 \(c\)，香草棒价格为 \(c+2\)。Jamie 买了一块巧克力棒和三块香草棒，所以 Jamie 花费 \(c+3(c+2)=4c+6\)。Kevin 买了五块巧克力棒，所以 Kevin 花费 \(5c\)。因此，总花费是 \((4c+6)+5c=9c+6\)。所以答案是 \(9c+6\)。

[当前答案]
9c + 6

[初始正向验证流程]
forward_verify_total_spent(total_spent):
    # 正向验证思路：
    # 1. 从原始题目条件和当前解答回复出发。
    # 2. 检查当前回复中关于 Jamie 花费的关键中间主张。
    # 3. 从原始条件计算完整目标总花费。
    # 4. 验证当前最终答案是否等于完整目标总花费。

    ans := total_spent

    v(c) := c + 2

    J_claim(c) := 3*c + 6
    J_true(c) := c + 3*v(c)
    Check simplify(J_true(c) - J_claim(c)) = 0.

    K_true(c) := 5*c

    T(c) := J_true(c) + K_true(c)

    D(c) := simplify(T(c) - ans)

    正向验证通过当且仅当二者都成立：
    1. simplify(J_true(c) - J_claim(c)) = 0 identically;
    2. D(c) = 0 identically.

[正向验证过程]
前一次执行失败，因为流程检查 J_claim(c) := 3c + 6，而从原始条件进行正向计算得到 J_true(c) := c + 3(c+2) = 4c + 6。simplify(J_true(c) - J_claim(c)) 的工具结果是 c，因此前一个流程返回 False。

[验证结果]
<result>False</result>""",

    "output_case_1": r"""[Reflection]
最新当前解答与初始正向验证流程中编码的主张不同。最新解答正确说明 Jamie 花费 \(c+3(c+2)=4c+6\)，而初始验证流程仍在检查旧的中间主张 \(J_claim(c)=3c+6\)。

前一次失败记录是有用证据，但新的正向验证流程应验证最新解答。因此，应将 Jamie 花费的中间主张更新为 \(J_claim(c)=4c+6\)。最终目标总花费仍应从原始题目条件计算，并与当前答案 \(9c+6\) 比较。

[New Forward Verification Flow]
forward_verify_total_spent(total_spent):
    # 正向验证思路：
    # 1. 从原始题目条件和最新当前解答出发。
    # 2. 根据原始条件定义香草棒价格。
    # 3. 检查最新当前回复中关于 Jamie 花费的关键中间主张。
    # 4. 根据原始条件检查 Kevin 的花费。
    # 5. 从原始条件计算完整目标总花费。
    # 6. 验证当前最终答案是否等于完整目标总花费。

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

    正向验证通过当且仅当全部成立：
    1. simplify(J_true(c) - J_claim(c)) = 0 identically;
    2. simplify(K_true(c) - K_claim(c)) = 0 identically;
    3. D(c) = 0 identically.""",

    "input_case_2": r"""反思初始正向验证流程，并为最新当前解答提供一个新的正向验证流程。

[问题]
Amber、Micah 和 Ahito 总共跑了 52 英里。Amber 跑了 8 英里。Micah 跑的是 Amber 的 3.5 倍。Ahito 跑了多少英里？

[当前解答]
Amber 跑了 8 英里。Micah 跑了 \(3.5 \times 8 = 28\) 英里。由于三个人总共跑了 52 英里，Ahito 跑了 \(52 - 8 - 28 = 16\) 英里。因此答案是 \(\boxed{16}\)。

[当前答案]
16

[初始正向验证流程]
forward_verify_ahito_miles(ahito_miles):
    # 正向验证思路：
    # 1. 从原始距离条件出发。
    # 2. 根据 Amber 的距离计算 Micah 的距离。
    # 3. 根据总距离条件计算 Ahito 的距离。
    # 4. 验证当前答案是否等于计算值。

    ans := ahito_miles
    total := 52
    amber := 8
    micah := 3.5 * amber

    H := total - amber - micah

    正向验证通过当且仅当 ans = H。

[正向验证过程]
该流程计算 micah = 3.5*8 = 28，且 H = 52 - 8 - 28 = 16。由于 ans=16，验证结果为 True。

[验证结果]
<result>True</result>""",

    "output_case_2": r"""[Reflection]
初始正向验证流程准确地沿正向方向检查了最新解答。它从原始条件出发，根据 Amber 的距离计算 Micah 的距离，然后根据总距离计算 Ahito 的距离。它将计算值与当前答案比较。不需要修订。

[New Forward Verification Flow]
forward_verify_ahito_miles(ahito_miles):
    # 正向验证思路：
    # 1. 从原始距离条件出发。
    # 2. 根据 Amber 的距离计算 Micah 的距离。
    # 3. 根据总距离条件计算 Ahito 的距离。
    # 4. 验证当前答案是否等于计算值。

    ans := ahito_miles
    total := 52
    amber := 8
    micah := 3.5 * amber

    H := total - amber - micah

    正向验证通过当且仅当 ans = H.""",

    "input_template": r"""反思初始正向验证流程，并为最新当前解答提供一个新的正向验证流程。
（注意：如果该正向验证流程对最新当前解答是正确的，请在新的正向验证流程部分重复它。）

[问题]
{query}

[当前解答]
{response}

[当前答案]
{result}

[初始正向验证流程]
{verify_code}

[正向验证过程]
{verify_process}
"""
}
````

## 11. `refine_dict`（A15：正向反馈回复修订器）

来源：`upstream/progco/MATH/progco_infer_main.py`

````python
refine_dict = {
    "system_prompt": r"""你的任务是根据正向验证反馈修订一个数学解答。

给定一道题目 [Problem]、一个当前解答回复 [Initial Solution]，以及来自正向验证流程的反馈 [Feedback]，请仔细比较解答和反馈。

正向验证检查当前解答是否能从原始题目条件中被正确且完整地推出。它可能指出错误的中间主张、错误算术、无效公式、遗漏情况、不完整解集、错误定义域、错误计数或错误最终目标对象。

你需要完成以下任务：

1. 反思初始解答：
   - 比较解答和反馈。
   - 逐步检查解答的正向推理。
   - 判断反馈是否揭示了解答中的错误，或正向验证流程中可能的弱点。
   - 如果反馈正确，请修订解答。
   - 如果解答正确且反馈是由较弱的验证流程导致的，请保留解答并说明原因。

2. 提供一个新解答：
   - 如果初始解答正确，使用原始解答过程。
   - 如果发现错误或不足，请重写解答，使其能从原始题目条件中正确推出答案。
   - 新解答必须是完整的解答回复。
   - 最终答案必须清楚陈述。

当修订解答需要可靠的构造性计算时，你可以在 [Reflection] 部分写一个本地 <tool_flow>...</tool_flow>。
工具调用成功后，不得再次进行等价调用；调用失败时，应根据错误信息调整工具流后再重试。

<tool_flow> 应从原始题目条件和正向验证反馈中计算候选答案或关键数学对象。

在构造性工具流中使用 ASCII 数学记号：
- x in S
- CartesianProduct(A, B)
- expr mod n = r
- count(V)
- max(V), min(V), sum(V)
- V is nonempty

用于构造性解答修复的工具流示例：
<tool_flow>
Let c be a symbolic variable.
Compute T(c) = c + 3*(c + 2) + 5*c.
Return simplify(T(c)).
</tool_flow>

工具结果返回后，继续反思，并按要求格式生成最终输出：
[Reflection]
...
[New Solution]
...

注意：
- 一定要提供新的解答部分。
- 如果反馈不要求数学修正，不要只是改写或润色解答。保持原数学解答不变，并明确说明不需要数学修订。
- 除非反馈显示被检查的关键主张或完整目标对象存在数学不一致，否则不要更改最终答案。
- 当反馈指出某个中间结论错误时，不要只修补这一行；需要从该错误点开始，重新推导所有依赖它的后续步骤，并确保最终答案来自修正后的推理过程。
- 如果你修订了解答，请明确数学变化。不要在数学内容不变时输出一个只是表面不同的解答。

请严格按以下格式输出你的分析和修订：

[Reflection]
{{详细反思过程}}

[New Solution]
{{基于反思得到的完整解答过程}}""",

    "input_case_1": r"""反思初始解答，并提供一个新解答。

[问题]
在糖果店，一块巧克力棒的价格是 $c$ 美元，一块香草棒比巧克力棒贵 $2$ 美元。Jamie 买了一块巧克力棒和三块香草棒，Kevin 买了五块巧克力棒。用 $c$ 表示，Jamie 和 Kevin 在糖果店一共花了多少钱？

[初始解答]
香草棒的价格是 c+2。Jamie 花费 c + 3(c+2) = 3c + 6 美元。Kevin 花费 5c 美元。因此，总花费是 3c + 6 + 5c = 8c + 6。

[反馈]
当前解答在 Jamie 花费的正向计算中包含错误。它声称 c + 3(c+2) = 3c + 6，但正确化简是 c + 3(c+2) = c + 3c + 6 = 4c + 6。因为 Jamie 的花费错误，最终总额 8c+6 也错误。解答应重新计算 Jamie 的花费，然后重新计算 Jamie 和 Kevin 的总花费。""",

    "output_case_1": r"""[Reflection]
反馈是正确的。该解答错误地化简了 Jamie 的花费。由于 Jamie 买了一块巧克力棒和三块香草棒，且每块香草棒价格为 \(c+2\)，Jamie 的花费是：
\[
c + 3(c+2) = c + 3c + 6 = 4c + 6.
\]
Kevin 的花费是 \(5c\)。因此总花费是：
\[
(4c+6) + 5c = 9c + 6.
\]
所以最终答案应从 \(8c+6\) 修订为 \(9c+6\)。

[New Solution]
巧克力棒价格为 \(c\) 美元，香草棒价格为 \(c+2\) 美元。

Jamie 买了一块巧克力棒和三块香草棒，所以 Jamie 花费
\[
c + 3(c+2) = c + 3c + 6 = 4c + 6.
\]

Kevin 买了五块巧克力棒，所以 Kevin 花费
\[
5c.
\]

因此，Jamie 和 Kevin 的总花费是
\[
(4c+6) + 5c = 9c + 6.
\]

所以，总花费为
\[
\boxed{9c+6}.
\]""",

    "input_case_2": r"""反思初始解答，并提供一个新解答。

[问题]
在实数范围内解 $x^2 = 4$。

[初始解答]
解 $x^2=4$ 得到 $x=2$。因此，答案是 $x=2$。

[反馈]
当前解答不完整。题目要求给出 \(x^2=4\) 的所有实数解。对完整实数解集的正向计算给出 \(\{-2,2\}\)，但当前答案只给出 \(x=2\)。解答应包含两个实数解。""",

    "output_case_2": r"""[Reflection]
反馈是正确的。方程 \(x^2=4\) 在实数范围内有两个解，因为 \(2^2=4\) 且 \((-2)^2=4\)。初始解答只给出 \(x=2\)，所以遗漏了解 \(x=-2\)。

[New Solution]
我们求解
\[
x^2 = 4.
\]
在实数范围内开平方得到
\[
x = \pm 2.
\]
因此完整实数解集是
\[
\boxed{x=-2,\ 2}.
\]""",

    "input_template": r"""反思初始解答，并提供一个新解答。

[问题]
{query}

[初始解答]
（请记住，反馈并不总是正确的。）
{old_response}

[反馈]
{feedback}
"""
}
````

## 12. A16：答案差异控制器

A16 在 `ProgCoAgent.inference` 内部实现。

它不使用单独的提示词。

A16 只比较抽取出的旧答案 `a` 和修订答案 `a'`。

决策规则：

```text
If a' != a:
    update y <- y'
    update a <- a'
    regenerate V_f through A11 in the next turn

If a' == a:
    update y <- y'
    keep a unchanged
    use A14 to refine the current V_f into V_f'
    run A12 with the latest y' and V_f' in the next turn
```

这使 A16 接近 2-2-1-1 中原始 A6 的设计，同时将“答案相同”分支适配到正向验证场景。
## 共享提示词常量
## 13. `branch_math_rewrite_dict`（A17：ToolFlow 实现器）

来源：`repro/scripts/math_hard_test_parallel_python_tool.py`

````python
branch_math_rewrite_dict = {
    "system_prompt": r"""你是 MathTool 流实现专家。

一位验证专家在 <tool_flow>...</tool_flow> 内写了一个数学工具流。你的任务是在执行前，将该工具流实现为可靠的可执行 Python 代码。

要求：
1. 保留工具流的数学对象和验证意图。
2. 尽可能使用工具级操作：集合构造、count、sum、max、min、argmax、argmin、solve、simplify、factor、递推求值、DP 求值、非空性检查和反例搜索。
3. 优先使用精确符号、有理数、整数或穷举计算，而不是浮点近似。
4. 只实现输入中提供的数学工具流。
5. 精确保留验证目标。
6. 只使用工具流中陈述的假设、常数、变量和条件。
7. 对集合、计数、极值、递推、符号或完备性任务，实现工具流描述的完整数学对象。
8. 使用 ASCII 数学记号：
   - x in S
   - CartesianProduct(A, B)
   - expr mod n = r
   - count(V)
   - max(V), min(V), sum(V)
   - V is nonempty
9. 只输出可执行 Python 代码，包含完整的 execute() 函数和最后的 execute() 调用。
10. 使用下列批准数学库进行本地确定性计算。

工具偏好：
- 使用 sympy 进行精确符号代数、化简、展开、因式分解、平方根、根式、多项式表达式、符号等式检查、整数分解、约数、阶乘、二项式系数、方程求解、结式、Groebner 基和精确方程变形。
- 对符号等式检查，优先化简差值，例如 simplify(lhs - rhs) == 0，而不是直接比较未化简表达式。
- 使用 itertools 进行有限穷举枚举、笛卡尔积、排列、组合以及小有限域上的计数。
- 当计算涉及分数或比例时，使用 fractions.Fraction 进行精确有理数运算。尽可能从整数或字符串构造 Fraction 值，而不是从浮点数构造。
- 当任务涉及十进制算术和四舍五入规则时，使用 decimal.Decimal。从字符串而不是浮点数构造 Decimal 值。除非题目另有说明，使用 ROUND_HALF_UP 表示标准学校式四舍五入。
- 当不需要符号代数时，使用 math 进行简单整数工具计算，例如 gcd、lcm、factorial、comb、perm 和 isqrt。
- 只有在显式整数、布尔或有限逻辑条件约束满足问题中，且直接枚举或 sympy 不合适时，才使用 z3。
- 只有当任务明确需要数值近似或数值优化，并且精确方法不适用时，才使用 scipy。
""",

    "input_case_1": r"""[原始 PythonInterpreterTool 工具流]
Evaluate V = {d in positive integers : d divides 1200 and d is odd}.
Return V and max(V).""",

    "output_case_1": r"""def execute():
    import sympy as sp
    V = [d for d in sp.divisors(1200) if d > 0 and d % 2 == 1]
    return {"V": V, "max": max(V)}
execute()""",

    "input_case_2": r"""[原始 PythonInterpreterTool 工具流]
Evaluate V = {(A,B) in CartesianProduct({0,1,2,3,4,5,6,7,8,9}, {0,1,2,3,4,5,6,7,8,9}) :
    (2 + 5 + A + B) mod 9 = 0}.
Return V and count(V).""",

    "output_case_2": r"""def execute():
    import itertools
    D = range(10)
    V = [(A, B) for A, B in itertools.product(D, D) if (2 + 5 + A + B) % 9 == 0]
    return {"V": V, "count": len(V)}
execute()""",

    "input_case_3": r"""[原始 PythonInterpreterTool 工具流]
Let c be a symbolic variable.
ans := 8*c + 6
v(c) := c + 2
J(c) := c + 3*v(c)
K(c) := 5*c
T(c) := J(c) + K(c)
Return simplify(T(c) - ans).""",

    "output_case_3": r"""def execute():
    import sympy as sp
    c = sp.symbols("c")
    ans = 8 * c + 6
    v = c + 2
    J = c + 3 * v
    K = 5 * c
    T = J + K
    return sp.simplify(T - ans)
execute()""",

    "input_case_4": r"""[原始 PythonInterpreterTool 工具流]
Given ans = 90, leg1 = 9, S = ans - leg1, and h(l) = S - l.
Evaluate V(ans) = {l in positive integers :
    h(l) in positive integers
    and abs(h(l) - l) = 1
    and leg1^2 + l^2 = h(l)^2}.
Return V(ans) and whether V(ans) is nonempty.""",

    "output_case_4": r"""def execute():
    ans = 90
    leg1 = 9
    S = ans - leg1
    V = []
    for l in range(1, S):
        h = S - l
        if h > 0 and abs(h - l) == 1 and leg1**2 + l**2 == h**2:
            V.append(l)
    return {"V": V, "nonempty": len(V) > 0}
execute()""",

    "input_case_5": r"""[原始 PythonInterpreterTool 工具流]
Let dp_i(r) be the number of valid arrangements of the first i positions whose suffix has exactly r consecutive occupied positions, where r in {0,1,2}.
Use:
dp_0(0)=1, dp_0(1)=dp_0(2)=0.
dp_{i+1}(0)=dp_i(0)+dp_i(1)+dp_i(2)
dp_{i+1}(1)=dp_i(0)
dp_{i+1}(2)=dp_i(1)
For n = 10, return N = dp_n(0)+dp_n(1)+dp_n(2).""",

    "output_case_5": r"""def execute():
    n = 10
    dp = {0: 1, 1: 0, 2: 0}
    for _ in range(n):
        dp = {
            0: dp[0] + dp[1] + dp[2],
            1: dp[0],
            2: dp[1],
        }
    return {"dp": dp, "N": dp[0] + dp[1] + dp[2]}
execute()""",

    "input_case_6": r"""[原始 PythonInterpreterTool 工具流]
Compute CE = sqrt(4^2 + 9^2).
Return CE exactly and rounded to the nearest tenth.""",

    "output_case_6": r"""def execute():
    import sympy as sp
    CE = sp.sqrt(4**2 + 9**2)
    return {"exact": sp.simplify(CE), "rounded_nearest_tenth": round(float(CE), 1)}
execute()""",

    "input_case_7": r"""[原始 PythonInterpreterTool 工具流]
Let ans = 1050*sqrt(2).
Return simplify(ans^2).""",

    "output_case_7": r"""def execute():
    import sympy as sp
    ans = 1050 * sp.sqrt(2)
    return sp.simplify(ans**2)
execute()""",

    "input_case_8": r"""[原始 PythonInterpreterTool 工具流]
O = {A, B, C}.
r(A) = round_to_nearest_tenth(14.45001)
r(B) = round_to_nearest_tenth(14.54999)
r(C) = round_to_nearest_tenth(14.44999)
m = min {r(o) for o in O}
V = {o in O : r(o) = m}
Return r, m, and V.""",

    "output_case_8": r"""def execute():
    from decimal import Decimal, ROUND_HALF_UP
    values = {
        "A": Decimal("14.45001"),
        "B": Decimal("14.54999"),
        "C": Decimal("14.44999"),
    }
    r = {
        key: value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        for key, value in values.items()
    }
    m = min(r.values())
    V = [key for key, value in r.items() if value == m]
    return {"r": r, "m": m, "V": V}
execute()""",

    "input_template": r"""[原始 PythonInterpreterTool 工具流]
{original_code}"""
}
````

## 14. `extract_prompt`（A8：答案抽取器）

来源：`upstream/progco/MATH/extract_answer.py`

````python
extract_prompt="你的任务是从给定题目和解答过程中准确抽取最终答案。请仔细阅读整个问题和解答过程，以找到最终答案。\n\n重要说明：\n1. 只抽取最终答案，不包含单位或其他文字解释。\n2. 不要输出任何额外内容，例如 \\boxed{{}} 包裹等。\n3. 对于多个答案，使用列表格式输出。\n\n答案可以有多种形式，包括但不限于：\n\n1. 整数或小数：10, -1.5\n2. 分数：\\frac{{1}}{{2}}, -\\frac{{3}}{{4}}\n3. 文本答案：even, odd\n4. 平方根表达式：\\sqrt{{10}}, \\sqrt[3]{{8}}\n5. 常数：π, e\n6. 代数表达式：a+1, y = x + 1\n7. 选择题选项：A, B, C, D\n8. 矩阵（使用 LaTeX 格式）：\n   \\begin{{bmatrix}}\n   1 & 2 \\\\\n   3 & 4\n   \\end{{bmatrix}}\n\n题目如下：\n{query}\n\n解答如下：\n{response}\n\n再次确认，你的输出只能包含最终答案，不要包含任何额外解释或格式。"
````

## 15. 重复成功 Tool Flow 的运行时后续消息

来源：`repro/scripts/math_hard_test_parallel_python_tool.py`

这些内容不属于静态系统提示词。同一个 `<tool_flow>` 已经成功完成后，如果模型再次请求完全相同的 `<tool_flow>`，控制器会返回缓存结果，并追加以下一条中性的用户后续消息。重复请求仍计入现有的 PythonInterpreterTool 软限制和硬限制。

### 第一次重复请求

````text
这个完全相同的 <tool_flow> 已经成功执行过。

缓存的执行结果是：
{successful_tool_result}

请直接使用该结果继续当前验证，无需重新执行相同的 <tool_flow>。
````

### 第二次及后续重复请求

````text
上述 <tool_flow> 已经有可用的成功执行结果。

缓存的执行结果仍然是：
{successful_tool_result}

请沿用该结果继续当前验证，不必再次运行相同的 <tool_flow>。
````
