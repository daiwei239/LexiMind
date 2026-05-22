"""Test DeepSeek latency with ~1200 Chinese characters input."""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.generation.answer_agent import DeepSeekClient
from app.settings import Settings

# 约 1200 字的法律咨询场景正文（用于测输入长度对延迟的影响）
INPUT_1200 = """
我是一名消费者，于2023年3月通过某汽车金融服务公司办理了车辆分期付款业务，车辆总价为18.6万元，
首付3.6万元，剩余15万元分36期偿还，每月月供约4800元。合同约定车辆登记在我名下，但在贷款结清前，
机动车登记证书由分期公司保管，并办理了抵押权登记。自2024年4月起，我按照合同约定在保险公司购买
交强险和商业险，保费每年约6500元。2025年续保时，我发现分期公司指定的合作保险渠道报价明显高于
我在其他正规渠道获得的报价，相差约1200元至1500元，因此我选择在外部保险公司完成续保，并按时将
保单复印件发送至分期公司邮箱备案。

2025年6月，分期公司客服人员电话通知我，称我未在其指定渠道购买保险，构成违约，要求我补缴所谓
"渠道服务费"2000元，并威胁否则将派人扣押车辆。我明确告知对方，我一直按时偿还月供，从未逾期，
且保险合同的真实目的是保障车辆风险，我已依法投保，不应被强制绑定销售。对方随后发送书面通知，
声称依据分期合同第14条、第15条，其有权收回车辆、行使抵押权。我查阅合同后发现，相关条款表述
较为笼统，未明确"必须在分期公司指定渠道投保"的措辞，也未载明未在指定渠道投保即构成根本违约的
具体后果。

2025年7月某日，分期公司指派人员在我居住小区附近，使用备用钥匙将车辆开走，并留下纸条称因保险
违约暂扣车辆。我当场报警，派出所出警后告知属于民事纠纷，建议协商或起诉。我随后向市场监管部门
投诉，并咨询律师。目前我的诉求包括：第一，确认分期公司无权仅因未在其渠道购买保险而扣押车辆；
第二，如车辆已被扣押，要求返还车辆并赔偿因此产生的交通费、误工费等损失；第三，如合同条款确
有争议，希望了解是否可以通过诉讼确认条款无效或不予适用。请结合民法典、保险法、消费者权益
保护法等相关规定，分析我的处境及可行的维权路径。

补充说明：车辆目前仍被扣留，我无法正常上下班，日常需打车出行，已产生额外交通费用。分期公司曾口头表示，
只要补缴两千元即可还车，但未出具书面和解协议。我担心签字后被认定为承认违约。另，我查询征信报告
显示贷款状态仍为正常还款，未见逾期记录。我希望了解除诉讼外，是否可向消费者协会、银保监或市场
监管部门继续投诉，以及报警记录能否作为后续诉讼的证据使用。以上事实如有需要我可提供
分期合同、还款记录、保单、报警回执、微信聊天记录等证据材料。
""".strip()

# 补齐到约 1200 字（测试输入长度）
while len(INPUT_1200) < 1200:
    INPUT_1200 += "以上为本案补充说明。"


def main() -> None:
    settings = Settings.from_env(ROOT / ".env")
    char_count = len(INPUT_1200)
    print(f"输入字数（字符数）: {char_count}")
    print(f"model: {settings.deepseek_model}")
    print(f"thinking_enabled: {settings.deepseek_thinking_enabled}")
    print("-" * 60)

    client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        thinking_enabled=settings.deepseek_thinking_enabled,
    )

    times: list[float] = []
    answers: list[str] = []
    for i in range(3):
        t0 = time.perf_counter()
        answer = client.generate(
            [
                {
                    "role": "user",
                    "content": (
                        f"请阅读以下约{char_count}字的案情描述，用简洁要点回复"
                        f"（不超过500字）：是否可能有权扣车、建议维权步骤。\n\n{INPUT_1200}"
                    ),
                }
            ]
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)
        answers.append(answer)
        out_chars = len(answer)
        print(f"第{i + 1}轮: {elapsed_ms:.0f} ms | 回复 {out_chars} 字")
        print(f"  开头: {answer[:120].replace(chr(10), ' ')}...")

    print("-" * 60)
    print(
        f"平均: {statistics.mean(times):.0f} ms  "
        f"min={min(times):.0f} max={max(times):.0f}"
    )
    print(f"回复字数: {[len(a) for a in answers]}")


if __name__ == "__main__":
    main()
