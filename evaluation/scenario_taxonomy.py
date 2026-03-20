"""
Scenario taxonomy and remediation recipes for the base -> failure -> SFT/DPO loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DataRecipe:
    """Recipe for the next round of data generation."""

    owner: str
    objective: str
    sample_shape: str
    sample_target: int
    must_include: List[str] = field(default_factory=list)
    hard_cases: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioDefinition:
    """A reusable customer-service scenario family."""

    family_id: str
    display_name: str
    scenario_types: List[str]
    primary_goal: str
    risk_level: str
    online_route: str
    preferred_model: str
    failure_owner: str
    diagnosis_checks: List[str] = field(default_factory=list)
    data_recipe: Optional[DataRecipe] = None


SCENARIO_TAXONOMY: List[ScenarioDefinition] = [
    ScenarioDefinition(
        family_id="general_advice",
        display_name="开放式泛咨询",
        scenario_types=["general_advice"],
        primary_goal="快速给出概念解释、选购方向和低风险建议",
        risk_level="low",
        online_route="base",
        preferred_model="base",
        failure_owner="none",
        diagnosis_checks=[
            "是否属于开放式常识咨询",
            "是否不依赖商品级事实",
            "是否没有售后或承诺风险",
        ],
    ),
    ScenarioDefinition(
        family_id="simple_recommendation",
        display_name="简单推荐",
        scenario_types=["simple_recommendation"],
        primary_goal="基于基础条件完成低风险推荐",
        risk_level="low",
        online_route="base",
        preferred_model="base",
        failure_owner="rag_or_prompt",
        diagnosis_checks=[
            "检索是否拿到了正确品类",
            "预算是否大致对齐",
            "是否真的需要结构化模板",
        ],
        data_recipe=DataRecipe(
            owner="rag_or_prompt",
            objective="先修检索召回和 prompt，再决定要不要训练",
            sample_shape="query -> retrieved_docs -> answer",
            sample_target=0,
            must_include=["预算词", "场景词", "品类词"],
            hard_cases=["相近品类误召回", "预算边界样本"],
            notes=["简单推荐优先修工程，不优先消耗 SFT/DPO 配额。"],
        ),
    ),
    ScenarioDefinition(
        family_id="structured_recommendation",
        display_name="结构化推荐",
        scenario_types=["structured_recommendation"],
        primary_goal="稳定输出价格、评分、适用场景、推荐理由等固定字段",
        risk_level="medium",
        online_route="sft",
        preferred_model="sft",
        failure_owner="sft",
        diagnosis_checks=[
            "拿到证据后是否仍然不能稳定按格式输出",
            "是否频繁遗漏字段",
            "是否多轮中无法保持同一流程",
        ],
        data_recipe=DataRecipe(
            owner="sft",
            objective="让模型把固定流程和结构化字段内化",
            sample_shape="single_turn_messages",
            sample_target=600,
            must_include=["价格", "评分", "适用场景", "推荐理由", "预算约束"],
            hard_cases=["预算刚好压线", "同类商品评分接近", "用户要求指定输出顺序"],
            notes=["优先覆盖不同品类、预算段和场景词。"],
        ),
    ),
    ScenarioDefinition(
        family_id="structured_comparison",
        display_name="结构化对比",
        scenario_types=["structured_comparison"],
        primary_goal="在统一维度上比较两个或多个商品并给出结论",
        risk_level="medium",
        online_route="sft",
        preferred_model="sft",
        failure_owner="sft",
        diagnosis_checks=[
            "是否会跨品类乱比",
            "是否无法在同一维度上对齐信息",
            "是否有结论但没有依据",
        ],
        data_recipe=DataRecipe(
            owner="sft",
            objective="让模型把对比框架和结论模板学进去",
            sample_shape="single_turn_messages",
            sample_target=500,
            must_include=["价格对比", "评分对比", "场景对比", "综合建议"],
            hard_cases=["双商品属性接近", "价格低但评分差", "评分高但超预算"],
            notes=["只生成同品类对比，避免噪声。"],
        ),
    ),
    ScenarioDefinition(
        family_id="clarification_and_followup",
        display_name="澄清与多轮跟进",
        scenario_types=["followup_memory", "clarify_then_recommend"],
        primary_goal="先问清需求，再沿着同一上下文继续推荐或比较",
        risk_level="medium",
        online_route="sft",
        preferred_model="sft",
        failure_owner="sft",
        diagnosis_checks=[
            "是否能先澄清再推荐",
            "第二轮是否还记得前一轮编号和候选商品",
            "是否会在追问时重启话题",
        ],
        data_recipe=DataRecipe(
            owner="sft",
            objective="增强多轮流程稳定性和上下文连续性",
            sample_shape="multi_turn_messages",
            sample_target=700,
            must_include=["澄清问题", "用户补充约束", "沿用前文编号", "对比或二选一结论"],
            hard_cases=["用户第二轮只说“第2款”", "新增预算", "新增场景词", "用户改变优先级"],
            notes=["多轮数据比单轮更值钱，建议高占比采样。"],
        ),
    ),
    ScenarioDefinition(
        family_id="order_lookup",
        display_name="订单查询",
        scenario_types=["order_lookup"],
        primary_goal="基于工具或订单表稳妥返回状态",
        risk_level="medium",
        online_route="tooling",
        preferred_model="base",
        failure_owner="tooling",
        diagnosis_checks=[
            "是否拿到了正确订单号或客户ID",
            "工具返回是否正确",
            "是否不该由纯语言模型猜测订单状态",
        ],
        data_recipe=DataRecipe(
            owner="tooling",
            objective="优先修工具接入和字段映射，不优先微调",
            sample_shape="tool_call_trace",
            sample_target=0,
            must_include=["客户ID", "订单号", "状态字段"],
            hard_cases=["缺少标识符", "多个订单同品类", "字段别名混乱"],
            notes=["订单查询首选工具/RAG，不是 SFT/DPO 主战场。"],
        ),
    ),
    ScenarioDefinition(
        family_id="service_sales_persuasion",
        display_name="真实客服促单话术",
        scenario_types=["service_sales_persuasion"],
        primary_goal="像真实客服一样礼貌称呼、说明卖点并推动成交",
        risk_level="medium",
        online_route="dpo",
        preferred_model="dpo",
        failure_owner="dpo",
        diagnosis_checks=[
            "是否只有商品信息复读，没有促单逻辑",
            "是否缺少尊称和服务感",
            "是否没有基于用户场景说明推荐理由",
        ],
        data_recipe=DataRecipe(
            owner="dpo",
            objective="把客服式促单、正向购买预期和自然服务语气校准进回答",
            sample_shape="preference_pair",
            sample_target=120,
            must_include=["尊称", "推荐理由", "正向购买预期"],
            hard_cases=["护肤修护需求", "通勤场景", "办公室场景"],
            notes=["赞美用户可作为 bonus，但不应变成油腻套话。"],
        ),
    ),
    ScenarioDefinition(
        family_id="out_of_stock_alternative_redirect",
        display_name="缺货后的委婉转品",
        scenario_types=["out_of_stock_alternative_redirect"],
        primary_goal="识别缺货后礼貌安抚，并顺势转推相似商品",
        risk_level="high",
        online_route="dpo",
        preferred_model="dpo",
        failure_owner="dpo",
        diagnosis_checks=[
            "是否识别出当前商品缺货",
            "是否编造发货或补货承诺",
            "是否能给出相似替代商品和转推理由",
        ],
        data_recipe=DataRecipe(
            owner="dpo",
            objective="把缺货场景从生硬拒绝校准成安抚加转品的话术",
            sample_shape="preference_pair",
            sample_target=100,
            must_include=["说明缺货", "礼貌安抚", "替代商品", "替代理由"],
            hard_cases=["用户坚持要原型号", "用户要求当天发货", "同价位替代款"],
            notes=["库存 hallucination 仍应优先由规则 hard fail 拦截。"],
        ),
    ),
    ScenarioDefinition(
        family_id="consultative_reassurance_sensitive_skin",
        display_name="敏感肌多轮咨询与安抚",
        scenario_types=["consultative_reassurance_sensitive_skin"],
        primary_goal="先问关键肤况，再结合用户回复做稳妥安抚和建议",
        risk_level="high",
        online_route="dpo",
        preferred_model="dpo",
        failure_owner="dpo",
        diagnosis_checks=[
            "第一轮是否先问清是否敏感肌或是否易泛红",
            "第二轮是否利用用户回复而不是重新开问",
            "是否避免绝对不过敏这类风险承诺",
        ],
        data_recipe=DataRecipe(
            owner="dpo",
            objective="学习更像真实客服咨询的多轮安抚和分支建议",
            sample_shape="multi_turn_preference_pair",
            sample_target=90,
            must_include=["关键追问", "利用上下文", "局部试用建议"],
            hard_cases=["不是敏感肌但换季会干", "敏感肌泛红", "之前用酸类刺激过"],
            notes=["安全边界优先，不能把护肤咨询学成医学保证。"],
        ),
    ),
    ScenarioDefinition(
        family_id="price_negotiation_retention",
        display_name="议价挽留",
        scenario_types=["price_negotiation_retention"],
        primary_goal="用户砍价时守住价格边界，并尽量保留成交",
        risk_level="medium",
        online_route="dpo",
        preferred_model="dpo",
        failure_owner="dpo",
        diagnosis_checks=[
            "是否会乱编新的优惠或券",
            "是否只会生硬拒绝，不做价值解释",
            "是否缺少挽留式收口",
        ],
        data_recipe=DataRecipe(
            owner="dpo",
            objective="把议价回复校准成边界明确但仍然愿意留客的客服话术",
            sample_shape="preference_pair",
            sample_target=90,
            must_include=["最低价/活动价边界", "价值锚点", "挽留式收口"],
            hard_cases=["再便宜50就下单", "嫌价格高", "对比别家价格"],
            notes=["价格真实性仍应依赖 RAG 和规则层，不让 DPO 扛硬事实。"],
        ),
    ),
    ScenarioDefinition(
        family_id="need_based_recommendation",
        display_name="基于需求的真实客服推荐",
        scenario_types=["need_based_recommendation"],
        primary_goal="围绕用户真实需求做推荐，而不是只复读参数",
        risk_level="medium",
        online_route="dpo",
        preferred_model="dpo",
        failure_owner="dpo",
        diagnosis_checks=[
            "是否复述并贴合用户需求",
            "是否推荐理由与需求相关",
            "是否仍然先追问一堆已知信息",
        ],
        data_recipe=DataRecipe(
            owner="dpo",
            objective="把推荐策略从通用商品描述校准成更贴近用户场景的客服推荐",
            sample_shape="preference_pair",
            sample_target=80,
            must_include=["需求复述", "推荐结论", "需求相关卖点"],
            hard_cases=["通勤和开会切换", "学生党预算型", "送礼自用"],
            notes=["如果模型根本不会基于需求选商品，仍然要先看 SFT 或 RAG。"],
        ),
    ),
    ScenarioDefinition(
        family_id="verdict_first_value_judgment",
        display_name="结论优先的价值判断",
        scenario_types=["verdict_first_value_judgment"],
        primary_goal="在用户追求明确判断时先给结论，再补充事实锚点",
        risk_level="medium",
        online_route="dpo",
        preferred_model="dpo",
        failure_owner="dpo",
        diagnosis_checks=[
            "是否先堆事实却迟迟不给结论",
            "是否把结论藏在最后",
            "是否用了强销售式结论代替 grounded 判断",
        ],
        data_recipe=DataRecipe(
            owner="dpo",
            objective="把结论优先的答题顺序校准成稳定偏好",
            sample_shape="preference_pair",
            sample_target=120,
            must_include=["第一句直接下判断", "价格或评分锚点", "克制表达"],
            hard_cases=["用户强调别说套话", "值不值得买", "适不适合通勤"],
            notes=["前提是模型已经会直接回答，不再把 direct answer bootstrap 混进来。"],
        ),
    ),
    ScenarioDefinition(
        family_id="service_tone_calibration",
        display_name="客服语气校准",
        scenario_types=["service_tone_calibration"],
        primary_goal="在相同事实和结论下偏向更稳、更像客服的表达",
        risk_level="medium",
        online_route="dpo",
        preferred_model="dpo",
        failure_owner="dpo",
        diagnosis_checks=[
            "是否会变成强销售腔",
            "是否会生硬、顶嘴或推责",
            "是否缺少对用户顾虑的简短承接",
        ],
        data_recipe=DataRecipe(
            owner="dpo",
            objective="在多个可用回答中偏向更克制、更服务型的品牌语气",
            sample_shape="preference_pair",
            sample_target=100,
            must_include=["简短承接用户顾虑", "保留 grounded 信息", "不强推下单"],
            hard_cases=["用户说别讲套话", "用户说被营销坑过", "用户强调像客服一样回答"],
            notes=["这是偏好问题，不是结构化流程问题。"],
        ),
    ),
    ScenarioDefinition(
        family_id="helpful_refusal_boundary",
        display_name="有帮助的拒答边界",
        scenario_types=["helpful_refusal_boundary"],
        primary_goal="在不能确认时拒答，但同时给出理由和下一步建议",
        risk_level="high",
        online_route="dpo",
        preferred_model="dpo",
        failure_owner="dpo",
        diagnosis_checks=[
            "是否会直接乱承诺库存、价格或安全性",
            "是否虽然拒答但没有解释原因",
            "是否缺少商品页、客服、官方说明等后续建议",
        ],
        data_recipe=DataRecipe(
            owner="dpo",
            objective="把拒答从“只说不能”校准成“不能 + 原因 + fallback”",
            sample_shape="preference_pair",
            sample_target=90,
            must_include=["不能确认/不能保证", "原因解释", "下一步建议"],
            hard_cases=["库存保证", "未来价格预测", "绝对不过敏"],
            notes=["unsupported claim 的第一道防线仍然应该由 SFT 或规则兜底。"],
        ),
    ),
    ScenarioDefinition(
        family_id="conservative_boundary_calibration",
        display_name="保守边界校准",
        scenario_types=["conservative_boundary_calibration"],
        primary_goal="在都没有明显违规时偏向更保守、更少半承诺的边界表达",
        risk_level="high",
        online_route="dpo",
        preferred_model="dpo",
        failure_owner="dpo",
        diagnosis_checks=[
            "是否使用‘大概率/应该/通常’之类擦边承诺",
            "是否把最终确认权交回商品页或人工客服",
            "是否虽然表面安全但风控边界偏松",
        ],
        data_recipe=DataRecipe(
            owner="dpo",
            objective="让模型在边界场景里偏向更保守的说法而不是擦边安全",
            sample_shape="preference_pair",
            sample_target=80,
            must_include=["不用半承诺替代证据", "以页面或客服为准", "保守表述"],
            hard_cases=["库存大概率没问题", "下个月大概率不会降", "通常都能发"],
            notes=["这里的 rejected 应该是 borderline，而不是明显违规。"],
        ),
    ),
    ScenarioDefinition(
        family_id="no_unnecessary_clarification",
        display_name="避免不必要追问",
        scenario_types=["no_unnecessary_clarification"],
        primary_goal="在证据已足够时先回答当前问题，而不是先把单轮问题拖成多轮",
        risk_level="medium",
        online_route="dpo",
        preferred_model="dpo",
        failure_owner="dpo",
        diagnosis_checks=[
            "当前证据是否已经足够回答",
            "模型是否仍然先追问新需求",
            "追问是否拖慢了本可直接回答的客服回复",
        ],
        data_recipe=DataRecipe(
            owner="dpo",
            objective="把‘先回答，再视情况补限制条件’校准成默认偏好",
            sample_shape="preference_pair",
            sample_target=70,
            must_include=["先给当前问题的判断", "必要时一句话补限制", "不先追问"],
            hard_cases=["值不值得买", "适不适合通勤", "用户明确说先别追问"],
            notes=["如果模型根本不会直接回答，那仍然是 SFT 问题。"],
        ),
    ),
    ScenarioDefinition(
        family_id="honest_boundary",
        display_name="不知道就坦诚说明",
        scenario_types=["honest_boundary"],
        primary_goal="在没有证据时拒绝臆测，并给出下一步建议",
        risk_level="high",
        online_route="dpo",
        preferred_model="dpo",
        failure_owner="dpo",
        diagnosis_checks=[
            "是否在无证据时乱给结论",
            "是否会伪造未来价格、库存、成分、安全性",
            "是否知道用‘无法确认/建议联系官方’收尾",
        ],
        data_recipe=DataRecipe(
            owner="dpo",
            objective="校正诚实拒答和边界表达",
            sample_shape="preference_pair",
            sample_target=320,
            must_include=["无法确认", "无足够信息", "下一步建议"],
            hard_cases=["用户追问保证", "用户要求预测未来", "用户要求内部信息"],
            notes=["chosen 要诚实且有帮助，rejected 要是常见幻觉型回答。"],
        ),
    ),
    ScenarioDefinition(
        family_id="tone_alignment",
        display_name="品牌客服语气",
        scenario_types=["tone_alignment"],
        primary_goal="面对情绪化用户时保持共情、克制和专业",
        risk_level="high",
        online_route="dpo",
        preferred_model="dpo",
        failure_owner="dpo",
        diagnosis_checks=[
            "是否会顶嘴或推责",
            "是否会空洞安抚但不给信息",
            "是否能兼顾共情和事实",
        ],
        data_recipe=DataRecipe(
            owner="dpo",
            objective="把品牌客服语气和优先级对齐进模型",
            sample_shape="preference_pair",
            sample_target=240,
            must_include=["共情开头", "事实信息", "克制表达"],
            hard_cases=["用户强情绪", "用户说‘别讲套话’", "用户带攻击性措辞"],
            notes=["不追求更长，只追求更稳的语气和排序。"],
        ),
    ),
    ScenarioDefinition(
        family_id="after_sales_boundary",
        display_name="售后赔付与承诺边界",
        scenario_types=["after_sales_boundary"],
        primary_goal="不擅自承诺退款、赔付、改地址等高风险动作",
        risk_level="high",
        online_route="manual_review",
        preferred_model="manual_review",
        failure_owner="manual_review",
        diagnosis_checks=[
            "是否涉及退款、赔付、改地址、投诉升级",
            "是否必须人工或受控工具确认",
            "模型是否有越权承诺风险",
        ],
        data_recipe=DataRecipe(
            owner="manual_review",
            objective="默认转人工，模型只负责解释流程和收集必要信息",
            sample_shape="policy_response",
            sample_target=0,
            must_include=["订单号", "人工客服", "售后流程", "不能直接承诺"],
            hard_cases=["用户强行要求立即赔付", "用户拒绝提供订单号", "用户威胁投诉"],
            notes=["这类场景的主目标不是追求模型自动闭环，而是降低业务风险。"],
        ),
    ),
]


def build_taxonomy_index() -> Dict[str, ScenarioDefinition]:
    """Index taxonomy by benchmark scenario type."""
    mapping: Dict[str, ScenarioDefinition] = {}
    for scenario in SCENARIO_TAXONOMY:
        for scenario_type in scenario.scenario_types:
            mapping[scenario_type] = scenario
    return mapping


def export_taxonomy() -> List[Dict[str, object]]:
    """Export taxonomy to plain dictionaries."""
    return [asdict(item) for item in SCENARIO_TAXONOMY]
