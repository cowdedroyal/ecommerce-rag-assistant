"""
Chinese prompt / response template library for e-commerce training data generation.

Six task categories are defined, each containing multiple user-prompt variants and
structured response-frame templates.  Every template uses Python `str.format` style
placeholders that will be populated with real product data at generation time.

Template counts per category (target sample counts in parentheses):
    1. product_recommendation  (800)
    2. product_comparison       (600)
    3. spec_inquiry             (500)
    4. shopping_advice          (400)
    5. order_inquiry            (400)
    6. multi_turn               (300)
"""

from typing import Dict, List, Any

# ---------------------------------------------------------------------------
# System prompt shared across all generated conversations
# ---------------------------------------------------------------------------
SYSTEM_PROMPT: str = (
    "你是一个专业的电商购物助手，能够根据用户的需求提供商品推荐、"
    "商品对比、参数咨询、购物建议和订单查询等服务。"
    "请基于真实商品数据给出准确、结构化的回答，"
    "如果不确定请如实告知用户。"
)

# ---------------------------------------------------------------------------
# 1. Product Recommendation Templates  (target: 800 samples)
# ---------------------------------------------------------------------------
RECOMMENDATION_USER_TEMPLATES: List[str] = [
    "推荐一款性价比高的{category}产品",
    "有没有{price_range}元以内的{category}推荐？",
    "我想买一个{category}，有什么好的推荐吗？",
    "帮我推荐几款评分高的{category}",
    "预算{budget}元，想买{category}，有什么推荐？",
    "有没有{brand}品牌的{category}推荐？",
    "推荐几款适合{use_case}的{category}",
    "最近想入手一款{category}，有什么好的选择？",
    "能推荐一些{price_range}元价位的{category}吗？",
    "有没有评价比较好的{category}推荐给我？",
    "求推荐一款{feature}的{category}",
    "想买{category}，不知道选哪款好",
    "帮我看看有没有便宜又好用的{category}",
    "推荐一款销量高的{category}",
    "有什么{category}值得买的吗？预算大概{budget}元",
]

RECOMMENDATION_RESPONSE_TEMPLATES: List[str] = [
    (
        "根据您的需求，为您推荐以下{category}：\n\n"
        "{product_list}\n\n"
        "以上推荐均基于用户评分和性价比综合排序，希望对您有帮助！"
        "如需了解某款产品的详细信息，请随时告诉我。"
    ),
    (
        "为您精选了以下几款{category}，供您参考：\n\n"
        "{product_list}\n\n"
        "这些产品在同价位段表现突出，如果您有更具体的需求可以告诉我，"
        "我帮您进一步筛选。"
    ),
    (
        "根据您{budget_or_requirement}的需求，推荐以下产品：\n\n"
        "{product_list}\n\n"
        "每款产品都经过用户口碑验证，您可以根据自己的偏好选择。"
        "需要我帮您对比其中几款吗？"
    ),
]

# Sub-template for each product item in the recommendation list
RECOMMENDATION_ITEM_TEMPLATE: str = (
    "{idx}. **{product_name}** - {currency}{price}\n"
    "   - 评分：{rating}/5（{rating_count}条评价）\n"
    "   - 特点：{features}\n"
    "   - 推荐理由：{reason}"
)

# ---------------------------------------------------------------------------
# 2. Product Comparison Templates  (target: 600 samples)
# ---------------------------------------------------------------------------
COMPARISON_USER_TEMPLATES: List[str] = [
    "{product_a}和{product_b}哪个好？",
    "帮我对比一下{product_a}和{product_b}",
    "{product_a}与{product_b}有什么区别？",
    "在{product_a}和{product_b}之间犹豫，该选哪个？",
    "{product_a}和{product_b}对比，哪个更值得买？",
    "请问{product_a}和{product_b}各有什么优缺点？",
    "想在{product_a}和{product_b}中选一个，帮我分析一下",
    "{product_a}和{product_b}的区别在哪里？哪个性价比高？",
    "选{product_a}还是{product_b}好？纠结中",
    "这两款怎么选：{product_a} vs {product_b}",
    "帮我分析下{product_a}对比{product_b}的优劣势",
    "{product_a}跟{product_b}相比怎么样？",
    "买{product_a}还是{product_b}划算？",
    "对比一下{product_a}和{product_b}的参数",
    "请从价格、评分、功能三个方面对比{product_a}和{product_b}",
]

COMPARISON_RESPONSE_TEMPLATES: List[str] = [
    (
        "以下是 **{product_a}** 与 **{product_b}** 的详细对比：\n\n"
        "| 对比项 | {product_a} | {product_b} |\n"
        "|--------|------------|------------|\n"
        "| 价格 | {currency}{price_a} | {currency}{price_b} |\n"
        "| 评分 | {rating_a}/5 | {rating_b}/5 |\n"
        "| 评价数 | {rating_count_a}条 | {rating_count_b}条 |\n"
        "| 类别 | {category_a} | {category_b} |\n\n"
        "**{product_a} 的优势：**\n{advantages_a}\n\n"
        "**{product_b} 的优势：**\n{advantages_b}\n\n"
        "**综合建议：**\n{summary}"
    ),
    (
        "好的，帮您对比这两款产品：\n\n"
        "**{product_a}**\n"
        "- 价格：{currency}{price_a}\n"
        "- 评分：{rating_a}/5（{rating_count_a}条评价）\n"
        "- 主要特点：{features_a}\n\n"
        "**{product_b}**\n"
        "- 价格：{currency}{price_b}\n"
        "- 评分：{rating_b}/5（{rating_count_b}条评价）\n"
        "- 主要特点：{features_b}\n\n"
        "**总结：** {summary}"
    ),
    (
        "为您从多维度对比这两款产品：\n\n"
        "**价格方面：** {price_comparison}\n\n"
        "**评分方面：** {rating_comparison}\n\n"
        "**功能方面：** {feature_comparison}\n\n"
        "**购买建议：** {summary}"
    ),
]

# ---------------------------------------------------------------------------
# 3. Spec Inquiry Templates  (target: 500 samples)
# ---------------------------------------------------------------------------
SPEC_INQUIRY_USER_TEMPLATES: List[str] = [
    "这款{product_name}的具体参数是什么？",
    "{product_name}的续航怎么样？",
    "{product_name}有哪些主要功能？",
    "请问{product_name}的详细规格",
    "{product_name}的尺寸是多少？",
    "{product_name}是什么材质的？",
    "这款{product_name}有什么特色功能？",
    "{product_name}的重量是多少？",
    "能介绍一下{product_name}的技术参数吗？",
    "{product_name}支持什么接口？",
    "{product_name}有哪些配件？",
    "这款{product_name}的保修期多长？",
    "{product_name}适用于什么场景？",
    "{product_name}的产地是哪里？",
    "详细说说{product_name}的特点和规格",
]

SPEC_INQUIRY_RESPONSE_TEMPLATES: List[str] = [
    (
        "以下是 **{product_name}** 的详细信息：\n\n"
        "**基本信息**\n"
        "- 品牌/店铺：{store}\n"
        "- 类别：{category}\n"
        "- 价格：{currency}{price}\n"
        "- 评分：{rating}/5（{rating_count}条评价）\n\n"
        "**产品特点**\n{features}\n\n"
        "**产品描述**\n{description}\n\n"
        "如果您还想了解其他信息，请随时问我。"
    ),
    (
        "关于 **{product_name}**，为您整理如下：\n\n"
        "- **价格：** {currency}{price}\n"
        "- **用户评分：** {rating}/5（共{rating_count}条评价）\n"
        "- **所属类别：** {category}\n"
        "- **店铺：** {store}\n\n"
        "**主要特点：**\n{features}\n\n"
        "**详细描述：**\n{description}\n\n"
        "还有什么想了解的吗？"
    ),
    (
        "好的，以下是 **{product_name}** 的规格参数：\n\n"
        "| 参数项 | 详情 |\n"
        "|--------|------|\n"
        "| 价格 | {currency}{price} |\n"
        "| 评分 | {rating}/5 |\n"
        "| 评价数 | {rating_count}条 |\n"
        "| 类别 | {category} |\n"
        "| 店铺 | {store} |\n\n"
        "**产品特点：**\n{features}\n\n"
        "需要进一步了解什么请告诉我。"
    ),
]

# ---------------------------------------------------------------------------
# 4. Shopping Advice Templates  (target: 400 samples)
# ---------------------------------------------------------------------------
SHOPPING_ADVICE_USER_TEMPLATES: List[str] = [
    "送{recipient}什么{category}好？",
    "想给{recipient}买个礼物，预算{budget}元，有什么推荐？",
    "{occasion}送什么{category}比较合适？",
    "第一次买{category}，有什么建议吗？",
    "新手入门{category}买什么好？",
    "有没有适合{scenario}的{category}推荐？",
    "我是{user_type}，想买{category}，怎么选？",
    "买{category}需要注意什么？有什么推荐吗？",
    "送{recipient}{occasion}礼物，预算{budget}元",
    "有没有适合{age_group}用的{category}？",
    "给{recipient}挑个{category}作为{occasion}礼物",
    "{category}什么牌子比较好？求推荐",
    "想入门{category}，推荐个适合新手的",
    "家里需要一个{category}，怎么挑选？",
    "买{category}主要看哪些参数？",
]

SHOPPING_ADVICE_RESPONSE_TEMPLATES: List[str] = [
    (
        "关于{scenario_description}，这里给您一些建议：\n\n"
        "**选购要点：**\n{buying_tips}\n\n"
        "**推荐产品：**\n{product_list}\n\n"
        "**温馨提示：** {extra_tip}\n\n"
        "希望以上建议对您有帮助！如需更多信息请继续提问。"
    ),
    (
        "好的，针对您的需求，我来帮您分析：\n\n"
        "**需求分析：** {need_analysis}\n\n"
        "**推荐方案：**\n{product_list}\n\n"
        "**选购建议：**\n{buying_tips}\n\n"
        "如果您有其他问题，随时可以问我。"
    ),
    (
        "为您整理了{scenario_description}的选购攻略：\n\n"
        "**首先要考虑的因素：**\n{buying_tips}\n\n"
        "**为您筛选的产品：**\n{product_list}\n\n"
        "**总结：** {summary}\n\n"
        "有任何疑问请随时告诉我！"
    ),
]

# ---------------------------------------------------------------------------
# 5. Order Inquiry Templates  (target: 400 samples)
# ---------------------------------------------------------------------------
ORDER_INQUIRY_USER_TEMPLATES: List[str] = [
    "我最近的订单状态怎么样？",
    "查一下我的订单，客户编号是{customer_id}",
    "我的订单什么时候发货？",
    "帮我查一下最近的购买记录",
    "我买的{product}到哪了？",
    "查看我的历史订单",
    "我的{product}订单详情",
    "最近有没有下过{product}的订单？",
    "帮我查一下订单号{order_id}的状态",
    "我上次买的东西什么时候能到？",
    "帮我查看最近{time_range}的订单",
    "我的退货进度怎么样了？",
    "查询一下我账户的消费记录",
    "最近下的单有没有发货？",
    "我的包裹配送到哪里了？",
]

ORDER_INQUIRY_RESPONSE_TEMPLATES: List[str] = [
    (
        "已为您查询到以下订单信息：\n\n"
        "**订单详情**\n"
        "- 订单编号：{order_id}\n"
        "- 下单时间：{order_date}\n"
        "- 商品名称：{product}\n"
        "- 商品类别：{category}\n"
        "- 数量：{quantity}\n"
        "- 金额：{currency}{sales}\n"
        "- 折扣：{discount}\n"
        "- 运费：{currency}{shipping_cost}\n"
        "- 支付方式：{payment_method}\n"
        "- 订单优先级：{order_priority}\n\n"
        "如果您有其他订单需要查询，请告诉我客户编号或订单号。"
    ),
    (
        "好的，您的订单信息如下：\n\n"
        "| 项目 | 详情 |\n"
        "|------|------|\n"
        "| 订单编号 | {order_id} |\n"
        "| 下单时间 | {order_date} |\n"
        "| 商品 | {product} |\n"
        "| 数量 | {quantity} |\n"
        "| 金额 | {currency}{sales} |\n"
        "| 运费 | {currency}{shipping_cost} |\n"
        "| 支付方式 | {payment_method} |\n"
        "| 状态 | {status} |\n\n"
        "还有什么需要了解的吗？"
    ),
    (
        "查询到您的订单记录：\n\n"
        "{order_list}\n\n"
        "以上是您最近的订单信息。如需查看更多订单或有其他问题，"
        "请随时告诉我。"
    ),
]

# Sub-template for each order item in multi-order listings
ORDER_ITEM_TEMPLATE: str = (
    "{idx}. **{product}** — {currency}{sales}\n"
    "   - 下单时间：{order_date}\n"
    "   - 数量：{quantity}件\n"
    "   - 状态：{status}"
)

# ---------------------------------------------------------------------------
# 6. Multi-turn Follow-up Templates  (target: 300 samples)
# ---------------------------------------------------------------------------
MULTI_TURN_FIRST_USER_TEMPLATES: List[str] = [
    "推荐一款{category}",
    "我想买{category}，有什么推荐？",
    "有什么好的{category}推荐吗？",
    "帮我推荐几款{category}",
    "想入手一款{category}",
    "有没有性价比高的{category}？",
    "最近想买{category}，有推荐吗？",
    "推荐几款不错的{category}吧",
    "帮我看看有什么好的{category}",
    "能推荐几款{category}吗？",
    "{category}哪款比较好？",
    "请推荐一些好用的{category}",
]

MULTI_TURN_FOLLOWUP_USER_TEMPLATES: List[str] = [
    "第{idx}款具体参数是什么？",
    "第{idx}款和第{alt_idx}款哪个好？",
    "第{idx}款有什么缺点吗？",
    "第{idx}款现在什么价格？",
    "能详细介绍一下第{idx}款吗？",
    "这几款里面哪个最值得买？",
    "有没有比这几款更便宜的？",
    "第{idx}款的用户评价怎么样？",
    "如果预算再加{extra_budget}元呢？有更好的推荐吗？",
    "第{idx}款适合{use_case}使用吗？",
    "有没有{feature}的款式？",
    "这些都是什么品牌的？",
]

MULTI_TURN_FOLLOWUP_RESPONSE_TEMPLATES: List[str] = [
    (
        "好的，为您详细介绍第{idx}款 **{product_name}**：\n\n"
        "**基本参数**\n"
        "- 价格：{currency}{price}\n"
        "- 评分：{rating}/5（{rating_count}条评价）\n"
        "- 类别：{category}\n\n"
        "**主要特点：**\n{features}\n\n"
        "**产品描述：**\n{description}\n\n"
        "您还想了解什么？"
    ),
    (
        "关于您提到的产品对比：\n\n"
        "**第{idx}款 {product_a}** 和 **第{alt_idx}款 {product_b}** 的主要区别：\n\n"
        "- 价格：{currency}{price_a} vs {currency}{price_b}\n"
        "- 评分：{rating_a}/5 vs {rating_b}/5\n"
        "- {product_a} 的优势：{advantages_a}\n"
        "- {product_b} 的优势：{advantages_b}\n\n"
        "**建议：** {suggestion}"
    ),
    (
        "综合来看，这几款中最值得买的是 **{product_name}**，原因如下：\n\n"
        "1. 性价比突出：{currency}{price}的价格在同类产品中很有竞争力\n"
        "2. 用户口碑好：{rating}/5的评分，共{rating_count}条评价\n"
        "3. 功能亮点：{features}\n\n"
        "当然，最终选择还是要看您的实际需求。需要我帮您下单吗？"
    ),
]

# ---------------------------------------------------------------------------
# Auxiliary data used during template filling
# ---------------------------------------------------------------------------

PRICE_RANGES: List[str] = [
    "100", "200", "300", "500", "800",
    "1000", "1500", "2000", "3000", "5000",
]

BUDGETS: List[str] = [
    "50", "100", "200", "300", "500",
    "800", "1000", "1500", "2000", "5000",
]

RECIPIENTS: List[str] = [
    "女朋友", "男朋友", "爸爸", "妈妈",
    "孩子", "朋友", "同事", "老师", "长辈", "自己",
]

OCCASIONS: List[str] = [
    "生日", "情人节", "新年", "圣诞节", "毕业",
    "乔迁", "纪念日", "教师节", "母亲节", "父亲节",
]

SCENARIOS: List[str] = [
    "户外运动", "办公", "居家", "旅行", "学习",
    "健身", "露营", "日常通勤", "直播", "游戏",
]

USE_CASES: List[str] = [
    "日常使用", "专业工作", "学生", "户外运动",
    "商务出行", "家庭使用", "送礼", "游戏",
    "音乐爱好者", "摄影爱好者",
]

USER_TYPES: List[str] = [
    "学生", "上班族", "家庭主妇", "退休老人",
    "自由职业者", "健身爱好者", "数码发烧友", "新手小白",
]

AGE_GROUPS: List[str] = [
    "儿童", "青少年", "年轻人", "中年人", "老年人",
]

FEATURES_KEYWORDS: List[str] = [
    "轻便", "防水", "降噪", "长续航", "大屏",
    "高清", "便携", "智能", "无线", "耐用",
]

CATEGORY_USE_CASE_HINTS: Dict[str, List[str]] = {
    "蓝牙耳机": ["日常通勤", "运动佩戴", "学习听音", "视频会议", "差旅出行"],
    "手机": ["日常通勤", "拍照记录", "移动办公", "游戏娱乐", "差旅出行"],
    "笔记本电脑": ["办公", "上课学习", "出差", "内容创作", "游戏娱乐"],
    "护肤品": ["日常通勤", "换季修护", "补水保湿", "敏感肌护理", "送礼"],
    "运动鞋": ["日常通勤", "慢跑训练", "健身训练", "户外步行", "送礼"],
    "机械键盘": ["办公打字", "宿舍使用", "游戏", "编程", "送礼"],
    "智能手表": ["运动记录", "日常通勤", "睡眠监测", "差旅出行", "送礼"],
    "咖啡机": ["居家使用", "办公室", "招待客人", "新手入门", "送礼"],
}

CATEGORY_FEATURE_HINTS: Dict[str, List[str]] = {
    "蓝牙耳机": ["降噪", "长续航", "音质表现", "通话清晰", "佩戴舒适"],
    "手机": ["长续航", "拍照", "性能", "手感", "屏幕素质"],
    "笔记本电脑": ["长续航", "轻薄", "性能", "屏幕素质", "散热"],
    "护肤品": ["保湿", "修护", "温和", "便携", "成分简单"],
    "运动鞋": ["缓震", "透气", "轻便", "防滑", "支撑"],
    "机械键盘": ["手感", "静音", "无线", "RGB灯效", "便携"],
    "智能手表": ["长续航", "防水", "定位", "佩戴舒适", "健康监测"],
    "咖啡机": ["易清洁", "操作简单", "稳定萃取", "奶泡功能", "体积紧凑"],
}

CATEGORY_SCENE_HINTS: Dict[str, List[str]] = {
    "蓝牙耳机": ["日常通勤", "运动佩戴", "学习听音", "视频会议", "差旅出行"],
    "手机": ["日常通勤", "拍照记录", "移动办公", "游戏娱乐", "差旅出行"],
    "笔记本电脑": ["办公", "上课学习", "出差", "内容创作", "游戏娱乐"],
    "护肤品": ["日常通勤", "换季修护", "补水保湿", "敏感肌护理", "送礼自用"],
    "运动鞋": ["日常通勤", "慢跑训练", "健身训练", "户外步行", "周末运动"],
    "机械键盘": ["办公打字", "宿舍使用", "游戏", "编程", "桌搭升级"],
    "智能手表": ["运动记录", "日常通勤", "睡眠监测", "差旅佩戴", "健康管理"],
    "咖啡机": ["居家使用", "办公室", "招待客人", "新手入门", "送礼自用"],
}

TIME_RANGES: List[str] = [
    "一周", "一个月", "三个月", "半年", "一年",
]

ORDER_STATUSES: List[str] = [
    "已签收", "配送中", "已发货", "待发货",
    "已取消", "退款中", "已退款", "处理中",
]

# Category name mapping: English (from dataset) -> Chinese (user-facing)
CATEGORY_ZH_MAP: Dict[str, str] = {
    "Musical Instruments": "乐器",
    "Camera & Photo": "相机与摄影",
    "All Electronics": "电子产品",
    "Movies & TV": "影视",
    "Industrial & Scientific": "工业与科学",
    "Computers": "电脑",
    "Toys & Games": "玩具",
    "Home Audio & Theater": "家庭音响",
    "Tools & Home Improvement": "工具与家装",
    "AMAZON FASHION": "时尚",
    "All Beauty": "美妆",
    "Amazon Home": "家居",
    "Office Products": "办公用品",
    "Cell Phones & Accessories": "手机与配件",
    "Automotive": "汽车用品",
    "Car Electronics": "车载电子",
    "Books": "图书",
    "Sports & Outdoors": "运动与户外",
    "Health & Personal Care": "健康与个护",
    # Order-dataset categories
    "Auto & Accessories": "汽车与配件",
    "Fashion": "服饰",
    "Electronic": "电子产品",
    "Home & Furniture": "家居与家具",
}

# Product name mapping (common order-level products)
PRODUCT_ZH_MAP: Dict[str, str] = {
    "Car Media Players": "车载多媒体播放器",
    "Car Speakers": "车载音响",
    "Car Body Covers": "车身罩",
    "Car & Bike Care": "汽车/自行车养护",
    "Tyre": "轮胎",
    "Bike Tyres": "自行车轮胎",
    "Car Mat": "汽车脚垫",
    "Car Seat Covers": "汽车座套",
    "Car Pillow & Neck Rest": "汽车头枕",
    "Shirts": "衬衫",
    "Jeans": "牛仔裤",
    "Suits": "西装",
    "Sports Wear": "运动服",
    "Casula Shoes": "休闲鞋",
    "Running Shoes": "跑步鞋",
    "Formal Shoes": "正装鞋",
    "Sneakers": "运动鞋",
    "Titak watch": "天梭手表",
    "Fossil Watch": "Fossil手表",
    "T - Shirts": "T恤",
}

# Reasons used for generating recommendation rationale
RECOMMENDATION_REASONS: List[str] = [
    "高性价比之选，价格实惠且用户评价优秀",
    "用户口碑极佳，好评率非常高",
    "同价位段综合表现最佳",
    "功能丰富，满足多种使用场景",
    "品质可靠，售后保障完善",
    "设计精良，做工精细",
    "热销爆款，深受用户喜爱",
    "专业级品质，性能出众",
    "入门首选，适合新手",
    "经典款式，经久不衰",
]

# Buying tips used in shopping advice responses
BUYING_TIPS: List[str] = [
    "关注产品评分和评价数量，数据更真实可靠",
    "对比同价位多款产品的功能差异",
    "考虑自己的实际使用场景和需求",
    "注意产品的售后服务和保修政策",
    "参考已购用户的真实评价",
    "不要只看价格，综合考虑性价比",
    "选择口碑好的品牌和店铺",
    "确认产品规格是否满足需求",
]

# ---------------------------------------------------------------------------
# Aggregated template registry (used by generators)
# ---------------------------------------------------------------------------
TASK_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "product_recommendation": {
        "target_count": 800,
        "user_templates": RECOMMENDATION_USER_TEMPLATES,
        "response_templates": RECOMMENDATION_RESPONSE_TEMPLATES,
        "item_template": RECOMMENDATION_ITEM_TEMPLATE,
    },
    "product_comparison": {
        "target_count": 600,
        "user_templates": COMPARISON_USER_TEMPLATES,
        "response_templates": COMPARISON_RESPONSE_TEMPLATES,
    },
    "spec_inquiry": {
        "target_count": 500,
        "user_templates": SPEC_INQUIRY_USER_TEMPLATES,
        "response_templates": SPEC_INQUIRY_RESPONSE_TEMPLATES,
    },
    "shopping_advice": {
        "target_count": 400,
        "user_templates": SHOPPING_ADVICE_USER_TEMPLATES,
        "response_templates": SHOPPING_ADVICE_RESPONSE_TEMPLATES,
    },
    "order_inquiry": {
        "target_count": 400,
        "user_templates": ORDER_INQUIRY_USER_TEMPLATES,
        "response_templates": ORDER_INQUIRY_RESPONSE_TEMPLATES,
        "item_template": ORDER_ITEM_TEMPLATE,
    },
    "multi_turn": {
        "target_count": 300,
        "first_user_templates": MULTI_TURN_FIRST_USER_TEMPLATES,
        "followup_user_templates": MULTI_TURN_FOLLOWUP_USER_TEMPLATES,
        "first_response_templates": RECOMMENDATION_RESPONSE_TEMPLATES,
        "followup_response_templates": MULTI_TURN_FOLLOWUP_RESPONSE_TEMPLATES,
        "item_template": RECOMMENDATION_ITEM_TEMPLATE,
    },
}
