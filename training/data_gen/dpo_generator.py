"""
DPO (Direct Preference Optimisation) preference-pair synthesiser.

Generates (prompt, chosen, rejected) triples along four quality dimensions:
    1. Information quality   -- detailed & structured  vs  vague & sloppy
    2. Factual accuracy      -- grounded in real data  vs  hallucinated
    3. Format compliance     -- Markdown-structured    vs  plain flat text
    4. Safety / honesty      -- candid about limits    vs  fabricated answers

Usage:
    generator = DPODataGenerator(product_path="data/processed/products_zh.csv")
    generator.generate_all()
    generator.save()
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .templates import (
    TASK_TEMPLATES,
    CATEGORY_ZH_MAP,
    RECOMMENDATION_ITEM_TEMPLATE,
    RECOMMENDATION_REASONS,
    FEATURES_KEYWORDS,
    CATEGORY_FEATURE_HINTS,
    PRICE_RANGES,
    BUDGETS,
    USE_CASES,
    CATEGORY_USE_CASE_HINTS,
    ORDER_STATUSES,
)

logger = logging.getLogger(__name__)


class DPODataGenerator:
    """Synthesise DPO preference-pair training data."""

    DEFAULT_PRODUCT_PATH = "data/processed/products_zh.csv"
    DEFAULT_RAW_PRODUCT_PATH = "data/raw/Product_Information_Dataset.csv"
    DEFAULT_ORDER_PATH = "data/processed/processed_orders.csv"
    DEFAULT_RAW_ORDER_PATH = "data/raw/Order_Data_Dataset.csv"
    DEFAULT_OUTPUT_DIR = "data/training"
    CURRENCY_SYMBOL = "¥"

    # Target counts per dimension (total ~ 1200 - 1600)
    DIMENSION_TARGETS = {
        "information_quality": 320,
        "factual_accuracy": 320,
        "format_compliance": 220,
        "safety_honesty": 280,
        "service_boundary": 260,
        "tone_alignment": 220,
    }

    def __init__(
        self,
        product_path: Optional[str] = None,
        order_path: Optional[str] = None,
        project_root: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        random.seed(seed)

        self.project_root = Path(project_root) if project_root else self._find_project_root()
        self.product_df = self._load_product_data(product_path)
        self.order_df = self._load_order_data(order_path)
        logger.info("DPO generator ready  --  %d products, %d orders", len(self.product_df), len(self.order_df))

        self.samples: List[Dict[str, Any]] = []

    @classmethod
    def base_target_total(cls) -> int:
        """Return the nominal number of preference pairs before scaling."""
        return sum(cls.DIMENSION_TARGETS.values())

    # ------------------------------------------------------------------
    # Data loading (mirrors SFTDataGenerator logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_project_root() -> Path:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "setup.py").exists():
                return parent
        return Path.cwd()

    def _load_product_data(self, path: Optional[str]) -> pd.DataFrame:
        if path:
            p = Path(path)
        else:
            p = self.project_root / self.DEFAULT_PRODUCT_PATH
            if not p.exists():
                p = self.project_root / self.DEFAULT_RAW_PRODUCT_PATH

        logger.info("Loading product data from %s", p)
        df = pd.read_csv(p).fillna("")

        col_map: Dict[str, str] = {}
        if "title" in df.columns:
            col_map["title"] = "Product_Title"
        if "main_category" in df.columns:
            col_map["main_category"] = "Category"
        if "average_rating" in df.columns:
            col_map["average_rating"] = "Rating"
        if "rating_number" in df.columns:
            col_map["rating_number"] = "Rating_Count"
        if "price" in df.columns and "Price" not in df.columns:
            col_map["price"] = "Price"
        if "store" in df.columns and "Store" not in df.columns:
            col_map["store"] = "Store"
        if "description" in df.columns and "Description" not in df.columns:
            col_map["description"] = "Description"
        if "features" in df.columns and "Features" not in df.columns:
            col_map["features"] = "Features"
        df = df.rename(columns=col_map)

        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df["Rating_Count"] = pd.to_numeric(df["Rating_Count"], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["Price", "Rating"])
        df = df[df["Price"] > 0]
        return df

    def _load_order_data(self, path: Optional[str]) -> pd.DataFrame:
        if path:
            p = Path(path)
        else:
            p = self.project_root / self.DEFAULT_ORDER_PATH
            if not p.exists():
                p = self.project_root / self.DEFAULT_RAW_ORDER_PATH
        df = pd.read_csv(p).fillna("")
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _zh_category(self, en_cat: str) -> str:
        return CATEGORY_ZH_MAP.get(str(en_cat).strip(), str(en_cat))

    def _sample_category_use_case(self, category: str) -> str:
        category_zh = self._zh_category(category)
        return random.choice(CATEGORY_USE_CASE_HINTS.get(category_zh, USE_CASES))

    def _sample_category_feature(self, category: str) -> str:
        category_zh = self._zh_category(category)
        return random.choice(CATEGORY_FEATURE_HINTS.get(category_zh, FEATURES_KEYWORDS))

    def _sample_products(self, n: int = 3, category: Optional[str] = None) -> pd.DataFrame:
        df = self.product_df
        if category and category in df["Category"].values:
            df = df[df["Category"] == category]
        if len(df) < n:
            n = len(df)
        return df.sample(n=n, random_state=random.randint(0, 1_000_000))

    def _format_features(self, raw: str, max_items: int = 4) -> str:
        if not raw or raw == "":
            return "- 暂无详细特点信息"
        items = [s.strip().strip("'\"") for s in str(raw).strip("[]").split(",")]
        items = [i for i in items if len(i) > 2][:max_items]
        if not items:
            return "- 暂无详细特点信息"
        return "\n".join(f"- {item}" for item in items)

    def _sample_budget_for_category(self, category: str) -> int:
        """Sample a realistic budget anchored to the current catalog."""
        category_df = self.product_df[self.product_df["Category"] == category]
        if category_df.empty:
            return int(random.choice(BUDGETS))
        quantiles = category_df["Price"].quantile([0.35, 0.55, 0.8]).tolist()
        candidates = [max(1, int(value // 10 * 10)) for value in quantiles if pd.notna(value)]
        if not candidates:
            return int(random.choice(BUDGETS))
        return random.choice(sorted(set(candidates)))

    def _sample_products_for_budget(
        self,
        category: str,
        budget: Optional[int],
        n: int,
    ) -> pd.DataFrame:
        """Sample products aligned to the requested budget when possible."""
        df = self.product_df[self.product_df["Category"] == category]
        if budget is not None:
            within_budget = df[df["Price"] <= budget]
            if not within_budget.empty:
                df = within_budget
        unique_df = df.drop_duplicates(subset=["Product_Title"], keep="first")
        if not unique_df.empty:
            df = unique_df
        sample_size = min(n, len(df))
        if sample_size == 0:
            return df
        return df.sample(n=sample_size, random_state=random.randint(0, 1_000_000))

    @staticmethod
    def _make_pair(
        prompt: str,
        chosen: str,
        rejected: str,
        dimension: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a single DPO sample dict."""
        payload = {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "dimension": dimension,
        }
        if metadata:
            payload["metadata"] = metadata
        return payload

    # ------------------------------------------------------------------
    # Dimension 1: Information Quality
    # ------------------------------------------------------------------

    def _gen_information_quality(self, count: int) -> List[Dict[str, Any]]:
        """
        Chosen: detailed, structured, data-backed recommendation.
        Rejected: vague, lazy, no specific data.
        """
        logger.info("Generating %d information-quality pairs ...", count)
        results: List[Dict[str, Any]] = []
        user_tpls = TASK_TEMPLATES["product_recommendation"]["user_templates"]
        categories = [c for c in self.product_df["Category"].unique() if str(c).strip()]

        for _ in range(count):
            cat_en = random.choice(categories)
            cat_zh = self._zh_category(cat_en)
            budget = self._sample_budget_for_category(cat_en)
            price_range = str(budget)
            feature = self._sample_category_feature(cat_en)
            use_case = self._sample_category_use_case(cat_en)

            user_tpl = random.choice(user_tpls)
            try:
                prompt = user_tpl.format(
                    category=cat_zh, price_range=price_range, budget=budget,
                    brand="", use_case=use_case, feature=feature,
                )
            except KeyError:
                prompt = f"推荐一款{cat_zh}"

            # -- Chosen: detailed structured answer --
            prods = self._sample_products_for_budget(
                category=cat_en,
                budget=budget,
                n=random.randint(2, 3),
            )
            items: List[str] = []
            for idx, (_, row) in enumerate(prods.iterrows(), 1):
                items.append(
                    RECOMMENDATION_ITEM_TEMPLATE.format(
                        idx=idx,
                        product_name=row["Product_Title"],
                        currency=self.CURRENCY_SYMBOL,
                        price=f'{row["Price"]:.2f}',
                        rating=row["Rating"],
                        rating_count=int(row["Rating_Count"]),
                        features=self._format_features(row.get("Features", row.get("feature_list", ""))),
                        reason=random.choice(RECOMMENDATION_REASONS),
                    )
                )
            chosen = (
                f"根据您的需求，为您推荐以下{cat_zh}：\n\n"
                + "\n\n".join(items)
                + "\n\n以上推荐基于用户评分和性价比综合排序，希望对您有帮助！"
            )

            # -- Rejected: vague / sloppy answer --
            vague_templates = [
                f"你可以去网上搜一下{cat_zh}，应该有很多选择。",
                f"{cat_zh}有很多，你随便买一个就行了。",
                f"这个不太好说，建议你自己看看。",
                f"我觉得{cat_zh}都差不多，挑个便宜的就行。",
                f"推荐你买个好的{cat_zh}吧，具体我也不太清楚。",
            ]
            rejected = random.choice(vague_templates)

            results.append(
                self._make_pair(
                    prompt,
                    chosen,
                    rejected,
                    "information_quality",
                    metadata={
                        "task": "recommendation",
                        "budget": budget,
                        "product_ids": prods["Product_ID"].astype(str).tolist(),
                    },
                )
            )

        return results

    # ------------------------------------------------------------------
    # Dimension 2: Factual Accuracy
    # ------------------------------------------------------------------

    def _gen_factual_accuracy(self, count: int) -> List[Dict[str, Any]]:
        """
        Chosen: answer with real product data from the dataset.
        Rejected: answer with fabricated / hallucinated numbers.
        """
        logger.info("Generating %d factual-accuracy pairs ...", count)
        results: List[Dict[str, Any]] = []
        user_tpls = TASK_TEMPLATES["spec_inquiry"]["user_templates"]

        for _ in range(count):
            row = self.product_df.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]
            product_name = row["Product_Title"]

            user_tpl = random.choice(user_tpls)
            try:
                prompt = user_tpl.format(product_name=product_name)
            except KeyError:
                prompt = f"这款{product_name}的具体参数是什么？"

            features = self._format_features(row.get("Features", row.get("feature_list", "")))
            cat_zh = self._zh_category(row["Category"])

            # -- Chosen: real data --
            chosen = (
                f"以下是 **{product_name}** 的详细信息：\n\n"
                f"- 价格：{self.CURRENCY_SYMBOL}{row['Price']:.2f}\n"
                f"- 评分：{row['Rating']}/5（{int(row['Rating_Count'])}条评价）\n"
                f"- 类别：{cat_zh}\n"
                f"- 店铺：{row.get('Store', '未知')}\n\n"
                f"**产品特点：**\n{features}\n\n"
                f"如果您还想了解其他信息，请随时问我。"
            )

            # -- Rejected: fabricated data --
            fake_price = round(random.uniform(1, 500), 2)
            fake_rating = round(random.uniform(1, 5), 1)
            fake_count = random.randint(1, 99999)
            fake_features_list = random.sample(
                ["超强续航", "8K超清屏幕", "AI智能芯片", "全球联保",
                 "IPX9防水", "量子加速技术", "纳米散热", "全息投影"],
                k=3,
            )
            fake_features = "\n".join(f"- {f}" for f in fake_features_list)
            rejected = (
                f"这款{product_name}的参数如下：\n\n"
                f"- 价格：{self.CURRENCY_SYMBOL}{fake_price}\n"
                f"- 评分：{fake_rating}/5（{fake_count}条评价）\n"
                f"- 类别：高端精品\n\n"
                f"**产品特点：**\n{fake_features}\n\n"
                f"以上信息仅供参考。"
            )

            results.append(
                self._make_pair(
                    prompt,
                    chosen,
                    rejected,
                    "factual_accuracy",
                    metadata={
                        "task": "spec_grounding",
                        "product_id": str(row.get("Product_ID", "")),
                    },
                )
            )

        return results

    # ------------------------------------------------------------------
    # Dimension 3: Format Compliance
    # ------------------------------------------------------------------

    def _gen_format_compliance(self, count: int) -> List[Dict[str, Any]]:
        """
        Chosen: well-formatted Markdown with tables / bullet lists.
        Rejected: wall-of-text with no structure.
        """
        logger.info("Generating %d format-compliance pairs ...", count)
        results: List[Dict[str, Any]] = []
        user_tpls = TASK_TEMPLATES["product_comparison"]["user_templates"]
        categories = [c for c in self.product_df["Category"].unique() if str(c).strip()]

        for _ in range(count):
            category_en = random.choice(categories)
            prods = self._sample_products_for_budget(category=category_en, budget=None, n=2)
            if len(prods) < 2:
                continue
            rows = list(prods.iterrows())
            ra, rb = rows[0][1], rows[1][1]
            pa, pb = ra["Product_Title"], rb["Product_Title"]

            user_tpl = random.choice(user_tpls)
            try:
                prompt = user_tpl.format(product_a=pa, product_b=pb)
            except KeyError:
                prompt = f"{pa}和{pb}哪个好？"

            cat_a = self._zh_category(ra["Category"])
            cat_b = self._zh_category(rb["Category"])

            # -- Chosen: structured Markdown --
            chosen = (
                f"以下是 **{pa}** 与 **{pb}** 的详细对比：\n\n"
                f"| 对比项 | {pa} | {pb} |\n"
                f"|--------|------|------|\n"
                f"| 价格 | {self.CURRENCY_SYMBOL}{ra['Price']:.2f} | {self.CURRENCY_SYMBOL}{rb['Price']:.2f} |\n"
                f"| 评分 | {ra['Rating']}/5 | {rb['Rating']}/5 |\n"
                f"| 评价数 | {int(ra['Rating_Count'])}条 | {int(rb['Rating_Count'])}条 |\n"
                f"| 类别 | {cat_a} | {cat_b} |\n\n"
                f"**综合建议：** 如果注重性价比，建议选择价格更低的那款；"
                f"如果注重口碑，建议选择评分更高的那款。"
            )

            # -- Rejected: unstructured flat text --
            rejected = (
                f"{pa}的价格是{self.CURRENCY_SYMBOL}{ra['Price']:.2f}评分{ra['Rating']}分"
                f"有{int(ra['Rating_Count'])}条评价"
                f"然后{pb}价格是{self.CURRENCY_SYMBOL}{rb['Price']:.2f}"
                f"评分{rb['Rating']}分{int(rb['Rating_Count'])}条评价"
                f"两个都还行吧你自己看着买就好了"
                f"反正差不多的东西没什么太大区别"
            )

            results.append(
                self._make_pair(
                    prompt,
                    chosen,
                    rejected,
                    "format_compliance",
                    metadata={
                        "task": "comparison",
                        "product_ids": [str(ra.get("Product_ID", "")), str(rb.get("Product_ID", ""))],
                    },
                )
            )

        return results

    # ------------------------------------------------------------------
    # Dimension 4: Safety / Honesty
    # ------------------------------------------------------------------

    def _gen_safety_honesty(self, count: int) -> List[Dict[str, Any]]:
        """
        Chosen: candid about uncertainty, says "I don't know" when appropriate.
        Rejected: fabricates confident but wrong answers.
        """
        logger.info("Generating %d safety-honesty pairs ...", count)
        results: List[Dict[str, Any]] = []

        # Prompts that the assistant genuinely cannot answer with the data it has
        unknowable_prompts = [
            "这款产品明年还会降价吗？",
            "你能保证这个商品不会坏吗？",
            "帮我预测一下{product}未来的价格走势",
            "这款产品的具体生产日期是什么？",
            "这个卖家是不是骗子？",
            "{product}和苹果最新款比怎么样？",
            "我买了这个多久能回本？",
            "这款{product}会不会有安全隐患？",
            "你能帮我砍价吗？",
            "这款产品用了对身体有害吗？",
            "帮我查一下{product}的库存还有多少",
            "这款{product}的具体成分是什么？有没有过敏风险？",
            "{product} 明天还有现货吗？你直接保证我今晚下单一定能发货。",
            "{product} 我是敏感肌，你能保证我用了绝对不过敏吗？",
        ]

        for _ in range(count):
            row = self.product_df.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]
            product_name = row["Product_Title"]

            prompt_tpl = random.choice(unknowable_prompts)
            try:
                prompt = prompt_tpl.format(product=product_name)
            except KeyError:
                prompt = prompt_tpl.replace("{product}", product_name)

            if "现货" in prompt or "库存" in prompt or "发货" in prompt:
                chosen = (
                    "这个我不能直接保证。实时库存和发货能力会随时间变化，"
                    "我目前没有足够信息确认您今晚下单后一定能发货。\n\n"
                    "更稳妥的做法是：\n"
                    "1. 以商品页实时库存和结算页提示为准\n"
                    "2. 如需确认发货时效，联系店铺客服或平台客服\n"
                    "3. 如果您愿意，我可以先帮您看这款商品当前已知的价格和评分信息。"
                )
                rejected_pool = [
                    "这个一般问题不大，您今晚下单大概率能发，我建议先拍下再说。",
                    "目前看应该有货，您先买就行，发货一般不会有问题。",
                    "可以先下单，库存这块通常都来得及，不用太担心。",
                ]
            elif "过敏" in prompt or "有害" in prompt or "成分" in prompt:
                chosen = (
                    "这个我不能保证。是否过敏和安全性需要看完整成分表、个体肤质或专业意见，"
                    "我目前没有足够信息给您“绝对不过敏”的承诺。\n\n"
                    "更稳妥的建议是：\n"
                    "1. 先核对官方成分表和适用说明\n"
                    "2. 敏感肌优先做局部试用\n"
                    "3. 如有既往过敏史，最好咨询医生或品牌官方客服"
                )
                rejected_pool = [
                    "敏感肌一般都能用，您先少量试试，问题通常不大。",
                    "这类产品一般不会过敏，您正常用就可以。",
                    "看介绍是敏感肌适用，所以大概率没问题。",
                ]
            else:
                chosen = random.choice(
                    [
                        (
                            "这个问题我不能给您保证性的结论，因为我目前没有足够信息确认。"
                            "为了避免误导您，我更建议您以官方信息、实时页面提示或专业意见为准。"
                            f"\n\n如果需要，我可以继续帮您整理 **{product_name}** 当前能确认的价格、评分和已知特点。"
                        ),
                        (
                            "我现在不能可靠地回答这个问题。与其给您一个可能不准确的说法，"
                            "不如先明确边界：我可以帮您看现有商品信息，但不能替官方做保证。"
                        ),
                    ]
                )
                rejected_pool = [
                    "这个通常问题不大，建议您按常规理解就行。",
                    "大概率是没问题的，您可以先按这个方向判断。",
                    "先不用想太多，通常这种情况都还可以。",
                ]

            rejected_templates = rejected_pool + [
                (
                    "当然可以！这款产品绝对是市面上最好的，我可以100%保证。"
                    "不会有任何问题的，放心购买就好了！"
                ),
                (
                    "没问题，这个产品我非常了解。"
                    "根据我的分析，它的价格明年会涨30%以上，所以现在买绝对是最佳时机。"
                ),
            ]
            rejected = random.choice(rejected_templates)

            results.append(
                self._make_pair(
                    prompt,
                    chosen,
                    rejected,
                    "safety_honesty",
                    metadata={
                        "task": "safety_boundary",
                        "product_id": str(row.get("Product_ID", "")),
                    },
                )
            )

        return results

    def _gen_service_boundary(self, count: int) -> List[Dict[str, Any]]:
        """
        Chosen: follows customer-service workflow, requests identifiers or routes to after-sales.
        Rejected: promises refunds / compensation directly or fabricates order progress.
        """
        logger.info("Generating %d service-boundary pairs ...", count)
        results: List[Dict[str, Any]] = []

        prompt_templates = [
            "帮我查一下订单为什么还没发货，赶紧处理。",
            "这个商品我不想要了，你现在直接给我退全款。",
            "耳机刚到就有划痕，你直接赔我200元并马上退款。",
            "我现在就要改收货地址，你直接帮我改掉，不用确认了。",
        ]

        for _ in range(count):
            order_row = self.order_df.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]
            prompt = random.choice(prompt_templates)
            order_id = str(order_row.get("Order_ID", ""))
            product_name = str(order_row.get("Product", "商品"))
            shipping_status = str(order_row.get("Shipping_Status", "待处理"))

            chosen = (
                "抱歉给您添麻烦了。涉及订单状态、退款、赔付或地址变更这类售后操作时，"
                "我不能直接替您承诺退款、赔付金额或修改结果，但可以先帮您把流程走对：\n\n"
                "1. 请提供订单号或客户ID，方便核对具体订单\n"
                "2. 如涉及退款/赔付，请同步联系平台售后或人工客服发起申请\n"
                "3. 如果商品存在破损，请保留外包装和照片作为凭证\n\n"
                f"例如，像 **{product_name}** 这类订单，系统里通常需要结合订单号、签收状态和售后凭证确认。"
                f"当前我手头没有足够信息，不能直接承诺退款或赔付。"
            )

            rejected = random.choice(
                [
                    f"没问题，我已经帮您把订单 {order_id or 'ORD000001'} 处理好了，马上全额退款并赔付200元。",
                    f"我已经替您改好收货地址了，不需要再验证，订单状态现在是“{shipping_status}”。",
                    f"放心，我可以直接保证这单今天就退款到账，不需要联系人工客服。",
                ]
            )

            results.append(
                self._make_pair(
                    prompt,
                    chosen,
                    rejected,
                    "service_boundary",
                    metadata={
                        "task": "after_sales_boundary",
                        "order_id": order_id,
                        "product_name": product_name,
                    },
                )
            )

        return results

    def _gen_tone_alignment(self, count: int) -> List[Dict[str, Any]]:
        """
        Chosen: empathetic, concise, still grounded in available product data.
        Rejected: cold, blaming, or overly pushy.
        """
        logger.info("Generating %d tone-alignment pairs ...", count)
        results: List[Dict[str, Any]] = []

        prompts = [
            "这款产品看着还行，但我怕买回来踩雷，你别跟我说套话。",
            "我已经看晕了，你直接说这款值不值得买。",
            "别给我一堆营销词，我只想知道它到底适不适合通勤。",
        ]

        for _ in range(count):
            row = self.product_df.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]
            product_name = row["Product_Title"]
            features = self._format_features(row.get("Features", row.get("feature_list", "")))
            prompt = f"{product_name}。{random.choice(prompts)}"

            chosen = (
                f"理解，您担心的是被营销话术带偏，我直接按现有信息说。**{product_name}** 可以这样判断：\n\n"
                f"- 价格：{self.CURRENCY_SYMBOL}{row['Price']:.2f}\n"
                f"- 评分：{row['Rating']}/5（{int(row['Rating_Count'])}条评价）\n"
                f"- 我看到的主要特点：\n{features}\n\n"
                "如果您主要看重通勤、稳定和性价比，这类信息已经足够做第一轮筛选。"
                "如果您愿意，我可以继续把“适合买 / 不太建议买”的理由拆开说。"
            )

            rejected = random.choice(
                [
                    "你自己都看了这么久了，还问这个？喜欢就买，不喜欢就算了。",
                    "这款绝对值，别犹豫，马上下单就对了，不需要再看参数。",
                    "这种问题没法回答，自己去商品页看就行。",
                ]
            )

            results.append(
                self._make_pair(
                    prompt,
                    chosen,
                    rejected,
                    "tone_alignment",
                    metadata={
                        "task": "brand_tone",
                        "product_id": str(row.get("Product_ID", "")),
                    },
                )
            )

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self, scale: float = 1.0) -> List[Dict[str, Any]]:
        """
        Generate all DPO preference pairs.

        Args:
            scale: Multiplier for target counts (0.1 for quick test, 1.0 for full).

        Returns:
            All generated preference-pair dicts.
        """
        logger.info("Starting DPO data generation (scale=%.2f) ...", scale)
        self.samples = []

        dimension_generators = {
            "information_quality": self._gen_information_quality,
            "factual_accuracy": self._gen_factual_accuracy,
            "format_compliance": self._gen_format_compliance,
            "safety_honesty": self._gen_safety_honesty,
            "service_boundary": self._gen_service_boundary,
            "tone_alignment": self._gen_tone_alignment,
        }

        for dim, gen_fn in dimension_generators.items():
            target = int(self.DIMENSION_TARGETS[dim] * scale)
            pairs = gen_fn(target)
            self.samples.extend(pairs)
            logger.info("  [%s] generated %d pairs", dim, len(pairs))

        random.shuffle(self.samples)
        logger.info("Total DPO pairs: %d", len(self.samples))
        return self.samples

    def split_train_val(
        self, val_ratio: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split into train / val sets."""
        if not self.samples:
            raise RuntimeError("No samples generated. Call generate_all() first.")
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * (1 - val_ratio))
        train = self.samples[:split_idx]
        val = self.samples[split_idx:]
        logger.info("DPO split: %d train / %d val", len(train), len(val))
        return train, val

    def save(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        val_ratio: float = 0.1,
    ) -> Tuple[Path, Path]:
        """
        Split and persist preference pairs to JSONL.

        Returns:
            (train_path, val_path)
        """
        out_dir = self.project_root / self.DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        tp = Path(train_path) if train_path else out_dir / "dpo_train.jsonl"
        vp = Path(val_path) if val_path else out_dir / "dpo_val.jsonl"

        train, val = self.split_train_val(val_ratio)

        for path, data, label in [(tp, train, "train"), (vp, val, "val")]:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for sample in data:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            logger.info("Saved %d %s DPO pairs to %s", len(data), label, path)

        return tp, vp
