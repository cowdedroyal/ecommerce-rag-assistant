"""
SFT (Supervised Fine-Tuning) data synthesiser.

Loads real product / order data and populates prompt-response templates to produce
Qwen2.5-compatible conversation samples in JSONL format.

Usage:
    generator = SFTDataGenerator(product_path="data/processed/products_zh.csv")
    generator.generate_all()
    generator.save("data/training/sft_train.jsonl", "data/training/sft_val.jsonl")
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .templates import (
    SYSTEM_PROMPT,
    TASK_TEMPLATES,
    CATEGORY_ZH_MAP,
    PRODUCT_ZH_MAP,
    PRICE_RANGES,
    BUDGETS,
    RECIPIENTS,
    OCCASIONS,
    SCENARIOS,
    USE_CASES,
    CATEGORY_USE_CASE_HINTS,
    CATEGORY_FEATURE_HINTS,
    CATEGORY_SCENE_HINTS,
    USER_TYPES,
    AGE_GROUPS,
    FEATURES_KEYWORDS,
    TIME_RANGES,
    ORDER_STATUSES,
    RECOMMENDATION_REASONS,
    BUYING_TIPS,
    RECOMMENDATION_ITEM_TEMPLATE,
    ORDER_ITEM_TEMPLATE,
)

logger = logging.getLogger(__name__)


class SFTDataGenerator:
    """Synthesise SFT training samples from product data and templates."""

    # Default paths relative to the project root
    DEFAULT_PRODUCT_PATH = "data/processed/products_zh.csv"
    DEFAULT_RAW_PRODUCT_PATH = "data/raw/Product_Information_Dataset.csv"
    DEFAULT_ORDER_PATH = "data/processed/processed_orders.csv"
    DEFAULT_RAW_ORDER_PATH = "data/raw/Order_Data_Dataset.csv"
    DEFAULT_OUTPUT_DIR = "data/training"
    CURRENCY_SYMBOL = "¥"
    CLARIFICATION_TARGET_COUNT = 250
    STRUCTURE_REPAIR_TARGET_COUNT = 220
    FOLLOWUP_REPAIR_TARGET_COUNT = 240

    def __init__(
        self,
        product_path: Optional[str] = None,
        order_path: Optional[str] = None,
        project_root: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        """
        Initialise the generator.

        Args:
            product_path: Path to the processed product CSV.  Falls back to the
                          raw product CSV if the processed version does not exist.
            order_path:   Path to the order CSV.
            project_root: Root directory of the project.
            seed:         Random seed for reproducibility.
        """
        self.seed = seed
        random.seed(seed)

        self.project_root = Path(project_root) if project_root else self._find_project_root()

        # Load product data
        self.product_df = self._load_product_data(product_path)
        logger.info("Loaded %d products", len(self.product_df))

        # Load order data
        self.order_df = self._load_order_data(order_path)
        logger.info("Loaded %d orders", len(self.order_df))

        # Storage for generated samples
        self.samples: List[Dict[str, Any]] = []

    @classmethod
    def base_target_total(cls) -> int:
        """Return the nominal total number of SFT samples before scaling."""
        return (
            sum(int(item["target_count"]) for item in TASK_TEMPLATES.values())
            + cls.CLARIFICATION_TARGET_COUNT
            + cls.STRUCTURE_REPAIR_TARGET_COUNT
            + cls.FOLLOWUP_REPAIR_TARGET_COUNT
        )

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_project_root() -> Path:
        """Walk upward from this file to locate the project root (contains setup.py)."""
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "setup.py").exists():
                return parent
        return Path.cwd()

    def _load_product_data(self, path: Optional[str]) -> pd.DataFrame:
        """Load product data, handling both processed and raw formats."""
        if path:
            p = Path(path)
        else:
            p = self.project_root / self.DEFAULT_PRODUCT_PATH
            if not p.exists():
                p = self.project_root / self.DEFAULT_RAW_PRODUCT_PATH

        logger.info("Loading product data from %s", p)
        df = pd.read_csv(p)
        df = df.fillna("")

        # Normalise column names so downstream code works for both raw & processed
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

        # Ensure numeric types
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
        df["Rating_Count"] = pd.to_numeric(df["Rating_Count"], errors="coerce").fillna(0).astype(int)
        df = df.dropna(subset=["Price", "Rating"])
        df = df[df["Price"] > 0]

        return df

    def _load_order_data(self, path: Optional[str]) -> pd.DataFrame:
        """Load order data."""
        if path:
            p = Path(path)
        else:
            p = self.project_root / self.DEFAULT_ORDER_PATH
            if not p.exists():
                p = self.project_root / self.DEFAULT_RAW_ORDER_PATH

        logger.info("Loading order data from %s", p)
        df = pd.read_csv(p)
        df = df.fillna("")

        # Add a synthetic order ID
        if "Order_ID" not in df.columns:
            df["Order_ID"] = [f"ORD{i:06d}" for i in range(1, len(df) + 1)]

        return df

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def _zh_category(self, en_cat: str) -> str:
        """Translate an English category name to Chinese."""
        return CATEGORY_ZH_MAP.get(str(en_cat).strip(), str(en_cat))

    def _zh_product(self, en_name: str) -> str:
        """Translate an English product name to Chinese."""
        return PRODUCT_ZH_MAP.get(str(en_name).strip(), str(en_name))

    def _sample_category_use_case(self, category: str) -> str:
        """Sample a user use-case that stays within the current category domain."""
        category_zh = self._zh_category(category)
        return random.choice(CATEGORY_USE_CASE_HINTS.get(category_zh, USE_CASES))

    def _sample_category_feature(self, category: str) -> str:
        """Sample a preference keyword that matches the current category."""
        category_zh = self._zh_category(category)
        return random.choice(CATEGORY_FEATURE_HINTS.get(category_zh, FEATURES_KEYWORDS))

    def _sample_category_scene(self, category: str, preferred: Optional[str] = None) -> str:
        """Sample a scene hint that is consistent with the current category."""
        category_zh = self._zh_category(category)
        candidates = list(CATEGORY_SCENE_HINTS.get(category_zh, SCENARIOS))
        if preferred and preferred not in candidates:
            candidates.insert(0, preferred)
        return random.choice(candidates)

    def _sample_products(self, n: int = 3, category: Optional[str] = None) -> pd.DataFrame:
        """Sample *n* products, optionally from a given category."""
        df = self.product_df
        if category and category in df["Category"].values:
            df = df[df["Category"] == category]
        if len(df) < n:
            n = len(df)
        return df.sample(n=n, random_state=random.randint(0, 1_000_000))

    def _format_features(self, raw: str, max_items: int = 4) -> str:
        """Parse a raw feature string and return a Chinese-friendly bullet list."""
        if not raw or raw == "":
            return "- 暂无详细特点信息"
        items = [s.strip().strip("'\"") for s in str(raw).strip("[]").split(",")]
        items = [i for i in items if len(i) > 2][:max_items]
        if not items:
            return "- 暂无详细特点信息"
        return "\n".join(f"- {item}" for item in items)

    def _build_structured_recommendation_item(
        self,
        idx: int,
        row: pd.Series,
        scene_hint: str,
        reason: str,
    ) -> str:
        """Build a single structured recommendation item with stable four-field formatting."""
        return "\n".join(
            [
                f"{idx}. **{row['Product_Title']}**",
                f"- 价格：{self.CURRENCY_SYMBOL}{row['Price']:.2f}",
                f"- 评分：{row['Rating']}/5（{int(row['Rating_Count'])}条评价）",
                f"- 适用场景：{scene_hint}",
                f"- 推荐理由：{reason}",
            ]
        )

    def _sample_budget_for_category(self, category: str) -> int:
        """Sample a realistic budget from the current category price distribution."""
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
        allow_above_budget: bool = False,
    ) -> pd.DataFrame:
        """Sample products that are aligned to the requested budget when possible."""
        df = self.product_df[self.product_df["Category"] == category]
        if budget is not None:
            within_budget = df[df["Price"] <= budget]
            if len(within_budget) >= n:
                df = within_budget
            elif not allow_above_budget and not within_budget.empty:
                df = within_budget

        if len(df) < n:
            df = self.product_df[self.product_df["Category"] == category]

        unique_df = df.drop_duplicates(subset=["Product_Title"], keep="first")
        if len(unique_df) >= n:
            df = unique_df

        n = min(n, len(df))
        return df.sample(n=n, random_state=random.randint(0, 1_000_000))

    def _sample_products_strict_budget(
        self,
        category: str,
        budget: int,
        n: int,
    ) -> pd.DataFrame:
        """Sample products that are strictly within budget for failure-repair data."""
        df = self.product_df[self.product_df["Category"] == category]
        within_budget = df[df["Price"] <= budget].drop_duplicates(subset=["Product_Title"], keep="first")
        if len(within_budget) >= n:
            df = within_budget
        else:
            df = within_budget
        n = min(n, len(df))
        if n == 0:
            return df.head(0)
        return df.sample(n=n, random_state=random.randint(0, 1_000_000))

    def _build_metadata(
        self,
        task: str,
        scenario: str,
        products: Optional[pd.DataFrame] = None,
        orders: Optional[pd.DataFrame] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Attach light-weight metadata so evaluation can slice the dataset by capability."""
        metadata: Dict[str, Any] = {
            "task": task,
            "scenario": scenario,
            "tags": tags or [],
        }
        if products is not None and not products.empty:
            metadata["product_ids"] = products["Product_ID"].astype(str).tolist()
            metadata["categories"] = sorted(set(products["Category"].astype(str).tolist()))
        if orders is not None and not orders.empty:
            metadata["order_ids"] = orders["Order_ID"].astype(str).tolist()
        return metadata

    def _build_message(
        self,
        user_content: str,
        assistant_content: str,
        system_content: str = SYSTEM_PROMPT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a single Qwen2.5 messages-format sample."""
        payload = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }
        if metadata:
            payload["metadata"] = metadata
        return payload

    def _build_multi_turn_message(
        self,
        turns: List[Tuple[str, str]],
        system_content: str = SYSTEM_PROMPT,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a multi-turn conversation sample."""
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
        for user_text, assistant_text in turns:
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": assistant_text})
        payload: Dict[str, Any] = {"messages": messages}
        if metadata:
            payload["metadata"] = metadata
        return payload

    # ------------------------------------------------------------------
    # Per-task generators
    # ------------------------------------------------------------------

    def _generate_recommendation_samples(self, count: int) -> List[Dict[str, Any]]:
        """Generate product recommendation samples."""
        logger.info("Generating %d recommendation samples ...", count)
        results: List[Dict[str, Any]] = []
        categories = [c for c in self.product_df["Category"].unique() if str(c).strip()]

        for _ in range(count):
            category_en = random.choice(categories)
            category_zh = self._zh_category(category_en)
            budget = self._sample_budget_for_category(category_en)
            use_case = self._sample_category_use_case(category_en)
            feature = self._sample_category_feature(category_en)
            user_text = (
                f"预算控制在{budget}元以内，主要用于{use_case}，更看重{feature}。"
                f"请按“价格、评分、适用场景、推荐理由”列出3款{category_zh}，不要重复商品。"
            )

            # Pick products for the answer
            prods = self._sample_products_strict_budget(
                category=category_en,
                budget=budget,
                n=3,
            )
            if len(prods) < 3:
                continue

            product_items: List[str] = []
            for idx, (_, row) in enumerate(prods.iterrows(), 1):
                scene_hint = self._sample_category_scene(category_en, preferred=use_case)
                reason = (
                    f"价格在预算内，结合{feature}与当前评分表现，"
                    f"更适合{scene_hint}场景。"
                )
                product_items.append(
                    self._build_structured_recommendation_item(
                        idx=idx,
                        row=row,
                        scene_hint=scene_hint,
                        reason=reason,
                    )
                )

            assistant_text = (
                f"按您“{budget}元以内 + {use_case} + 更看重{feature}”的要求，"
                f"我整理了3款预算内且不重复的{category_zh}：\n\n"
                + "\n\n".join(product_items)
                + "\n\n以上为本次推荐结果。"
            )

            results.append(
                self._build_message(
                    user_text,
                    assistant_text,
                    metadata=self._build_metadata(
                        task="product_recommendation",
                        scenario="budget_grounded_recommendation",
                        products=prods,
                        tags=["grounded", "budget_aligned", "structured"],
                    ),
                )
            )

        return results

    def _generate_comparison_samples(self, count: int) -> List[Dict[str, Any]]:
        """Generate product comparison samples."""
        logger.info("Generating %d comparison samples ...", count)
        templates = TASK_TEMPLATES["product_comparison"]
        user_tpls = templates["user_templates"]

        results: List[Dict[str, Any]] = []
        categories = [c for c in self.product_df["Category"].unique() if str(c).strip()]

        for _ in range(count):
            category_en = random.choice(categories)
            prods = self._sample_products_for_budget(category=category_en, budget=None, n=2, allow_above_budget=True)
            if len(prods) < 2:
                continue
            rows = list(prods.iterrows())
            ra, rb = rows[0][1], rows[1][1]

            pa_name = ra["Product_Title"]
            pb_name = rb["Product_Title"]

            user_tpl = random.choice(user_tpls)
            try:
                user_text = user_tpl.format(product_a=pa_name, product_b=pb_name)
            except KeyError:
                user_text = f"{pa_name}和{pb_name}哪个好？"

            features_a = self._format_features(ra.get("Features", ra.get("feature_list", "")))
            features_b = self._format_features(rb.get("Features", rb.get("feature_list", "")))

            # Determine advantages
            adv_a_parts: List[str] = []
            adv_b_parts: List[str] = []
            if ra["Price"] < rb["Price"]:
                adv_a_parts.append("价格更低")
                adv_b_parts.append("品质可能更高")
            else:
                adv_b_parts.append("价格更低")
                adv_a_parts.append("品质可能更高")
            if ra["Rating"] >= rb["Rating"]:
                adv_a_parts.append("评分更高")
            else:
                adv_b_parts.append("评分更高")
            if ra["Rating_Count"] >= rb["Rating_Count"]:
                adv_a_parts.append("评价数量更多，参考价值大")
            else:
                adv_b_parts.append("评价数量更多，参考价值大")

            adv_a = "\n".join(f"- {a}" for a in adv_a_parts) if adv_a_parts else "- 无明显优势"
            adv_b = "\n".join(f"- {a}" for a in adv_b_parts) if adv_b_parts else "- 无明显优势"

            # Price / rating comparison sentences
            price_cmp = (
                f"{pa_name}售价{self.CURRENCY_SYMBOL}{ra['Price']:.2f}，"
                f"{pb_name}售价{self.CURRENCY_SYMBOL}{rb['Price']:.2f}，"
                + ("前者更便宜。" if ra["Price"] < rb["Price"] else "后者更便宜。")
            )
            rating_cmp = (
                f"{pa_name}评分{ra['Rating']}/5，{pb_name}评分{rb['Rating']}/5，"
                + ("前者评分更高。" if ra["Rating"] >= rb["Rating"] else "后者评分更高。")
            )
            feature_cmp = f"{pa_name}的特点包括：{features_a}；{pb_name}的特点包括：{features_b}。"

            if ra["Price"] < rb["Price"] and ra["Rating"] >= rb["Rating"]:
                summary = f"{pa_name}在价格和评分上都更占优，优先推荐前者。"
            elif rb["Price"] < ra["Price"] and rb["Rating"] >= ra["Rating"]:
                summary = f"{pb_name}在价格和评分上都更占优，优先推荐后者。"
            else:
                summary = (
                    f"如果更看重价格，建议选择{'前者' if ra['Price'] < rb['Price'] else '后者'}；"
                    f"如果更看重评分，建议选择{'前者' if ra['Rating'] >= rb['Rating'] else '后者'}。"
                )

            scene_a = self._sample_category_scene(category_en)
            scene_b = self._sample_category_scene(category_en)
            assistant_text = (
                f"下面从价格、评分和适用场景三个方面对比 **{pa_name}** 与 **{pb_name}**：\n\n"
                f"**{pa_name}**\n"
                f"- 价格：{self.CURRENCY_SYMBOL}{ra['Price']:.2f}\n"
                f"- 评分：{ra['Rating']}/5（{int(ra['Rating_Count'])}条评价）\n"
                f"- 适用场景：{scene_a}\n"
                f"- 主要特点：{features_a}\n\n"
                f"**{pb_name}**\n"
                f"- 价格：{self.CURRENCY_SYMBOL}{rb['Price']:.2f}\n"
                f"- 评分：{rb['Rating']}/5（{int(rb['Rating_Count'])}条评价）\n"
                f"- 适用场景：{scene_b}\n"
                f"- 主要特点：{features_b}\n\n"
                f"**结论：**\n"
                f"- 价格：{'前者更便宜' if ra['Price'] < rb['Price'] else '后者更便宜'}\n"
                f"- 评分：{'前者更高' if ra['Rating'] >= rb['Rating'] else '后者更高'}\n"
                f"- 综合建议：{summary}\n"
                f"以上为本次对比结论。"
            )

            results.append(
                self._build_message(
                    user_text,
                    assistant_text,
                    metadata=self._build_metadata(
                        task="product_comparison",
                        scenario="structured_comparison",
                        products=prods,
                        tags=["grounded", "comparison", "structured"],
                    ),
                )
            )

        return results

    def _generate_spec_inquiry_samples(self, count: int) -> List[Dict[str, Any]]:
        """Generate spec inquiry samples."""
        logger.info("Generating %d spec inquiry samples ...", count)
        templates = TASK_TEMPLATES["spec_inquiry"]
        user_tpls = templates["user_templates"]
        resp_tpls = templates["response_templates"]

        results: List[Dict[str, Any]] = []

        for _ in range(count):
            row = self.product_df.sample(n=1, random_state=random.randint(0, 1_000_000)).iloc[0]
            product_name = row["Product_Title"]

            user_tpl = random.choice(user_tpls)
            try:
                user_text = user_tpl.format(product_name=product_name)
            except KeyError:
                user_text = f"这款{product_name}的具体参数是什么？"

            features = self._format_features(row.get("Features", row.get("feature_list", "")))
            desc_raw = str(row.get("Description", ""))
            description = desc_raw[:500] if desc_raw else "暂无详细描述。"

            resp_tpl = random.choice(resp_tpls)
            try:
                assistant_text = resp_tpl.format(
                    product_name=product_name,
                    store=row.get("Store", "未知"),
                    category=self._zh_category(row["Category"]),
                    currency=self.CURRENCY_SYMBOL,
                    price=f'{row["Price"]:.2f}',
                    rating=row["Rating"],
                    rating_count=int(row["Rating_Count"]),
                    features=features,
                    description=description,
                )
            except KeyError:
                assistant_text = (
                    f"**{product_name}**\n- 价格：{self.CURRENCY_SYMBOL}{row['Price']:.2f}\n"
                    f"- 评分：{row['Rating']}/5\n{features}"
                )

            results.append(
                self._build_message(
                    user_text,
                    assistant_text,
                    metadata=self._build_metadata(
                        task="spec_inquiry",
                        scenario="product_spec_grounding",
                        products=pd.DataFrame([row]),
                        tags=["grounded", "spec", "structured"],
                    ),
                )
            )

        return results

    def _generate_shopping_advice_samples(self, count: int) -> List[Dict[str, Any]]:
        """Generate shopping advice samples."""
        logger.info("Generating %d shopping advice samples ...", count)
        templates = TASK_TEMPLATES["shopping_advice"]
        user_tpls = [tpl for tpl in templates["user_templates"] if "{category}" in tpl]
        resp_tpls = templates["response_templates"]
        item_tpl = RECOMMENDATION_ITEM_TEMPLATE

        results: List[Dict[str, Any]] = []
        categories = [c for c in self.product_df["Category"].unique() if str(c).strip()]

        for _ in range(count):
            category_en = random.choice(categories)
            category_zh = self._zh_category(category_en)
            recipient = random.choice(RECIPIENTS)
            occasion = random.choice(OCCASIONS)
            scenario = random.choice(SCENARIOS)
            budget = self._sample_budget_for_category(category_en)
            user_type = random.choice(USER_TYPES)
            age_group = random.choice(AGE_GROUPS)

            user_tpl = random.choice(user_tpls)
            try:
                user_text = user_tpl.format(
                    recipient=recipient,
                    category=category_zh,
                    budget=budget,
                    occasion=occasion,
                    scenario=scenario,
                    user_type=user_type,
                    age_group=age_group,
                )
            except KeyError:
                user_text = f"送{recipient}什么{category_zh}好？"

            # Build product recommendations for the answer
            prods = self._sample_products_for_budget(
                category=category_en,
                budget=budget,
                n=random.randint(2, 3),
            )
            product_items: List[str] = []
            for idx, (_, row) in enumerate(prods.iterrows(), 1):
                product_items.append(
                    item_tpl.format(
                        idx=idx,
                        product_name=row["Product_Title"],
                        currency=self.CURRENCY_SYMBOL,
                        price=f'{row["Price"]:.2f}',
                        rating=row["Rating"],
                        rating_count=int(row["Rating_Count"]),
                        features=self._format_features(
                            row.get("Features", row.get("feature_list", ""))
                        ),
                        reason=random.choice(RECOMMENDATION_REASONS),
                    )
                )

            tips = random.sample(BUYING_TIPS, k=min(3, len(BUYING_TIPS)))
            tips_text = "\n".join(f"{i}. {t}" for i, t in enumerate(tips, 1))

            if age_group in user_text:
                scenario_desc = f"给{age_group}用户挑选{category_zh}"
            elif recipient in user_text:
                scenario_desc = f"给{recipient}挑选{category_zh}"
            elif user_type in user_text:
                scenario_desc = f"{user_type}选购{category_zh}"
            elif scenario in user_text:
                scenario_desc = f"{scenario}场景下选购{category_zh}"
            else:
                scenario_desc = f"{category_zh}选购"

            if scenario in user_text:
                need_analysis = f"根据您提到的{scenario}场景，我优先筛选了更匹配的{category_zh}产品"
            elif "新手" in user_text or "第一次" in user_text or "入门" in user_text:
                need_analysis = f"您现在更需要上手门槛低、价格友好的{category_zh}，所以我优先推荐了更适合入门的款式"
            else:
                need_analysis = f"根据您的需求，我优先筛选了适合当前场景的{category_zh}产品"
            extra_tip = "购买前建议查看其他用户的评价，选择最适合自己需求的产品。"
            summary_text = f"综合考虑预算和需求，以上产品都是不错的选择，建议优先关注评分最高的那款。"

            resp_tpl = random.choice(resp_tpls)
            try:
                assistant_text = resp_tpl.format(
                    scenario_description=scenario_desc,
                    buying_tips=tips_text,
                    product_list="\n\n".join(product_items),
                    extra_tip=extra_tip,
                    need_analysis=need_analysis,
                    summary=summary_text,
                )
            except KeyError:
                assistant_text = (
                    f"关于{scenario_desc}：\n\n{tips_text}\n\n"
                    + "\n\n".join(product_items)
                )

            results.append(
                self._build_message(
                    user_text,
                    assistant_text,
                    metadata=self._build_metadata(
                        task="shopping_advice",
                        scenario="scenario_based_recommendation",
                        products=prods,
                        tags=["grounded", "shopping_advice", "structured"],
                    ),
                )
            )

        return results

    def _generate_order_inquiry_samples(self, count: int) -> List[Dict[str, Any]]:
        """Generate order inquiry samples."""
        logger.info("Generating %d order inquiry samples ...", count)
        templates = TASK_TEMPLATES["order_inquiry"]
        user_tpls = templates["user_templates"]
        resp_tpls = templates["response_templates"]

        results: List[Dict[str, Any]] = []

        for _ in range(count):
            order_row = self.order_df.sample(
                n=1, random_state=random.randint(0, 1_000_000)
            ).iloc[0]

            product_zh = self._zh_product(str(order_row.get("Product", "")))
            customer_id = str(order_row.get("Customer_Id", "00000"))
            order_id = str(order_row.get("Order_ID", "ORD000000"))
            order_date = str(order_row.get("Order_Date", ""))
            quantity = str(order_row.get("Quantity", 1))
            sales = f'{float(order_row.get("Sales", 0)):.2f}'
            discount = str(order_row.get("Discount", "0"))
            shipping_cost = f'{float(order_row.get("Shipping_Cost", 0)):.2f}'
            payment_method = str(order_row.get("Payment_Method", ""))
            order_priority = str(order_row.get("Order_Priority", ""))
            category = self._zh_category(str(order_row.get("Product_Category", "")))
            status = str(order_row.get("Shipping_Status", "")) or random.choice(ORDER_STATUSES)
            time_range = random.choice(TIME_RANGES)

            user_tpl = random.choice(user_tpls)
            try:
                user_text = user_tpl.format(
                    customer_id=customer_id,
                    product=product_zh,
                    order_id=order_id,
                    time_range=time_range,
                )
            except KeyError:
                user_text = f"查一下我的订单，客户编号是{customer_id}"

            # For multi-order listing template, build a list
            nearby_orders = self.order_df[
                self.order_df["Customer_Id"] == order_row.get("Customer_Id")
            ].head(3)
            order_items: List[str] = []
            for oidx, (_, orow) in enumerate(nearby_orders.iterrows(), 1):
                order_items.append(
                    ORDER_ITEM_TEMPLATE.format(
                        idx=oidx,
                        product=self._zh_product(str(orow.get("Product", ""))),
                        currency=self.CURRENCY_SYMBOL,
                        sales=f'{float(orow.get("Sales", 0)):.2f}',
                        order_date=str(orow.get("Order_Date", "")),
                        quantity=str(orow.get("Quantity", 1)),
                        status=str(orow.get("Shipping_Status", "")) or random.choice(ORDER_STATUSES),
                    )
                )

            resp_tpl = random.choice(resp_tpls)
            try:
                assistant_text = resp_tpl.format(
                    order_id=order_id,
                    order_date=order_date,
                    product=product_zh,
                    category=category,
                    quantity=quantity,
                    currency=self.CURRENCY_SYMBOL,
                    sales=sales,
                    discount=discount,
                    shipping_cost=shipping_cost,
                    payment_method=payment_method,
                    order_priority=order_priority,
                    status=status,
                    order_list="\n\n".join(order_items),
                )
            except KeyError:
                assistant_text = (
                    f"您的订单 {order_id}：{product_zh}，"
                    f"金额 {self.CURRENCY_SYMBOL}{sales}，状态：{status}"
                )

            results.append(
                self._build_message(
                    user_text,
                    assistant_text,
                    metadata=self._build_metadata(
                        task="order_inquiry",
                        scenario="order_lookup",
                        orders=pd.DataFrame([order_row]),
                        tags=["order", "lookup", "structured"],
                    ),
                )
            )

        return results

    def _generate_multi_turn_samples(self, count: int) -> List[Dict[str, Any]]:
        """Generate multi-turn conversation samples."""
        logger.info("Generating %d multi-turn samples ...", count)
        templates = TASK_TEMPLATES["multi_turn"]
        first_user_tpls = templates["first_user_templates"]
        followup_user_tpls = templates["followup_user_templates"]
        first_resp_tpls = templates["first_response_templates"]
        followup_resp_tpls = templates["followup_response_templates"]
        item_tpl = templates["item_template"]

        results: List[Dict[str, Any]] = []
        categories = [c for c in self.product_df["Category"].unique() if str(c).strip()]

        for _ in range(count):
            category_en = random.choice(categories)
            category_zh = self._zh_category(category_en)

            # --- Turn 1: initial recommendation ---
            first_user_tpl = random.choice(first_user_tpls)
            try:
                first_user_text = first_user_tpl.format(category=category_zh)
            except KeyError:
                first_user_text = f"推荐一款{category_zh}"

            prods = self._sample_products(n=random.randint(2, 4), category=category_en)
            prod_rows = list(prods.iterrows())
            product_items: List[str] = []
            for idx, (_, row) in enumerate(prod_rows, 1):
                product_items.append(
                    item_tpl.format(
                        idx=idx,
                        product_name=row["Product_Title"],
                        currency=self.CURRENCY_SYMBOL,
                        price=f'{row["Price"]:.2f}',
                        rating=row["Rating"],
                        rating_count=int(row["Rating_Count"]),
                        features=self._format_features(
                            row.get("Features", row.get("feature_list", ""))
                        ),
                        reason=random.choice(RECOMMENDATION_REASONS),
                    )
                )

            first_resp_tpl = random.choice(first_resp_tpls)
            try:
                first_assistant_text = first_resp_tpl.format(
                    category=category_zh,
                    product_list="\n\n".join(product_items),
                    budget_or_requirement=category_zh,
                )
            except KeyError:
                first_assistant_text = (
                    f"推荐以下{category_zh}：\n\n" + "\n\n".join(product_items)
                )

            # --- Turn 2: follow-up question ---
            picked_idx = random.randint(1, len(prod_rows))
            alt_idx = random.choice([i for i in range(1, len(prod_rows) + 1) if i != picked_idx]) if len(prod_rows) > 1 else picked_idx
            picked_row = prod_rows[picked_idx - 1][1]

            followup_user_tpl = random.choice(followup_user_tpls)
            try:
                followup_user_text = followup_user_tpl.format(
                    idx=picked_idx,
                    alt_idx=alt_idx,
                    extra_budget=random.choice(["200", "500", "1000"]),
                    use_case=random.choice(USE_CASES),
                    feature=random.choice(FEATURES_KEYWORDS),
                )
            except KeyError:
                followup_user_text = f"第{picked_idx}款具体参数是什么？"

            followup_resp_tpl = random.choice(followup_resp_tpls)
            try:
                # Prepare variables for all possible followup templates
                alt_row = prod_rows[alt_idx - 1][1] if alt_idx <= len(prod_rows) else picked_row
                followup_assistant_text = followup_resp_tpl.format(
                    idx=picked_idx,
                    alt_idx=alt_idx,
                    product_name=picked_row["Product_Title"],
                    product_a=picked_row["Product_Title"],
                    product_b=alt_row["Product_Title"],
                    currency=self.CURRENCY_SYMBOL,
                    price=f'{picked_row["Price"]:.2f}',
                    price_a=f'{picked_row["Price"]:.2f}',
                    price_b=f'{alt_row["Price"]:.2f}',
                    rating=picked_row["Rating"],
                    rating_a=picked_row["Rating"],
                    rating_b=alt_row["Rating"],
                    rating_count=int(picked_row["Rating_Count"]),
                    rating_count_a=int(picked_row["Rating_Count"]),
                    rating_count_b=int(alt_row["Rating_Count"]),
                    category=self._zh_category(picked_row["Category"]),
                    features=self._format_features(
                        picked_row.get("Features", picked_row.get("feature_list", ""))
                    ),
                    description=(str(picked_row.get("Description", ""))[:300] or "暂无详细描述。"),
                    advantages_a="价格更具竞争力" if picked_row["Price"] < alt_row["Price"] else "评分更高",
                    advantages_b="价格更具竞争力" if alt_row["Price"] < picked_row["Price"] else "评分更高",
                    suggestion="建议根据您的实际需求和预算做最终决定。",
                )
            except KeyError:
                followup_assistant_text = (
                    f"第{picked_idx}款 **{picked_row['Product_Title']}**\n"
                    f"- 价格：{self.CURRENCY_SYMBOL}{picked_row['Price']:.2f}\n"
                    f"- 评分：{picked_row['Rating']}/5"
                )

            turns = [
                (first_user_text, first_assistant_text),
                (followup_user_text, followup_assistant_text),
            ]
            results.append(
                self._build_multi_turn_message(
                    turns,
                    metadata=self._build_metadata(
                        task="multi_turn",
                        scenario="followup_grounded_memory",
                        products=prods,
                        tags=["multi_turn", "grounded", "followup"],
                    ),
                )
            )

        return results

    def _generate_clarification_samples(self, count: int) -> List[Dict[str, Any]]:
        """Generate multi-turn samples where the assistant clarifies before recommending."""
        logger.info("Generating %d clarification samples ...", count)
        results: List[Dict[str, Any]] = []
        categories = [c for c in self.product_df["Category"].unique() if str(c).strip()]
        clarification_questions = [
            "可以的。为了推荐得更准确，我先确认两点：您的预算大概多少？更偏向什么使用场景？",
            "没问题。我先帮您缩小范围：预算大概在什么区间？您更看重续航、便携还是性价比？",
            "可以，我先确认一下需求：您准备控制在多少预算内，主要是日常通勤、办公还是运动使用？",
        ]

        for _ in range(count):
            category_en = random.choice(categories)
            category_zh = self._zh_category(category_en)
            budget = self._sample_budget_for_category(category_en)
            scenario = self._sample_category_use_case(category_en)
            feature = self._sample_category_feature(category_en)

            initial_user_text = random.choice(
                [
                    f"想买个{category_zh}，你帮我看看。",
                    f"最近想入手{category_zh}，但不知道怎么选。",
                    f"{category_zh}有什么值得买的吗？",
                ]
            )
            clarification_text = random.choice(clarification_questions)
            followup_user_text = f"预算先控制在{budget}元左右，主要用于{scenario}，最好有{feature}。"

            prods = self._sample_products_for_budget(
                category=category_en,
                budget=budget,
                n=random.randint(2, 3),
            )
            item_texts: List[str] = []
            for idx, (_, row) in enumerate(prods.iterrows(), 1):
                item_texts.append(
                    RECOMMENDATION_ITEM_TEMPLATE.format(
                        idx=idx,
                        product_name=row["Product_Title"],
                        currency=self.CURRENCY_SYMBOL,
                        price=f'{row["Price"]:.2f}',
                        rating=row["Rating"],
                        rating_count=int(row["Rating_Count"]),
                        features=self._format_features(
                            row.get("Features", row.get("feature_list", ""))
                        ),
                        reason=random.choice(RECOMMENDATION_REASONS),
                    )
                )

            final_answer = (
                f"了解了，按照您“{budget}元左右 + {scenario} + 偏向{feature}”的需求，"
                f"我更建议优先看下面这几款{category_zh}：\n\n"
                + "\n\n".join(item_texts)
                + "\n\n如果您愿意，我下一步可以继续帮您把这几款按“更便宜 / 更耐用 / 更适合新手”再细分一次。"
            )
            turns = [
                (initial_user_text, clarification_text),
                (followup_user_text, final_answer),
            ]
            results.append(
                self._build_multi_turn_message(
                    turns,
                    metadata=self._build_metadata(
                        task="clarification_flow",
                        scenario="clarify_then_recommend",
                        products=prods,
                        tags=["multi_turn", "clarification", "grounded", "workflow"],
                    ),
                )
            )

        return results

    def _generate_structure_repair_samples(self, count: int) -> List[Dict[str, Any]]:
        """Generate stricter structured recommendation samples to repair missed-field failures."""
        logger.info("Generating %d structure-repair samples ...", count)
        results: List[Dict[str, Any]] = []
        categories = [c for c in self.product_df["Category"].unique() if str(c).strip()]

        for _ in range(count):
            category_en = random.choice(categories)
            category_zh = self._zh_category(category_en)
            budget = self._sample_budget_for_category(category_en)
            scenario = self._sample_category_use_case(category_en)
            feature = self._sample_category_feature(category_en)

            prods = self._sample_products_strict_budget(
                category=category_en,
                budget=budget,
                n=3,
            )
            if len(prods) < 2:
                continue

            user_text = (
                f"预算控制在{budget}元以内，主要用于{scenario}，更看重{feature}。"
                f"请按“价格、评分、适用场景、推荐理由”列出3款{category_zh}，不要重复商品。"
            )

            item_texts: List[str] = []
            used_titles = set()
            for idx, (_, row) in enumerate(prods.iterrows(), 1):
                title = str(row["Product_Title"])
                if title in used_titles:
                    continue
                used_titles.add(title)
                scene_hint = self._sample_category_scene(category_en, preferred=scenario)
                item_texts.append(
                    "\n".join(
                        [
                            f"{idx}. **{title}**",
                            f"- 价格：{self.CURRENCY_SYMBOL}{row['Price']:.2f}",
                            f"- 评分：{row['Rating']}/5（{int(row['Rating_Count'])}条评价）",
                            f"- 适用场景：{scene_hint}",
                            f"- 推荐理由：价格在预算内，结合{feature}与当前评分表现，更适合{scene_hint}场景。",
                        ]
                    )
                )

            assistant_text = (
                f"按您“{budget}元以内 + {scenario} + 更看重{feature}”的要求，"
                f"我先筛出 3 款不重复且预算内的{category_zh}：\n\n"
                + "\n\n".join(item_texts)
                + "\n\n以上为本次推荐结果。"
            )

            results.append(
                self._build_message(
                    user_text,
                    assistant_text,
                    metadata=self._build_metadata(
                        task="failure_repair_sft",
                        scenario="structured_recommendation_repair",
                        products=prods,
                        tags=["p0", "structured", "budget_aligned", "no_duplicates"],
                    ),
                )
            )

        return results

    def _generate_followup_repair_samples(self, count: int) -> List[Dict[str, Any]]:
        """Generate multi-turn samples where the assistant answers the follow-up directly first."""
        logger.info("Generating %d followup-repair samples ...", count)
        results: List[Dict[str, Any]] = []
        categories = [c for c in self.product_df["Category"].unique() if str(c).strip()]

        for _ in range(count):
            category_en = random.choice(categories)
            category_zh = self._zh_category(category_en)
            budget = self._sample_budget_for_category(category_en)
            primary_need = random.choice(FEATURES_KEYWORDS)
            scenario = random.choice(SCENARIOS)

            prods = self._sample_products_strict_budget(
                category=category_en,
                budget=budget,
                n=3,
            )
            if len(prods) < 2:
                continue

            prod_rows = list(prods.iterrows())
            item_texts: List[str] = []
            for idx, (_, row) in enumerate(prod_rows, 1):
                item_texts.append(
                    "\n".join(
                        [
                            f"{idx}. **{row['Product_Title']}**",
                            f"- 价格：{self.CURRENCY_SYMBOL}{row['Price']:.2f}",
                            f"- 评分：{row['Rating']}/5",
                            f"- 适用场景：{scenario}",
                            f"- 推荐理由：更偏向{primary_need}，适合作为第一轮候选。",
                        ]
                    )
                )

            first_user_text = (
                f"预算{budget}元以内，主要用于{scenario}，更看重{primary_need}。"
                f"先推荐三款{category_zh}给我。"
            )
            first_assistant_text = (
                f"可以，先给您三款预算内候选：\n\n"
                + "\n\n".join(item_texts)
                + "\n\n以上为本轮推荐结果。"
            )

            chosen_idx = random.choice([1, 2])
            alt_idx = 2 if chosen_idx == 1 else 1
            chosen_row = prod_rows[chosen_idx - 1][1]
            alt_row = prod_rows[alt_idx - 1][1]
            focus = random.choice(["通勤", "安静", "性价比", "稳定性", primary_need])
            followup_user_text = (
                f"第{chosen_idx}款和第{alt_idx}款相比，哪一款更适合{focus}？"
                "你直接给结论，再告诉我原因。"
            )
            followup_assistant_text = (
                f"如果主要看{focus}，我更建议第{chosen_idx}款 **{chosen_row['Product_Title']}**。\n\n"
                f"原因1：它的价格是 {self.CURRENCY_SYMBOL}{chosen_row['Price']:.2f}，"
                f"与第{alt_idx}款的 {self.CURRENCY_SYMBOL}{alt_row['Price']:.2f} 相比更容易控制预算。\n"
                f"原因2：当前评分是 {chosen_row['Rating']}/5，作为同一轮候选更稳妥。\n"
                f"如果您更看重另一项需求，第{alt_idx}款 **{alt_row['Product_Title']}** 也可以作为备选。"
            )

            results.append(
                self._build_multi_turn_message(
                    [
                        (first_user_text, first_assistant_text),
                        (followup_user_text, followup_assistant_text),
                    ],
                    metadata=self._build_metadata(
                        task="failure_repair_sft",
                        scenario="followup_direct_answer_repair",
                        products=prods,
                        tags=["p0", "multi_turn", "followup", "direct_answer", "budget_aligned"],
                    ),
                )
            )

        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self, scale: float = 1.0) -> List[Dict[str, Any]]:
        """
        Generate all training samples.

        Args:
            scale: Multiplier for the target counts (0.1 for quick testing, 1.0 for full).

        Returns:
            A list of all generated samples.
        """
        logger.info("Starting SFT data generation (scale=%.2f) ...", scale)
        self.samples = []

        generators = [
            ("product_recommendation", self._generate_recommendation_samples),
            ("product_comparison", self._generate_comparison_samples),
            ("spec_inquiry", self._generate_spec_inquiry_samples),
            ("shopping_advice", self._generate_shopping_advice_samples),
            ("order_inquiry", self._generate_order_inquiry_samples),
            ("multi_turn", self._generate_multi_turn_samples),
        ]

        for task_name, gen_fn in generators:
            target = int(TASK_TEMPLATES[task_name]["target_count"] * scale)
            samples = gen_fn(target)
            self.samples.extend(samples)
            logger.info("  [%s] generated %d samples", task_name, len(samples))

        clarification_target = int(self.CLARIFICATION_TARGET_COUNT * scale)
        clarification_samples = self._generate_clarification_samples(clarification_target)
        self.samples.extend(clarification_samples)
        logger.info("  [%s] generated %d samples", "clarification_flow", len(clarification_samples))

        structure_repair_target = int(self.STRUCTURE_REPAIR_TARGET_COUNT * scale)
        structure_repair_samples = self._generate_structure_repair_samples(structure_repair_target)
        self.samples.extend(structure_repair_samples)
        logger.info("  [%s] generated %d samples", "structured_recommendation_repair", len(structure_repair_samples))

        followup_repair_target = int(self.FOLLOWUP_REPAIR_TARGET_COUNT * scale)
        followup_repair_samples = self._generate_followup_repair_samples(followup_repair_target)
        self.samples.extend(followup_repair_samples)
        logger.info("  [%s] generated %d samples", "followup_direct_answer_repair", len(followup_repair_samples))

        random.shuffle(self.samples)
        logger.info("Total SFT samples: %d", len(self.samples))
        return self.samples

    def split_train_val(
        self, val_ratio: float = 0.1
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split samples into train and validation sets.

        Args:
            val_ratio: Fraction of data to use for validation.

        Returns:
            (train_samples, val_samples)
        """
        if not self.samples:
            raise RuntimeError("No samples generated yet. Call generate_all() first.")

        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * (1 - val_ratio))
        train = self.samples[:split_idx]
        val = self.samples[split_idx:]
        logger.info("Split: %d train / %d val", len(train), len(val))
        return train, val

    def save(
        self,
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        val_ratio: float = 0.1,
    ) -> Tuple[Path, Path]:
        """
        Split and save samples to JSONL files.

        Args:
            train_path: Output path for training set.
            val_path:   Output path for validation set.
            val_ratio:  Fraction of data for validation.

        Returns:
            Tuple of (train_path, val_path) actually written.
        """
        out_dir = self.project_root / self.DEFAULT_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        tp = Path(train_path) if train_path else out_dir / "sft_train.jsonl"
        vp = Path(val_path) if val_path else out_dir / "sft_val.jsonl"

        train, val = self.split_train_val(val_ratio)

        for path, data, label in [(tp, train, "train"), (vp, val, "val")]:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                for sample in data:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            logger.info("Saved %d %s samples to %s", len(data), label, path)

        return tp, vp
