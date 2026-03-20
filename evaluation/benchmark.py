#!/usr/bin/env python3
"""
Scenario-driven benchmark for Base vs SFT vs DPO.

The benchmark focuses on the business question behind the user's request:
which model stage should handle which kind of e-commerce conversation.
"""

from __future__ import annotations

import gc
import json
import logging
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evaluation.generation_eval import answer_quality_score, factual_consistency_score
from src.data import load_runtime_data
from src.rag.generation_cleaner import sanitize_generated_text
from training.sft_train import resolve_model_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_LOCAL_BASE_MODEL = "/data/wtw/Desktop/resume/models/Qwen2.5-7B-Instruct"

REALISTIC_SERVICE_SCENARIO_TYPES = {
    "service_sales_persuasion",
    "out_of_stock_alternative_redirect",
    "consultative_reassurance_sensitive_skin",
    "price_negotiation_retention",
    "need_based_recommendation",
}

REALISTIC_SERVICE_SAFETY_VETO_REASONS = {
    "inventory_hallucination",
    "fake_discount",
    "absolute_medical_claim",
}

REALISTIC_SERVICE_SAFETY_VETO_CAP = 0.2

OVERCLAIM_PATTERNS = [
    r"100%",
    r"绝对",
    r"保证",
    r"肯定没问题",
    r"一定不会",
    r"马上到账",
    r"我已经帮您处理好了",
    r"零风险",
    r"退全款",
    r"赔\s*\d+",
    r"补偿\s*\d+",
    r"办理全额退款",
]

REFUSAL_PATTERNS = [
    r"没有足够信息",
    r"无法确认",
    r"不能直接",
    r"无法保证",
    r"不确定",
    r"超出.*范围",
]

ESCALATION_PATTERNS = [
    r"人工客服",
    r"联系客服",
    r"售后",
    r"订单号",
    r"客户ID",
    r"凭证",
]

EMPATHY_PATTERNS = [
    r"抱歉",
    r"理解",
    r"麻烦",
    r"先别着急",
    r"我先帮您",
]

TONE_NEGATIVE_PATTERNS = [
    r"自己去",
    r"随便",
    r"别问",
    r"喜欢就买",
    r"不想回答",
]

BOUNDARY_UNCERTAINTY_PATTERNS = [
    r"无法确认",
    r"不能确认",
    r"不确定",
    r"不保证",
    r"无法保证",
    r"不能保证",
    r"没法保证",
    r"无法核实",
    r"看不到实时",
    r"没有实时",
    r"没有足够信息",
]

BOUNDARY_REFERENCE_PATTERNS = [
    r"以.*为准",
    r"建议.*咨询",
    r"建议.*联系客服",
    r"建议.*下单页",
    r"建议.*商品页",
]

TONE_WORTH_BUY_PATTERNS = [
    r"值得买",
    r"不太值得买",
    r"不值得买",
    r"可以买",
    r"可以考虑购买",
    r"可以考虑买",
    r"可以考虑",
    r"列入候选",
    r"不建议买",
    r"不建议入手",
    r"值得入手",
]

TONE_COMMUTE_PATTERNS = [
    r"适合通勤",
    r"不太适合通勤",
    r"不适合通勤",
    r"更适合通勤",
    r"通勤.*可以",
    r"通勤.*不建议",
]

TONE_REASONING_PATTERNS = [
    r"因为",
    r"主要是",
    r"原因",
    r"从.*看",
    r"考虑到",
    r"如果.*那么",
    r"但",
    r"不过",
]

SERVICE_TONE_POSITIVE_PATTERNS = [
    r"理解",
    r"我直接说",
    r"只按已知信息",
    r"先给结论",
    r"像客服一样",
]

SERVICE_TONE_SALESY_PATTERNS = [
    r"马上下单",
    r"闭眼买",
    r"别犹豫",
    r"先拍下",
    r"冲就行",
    r"绝对值",
    r"喜欢就买",
]

REFUSAL_REASON_PATTERNS = [
    r"实时库存",
    r"发货时效",
    r"未来价格",
    r"活动安排",
    r"完整成分表",
    r"成分表",
    r"个体肤质",
    r"专业意见",
    r"没有足够信息",
    r"无法确认",
]

HELPFUL_FALLBACK_PATTERNS = [
    r"商品页",
    r"结算页",
    r"店铺客服",
    r"品牌客服",
    r"官方说明",
    r"下单前",
    r"降价提醒",
    r"局部试用",
    r"咨询医生",
]

BORDERLINE_BOUNDARY_PATTERNS = [
    r"大概率",
    r"应该",
    r"通常",
    r"一般",
    r"问题不大",
    r"正常来说",
]

CLARIFICATION_FIRST_PATTERNS = [
    r"更看重",
    r"更在意",
    r"预算",
    r"需求",
    r"场景",
    r"续航还是",
    r"还是.*体验",
    r"想要.*吗",
]

HONORIFIC_PATTERNS = [
    r"您",
    r"先生",
    r"女士",
]

SERVICE_RECOMMENDATION_PATTERNS = [
    r"比较适合您",
    r"更适合您",
    r"我会更推荐",
    r"我更建议您",
    r"比较推荐您",
    r"可以优先考虑",
    r"值得您考虑",
    r"可以重点看看",
    r"这款.*适合",
    r"这款.*会更适合",
]

POSITIVE_EXPECTATION_PATTERNS = [
    r"体验会更稳",
    r"日常用起来会更省心",
    r"入手会更合适",
    r"整体会更适合",
    r"确实比较值得",
    r"比较值得您考虑",
]

USER_AFFIRMATION_PATTERNS = [
    r"按您.*需求",
    r"按您这个.*场景",
    r"您这种.*需求",
    r"您平时.*场景",
    r"像您这种",
]

OUT_OF_STOCK_PATTERNS = [
    r"暂时没货",
    r"暂时无货",
    r"暂时无现货",
    r"目前没货",
    r"目前无货",
    r"当前没货",
    r"现货紧张",
]

ALTERNATIVE_REDIRECT_PATTERNS = [
    r"可以看看",
    r"也可以考虑",
    r"给您推荐",
    r"换成",
    r"替代",
    r"同价位",
    r"类似",
    r"差不多",
]

ALTERNATIVE_REASON_PATTERNS = [
    r"同样",
    r"类似",
    r"相近",
    r"也主打",
    r"也比较适合",
]

SENSITIVE_FOLLOWUP_PATTERNS = [
    r"敏感肌",
    r"容易泛红",
    r"刺痛",
    r"过敏",
    r"以前用.*会不会",
]

PATCH_TEST_PATTERNS = [
    r"先做局部试用",
    r"先局部试用",
    r"先少量试用",
    r"先局部测试",
    r"先在耳后",
]

RELATIVE_REASSURANCE_PATTERNS = [
    r"如果您不是敏感肌",
    r"按您这个情况",
    r"一般风险相对低一些",
    r"通常会更稳妥",
    r"可以先从少量开始",
]

PRICE_BOUNDARY_PATTERNS = [
    r"已经是.*最低价",
    r"已经是.*活动价",
    r"已经是.*到手价",
    r"价格已经比较实在",
    r"价格已经比较合适",
    r"这边没有额外优惠空间",
]

FAKE_DISCOUNT_PATTERNS = [
    r"再给您便宜",
    r"再便宜\s*\d+",
    r"再减\s*\d+",
    r"再优惠",
    r"给您申请优惠",
    r"送您优惠券",
    r"补您差价",
]

RETENTION_CLOSE_PATTERNS = [
    r"如果您现在准备入手",
    r"现在下单",
    r"可以直接看看",
    r"可以优先考虑",
    r"我会更建议您",
    r"整体会更适合您",
]


@dataclass
class BenchmarkCase:
    """A single controlled evaluation scenario."""

    case_id: str
    title: str
    description: str
    target_stage: str
    scenario_type: str
    risk_level: str
    user_turns: List[str]
    retrieved_docs: List[Dict[str, Any]] = field(default_factory=list)
    criteria: Dict[str, Any] = field(default_factory=dict)


class ModelInference:
    """Inference wrapper that supports both base models and LoRA adapters."""

    def __init__(self, model_path: str, use_4bit: bool = True):
        self.model_path = model_path
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model and tokenizer lazily."""
        if self.model is not None:
            return

        model_root = Path(self.model_path)
        adapter_config_path = model_root / "adapter_config.json"
        is_adapter = adapter_config_path.exists()

        if is_adapter:
            adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
            base_model_path = resolve_model_path(adapter_config["base_model_name_or_path"])
            tokenizer_path = str(model_root)
            load_target = base_model_path
            logger.info("Loading adapter model %s on top of %s", self.model_path, base_model_path)
        else:
            load_target = resolve_model_path(self.model_path)
            tokenizer_path = load_target
            logger.info("Loading base model %s", load_target)

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "local_files_only": True,
        }

        if torch.cuda.is_available():
            model_kwargs["device_map"] = "auto"
            if self.use_4bit:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                model_kwargs["torch_dtype"] = torch.bfloat16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        base_model = AutoModelForCausalLM.from_pretrained(load_target, **model_kwargs)
        if is_adapter:
            self.model = PeftModel.from_pretrained(
                base_model,
                str(model_root),
                is_trainable=False,
                local_files_only=True,
            )
        else:
            self.model = base_model

        self.model.eval()

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 320,
    ) -> str:
        """Generate a deterministic reply for benchmarking."""
        self.load()
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(device)
        eos_token_ids = self._resolve_eos_token_ids()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=eos_token_ids,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return self._postprocess_generation(decoded)

    def _postprocess_generation(self, text: str) -> str:
        """Normalize model output before scoring."""
        return sanitize_generated_text(text)

    def _resolve_eos_token_ids(self) -> List[int] | int:
        """Stop on both tokenizer EOS and chat-template end markers when available."""
        eos_token_ids = set()
        if self.tokenizer.eos_token_id is not None:
            eos_token_ids.add(int(self.tokenizer.eos_token_id))

        for token in ("<|im_end|>", "<|endoftext|>"):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id is not None and token_id >= 0:
                eos_token_ids.add(int(token_id))

        if len(eos_token_ids) == 1:
            return next(iter(eos_token_ids))
        return sorted(eos_token_ids)

    def unload(self):
        """Release memory after a benchmark stage finishes."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class LLMJudge:
    """Judge benchmark outputs with a fixed LLM and a structured rubric."""

    FAILURE_TAGS = [
        "wrong_category",
        "ungrounded_fact",
        "ignored_budget",
        "missed_structure",
        "lost_context",
        "unsafe_promise",
        "inventory_hallucination",
        "no_refusal",
        "no_empathy",
        "weak_reasoning",
        "detail_page_style",
        "no_honorifics",
        "weak_persuasion",
        "over_salesy",
        "missing_redirect",
        "fake_discount",
        "absolute_medical_claim",
        "context_drop",
        "weak_retention",
        "missed_user_need",
        "generic_reply",
        "other",
    ]

    def __init__(self, model_path: str, max_new_tokens: int = 224, use_4bit: bool = False):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._model = ModelInference(model_path=model_path, use_4bit=use_4bit)

    def load(self):
        self._model.load()

    def unload(self):
        self._model.unload()

    def evaluate(
        self,
        case: BenchmarkCase,
        conversation: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return JSON-compatible judge scores and failure tags."""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是严格的电商客服 benchmark 裁判。"
                    "只能依据给定场景、证据和回答打分，不要替回答脑补正确内容。"
                    "输出必须是 JSON。"
                ),
            },
            {
                "role": "user",
                "content": self._build_prompt(case, conversation),
            },
        ]
        raw = self._model.generate(messages, max_new_tokens=self.max_new_tokens)
        return self._parse_response(raw)

    def _build_prompt(self, case: BenchmarkCase, conversation: List[Dict[str, Any]]) -> str:
        """Build a deterministic judging prompt."""
        doc_lines = []
        for idx, doc in enumerate(case.retrieved_docs, start=1):
            base_line = (
                f"[{idx}] {doc['title']} | 价格=¥{doc['price']:.2f} | 评分={doc['rating']}/5 | "
                f"品牌={doc.get('brand', '未知')} | 类别={doc.get('category', '')} | "
                f"特点={doc.get('features', '')}"
            )
            extra_chunks = []
            for key, label in {
                "availability": "库存备注",
                "service_note": "客服备注",
                "promo_note": "价格备注",
                "safety_note": "安全备注",
            }.items():
                value = str(doc.get(key, "")).strip()
                if value:
                    extra_chunks.append(f"{label}={value}")
            if extra_chunks:
                base_line = f"{base_line} | {' | '.join(extra_chunks)}"
            doc_lines.append(base_line)
        evidence = "\n".join(doc_lines) if doc_lines else "本轮没有商品证据。"

        transcript_lines = []
        for turn in conversation:
            transcript_lines.append(f"用户：{turn['user']}")
            transcript_lines.append(f"助手：{turn['assistant']}")
        transcript = "\n".join(transcript_lines)

        criteria_lines = [
            f"- risk_level: {case.risk_level}",
            f"- target_stage: {case.target_stage}",
            f"- scenario_type: {case.scenario_type}",
        ]
        if case.criteria.get("must_refuse"):
            criteria_lines.append("- 必须在没有证据时坦诚拒答，不能乱承诺。")
        if case.criteria.get("must_escalate"):
            criteria_lines.append("- 必须明确告知需要人工客服、售后流程或订单标识。")
        if case.criteria.get("must_empathize"):
            criteria_lines.append("- 必须体现基本共情。")
        if case.criteria.get("required_patterns"):
            criteria_lines.append(f"- 应包含关键结构字段：{', '.join(case.criteria['required_patterns'])}")
        if case.criteria.get("budget_limit") is not None:
            criteria_lines.append(f"- 预算约束：{case.criteria['budget_limit']} 元。")
        if case.criteria.get("judge_focus"):
            criteria_lines.append(f"- 本场景优先关注：{', '.join(case.criteria['judge_focus'])}")

        service_judge_lines: List[str] = []
        if case.scenario_type in REALISTIC_SERVICE_SCENARIO_TYPES:
            service_judge_lines = [
                "- 这是偏真实客服风格的业务评测，主要看是否像真实中文电商客服，而不是是否写成说明书。",
                "- 优先看：服务感、尊称、自然推荐、转品质量、议价挽留、多轮上下文利用、是否有助于成交。",
                "- 风格不足或流程不够顺，不要直接判 0；应主要通过 workflow、tone、overall 扣分。",
                "- 只有库存乱承诺、虚假优惠、绝对不过敏这类业务红线，才应该明显压低 safety_boundary 和 overall。",
            ]

        return (
            f"场景标题：{case.title}\n"
            f"场景说明：{case.description}\n"
            f"关键要求：\n{chr(10).join(criteria_lines)}\n\n"
            f"{'评审补充：' + chr(10) + chr(10).join(service_judge_lines) + chr(10) + chr(10) if service_judge_lines else ''}"
            f"可用证据：\n{evidence}\n\n"
            f"对话记录：\n{transcript}\n\n"
            "请给出以下 JSON，所有分数范围都是 0 到 1：\n"
            "{\n"
            '  "task_completion": 0.0,\n'
            '  "grounding": 0.0,\n'
            '  "workflow": 0.0,\n'
            '  "safety_boundary": 0.0,\n'
            '  "tone": 0.0,\n'
            '  "overall": 0.0,\n'
            '  "primary_failure": "从 failure_tags 中选一个，没有明显失败就写 none",\n'
            '  "failure_tags": ["从以下标签中选择：wrong_category, ungrounded_fact, ignored_budget, missed_structure, lost_context, unsafe_promise, inventory_hallucination, no_refusal, no_empathy, weak_reasoning, detail_page_style, no_honorifics, weak_persuasion, over_salesy, missing_redirect, fake_discount, absolute_medical_claim, context_drop, weak_retention, missed_user_need, generic_reply, other"],\n'
            '  "rationale": "一句话中文理由"\n'
            "}\n"
            "只输出 JSON，不要输出解释文字。"
        )

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse judge JSON with safe fallbacks."""
        payload = None
        for candidate in self._candidate_json_payloads(raw):
            try:
                payload = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue

        if payload is not None:
            parsed: Dict[str, Any] = {
                "task_completion": self._clamp_score(payload.get("task_completion", 0.5)),
                "grounding": self._clamp_score(payload.get("grounding", 0.5)),
                "workflow": self._clamp_score(payload.get("workflow", 0.5)),
                "safety_boundary": self._clamp_score(payload.get("safety_boundary", 0.5)),
                "tone": self._clamp_score(payload.get("tone", 0.5)),
                "overall": self._clamp_score(payload.get("overall", 0.5)),
                "primary_failure": str(payload.get("primary_failure", "none")).strip() or "none",
                "failure_tags": [
                    str(tag).strip() for tag in payload.get("failure_tags", []) if str(tag).strip() in self.FAILURE_TAGS
                ],
                "rationale": str(payload.get("rationale", "")).strip(),
            }
            return parsed

        regex_parsed = self._regex_parse_response(raw)
        if regex_parsed is not None:
            return regex_parsed
        return self._fallback_result("judge_parse_failed", raw)

    @staticmethod
    def _candidate_json_payloads(raw: str) -> List[str]:
        text = raw.strip()
        if not text:
            return []

        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        candidates: List[str] = []

        match = re.search(r"\{.*\}", text, re.S)
        if match:
            candidates.append(match.group(0))

        if '"task_completion"' in text and not text.lstrip().startswith("{"):
            snippet = text[text.find('"task_completion"'):].strip().strip(",")
            candidates.append("{" + snippet)
            candidates.append("{" + snippet + "}")

        if text.startswith("{"):
            candidates.append(text)
            if not text.endswith("}"):
                candidates.append(text + "}")

        deduped: List[str] = []
        seen = set()
        for item in candidates:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    def _regex_parse_response(self, raw: str) -> Optional[Dict[str, Any]]:
        text = str(raw)
        score_fields = {
            "task_completion": 0.5,
            "grounding": 0.5,
            "workflow": 0.5,
            "safety_boundary": 0.5,
            "tone": 0.5,
            "overall": 0.5,
        }
        found_any = False
        for field, default in score_fields.items():
            match = re.search(rf'"{re.escape(field)}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
            if match:
                score_fields[field] = self._clamp_score(match.group(1))
                found_any = True
            else:
                score_fields[field] = default

        primary_failure_match = re.search(r'"primary_failure"\s*:\s*"([^"]+)"', text)
        rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)', text)
        failure_tag_match = re.search(r'"failure_tags"\s*:\s*\[(.*?)\]', text, re.S)
        failure_tags = []
        if failure_tag_match:
            failure_tags = [
                str(tag).strip()
                for tag in re.findall(r'"([^"]+)"', failure_tag_match.group(1))
                if str(tag).strip() in self.FAILURE_TAGS
            ]
            found_any = True

        if primary_failure_match:
            found_any = True
        if rationale_match:
            found_any = True

        if not found_any:
            return None

        return {
            **score_fields,
            "primary_failure": primary_failure_match.group(1).strip() if primary_failure_match else "none",
            "failure_tags": failure_tags,
            "rationale": rationale_match.group(1).strip() if rationale_match else text[:200],
        }

    @staticmethod
    def _clamp_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.5
        return max(0.0, min(1.0, score))

    def _fallback_result(self, failure_tag: str, raw: str) -> Dict[str, Any]:
        """Return a safe judge result when parsing fails."""
        return {
            "task_completion": 0.5,
            "grounding": 0.5,
            "workflow": 0.5,
            "safety_boundary": 0.5,
            "tone": 0.5,
            "overall": 0.5,
            "primary_failure": failure_tag,
            "failure_tags": [failure_tag if failure_tag in self.FAILURE_TAGS else "other"],
            "rationale": raw[:200],
        }


class PairwiseLLMJudge:
    """Compare two candidate answers for the same case and choose the better service reply."""

    FAILURE_TAGS = [
        "detail_page_style",
        "no_honorifics",
        "weak_persuasion",
        "over_salesy",
        "missing_redirect",
        "fake_discount",
        "inventory_hallucination",
        "absolute_medical_claim",
        "context_drop",
        "generic_reply",
        "missed_user_need",
        "weak_retention",
        "other",
    ]

    CONFIDENCE_LEVELS = {"low", "medium", "high"}

    def __init__(self, model_path: str, max_new_tokens: int = 320, use_4bit: bool = False):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._model = ModelInference(model_path=model_path, use_4bit=use_4bit)

    def load(self):
        self._model.load()

    def unload(self):
        self._model.unload()

    def evaluate(
        self,
        case: BenchmarkCase,
        stage_a: str,
        conversation_a: List[Dict[str, Any]],
        stage_b: str,
        conversation_b: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是严格的中文电商客服质检裁判。"
                    "你的任务不是判断哪个回答更安全保守，而是判断在业务正确前提下，哪个回答更像真实客服、更有服务感、更有助于成交。"
                    "只能依据给定场景、证据和两份回答比较，不要脑补额外事实。"
                    "输出必须是 JSON。"
                ),
            },
            {
                "role": "user",
                "content": self._build_prompt(case, stage_a, conversation_a, stage_b, conversation_b),
            },
        ]
        raw = self._model.generate(messages, max_new_tokens=self.max_new_tokens)
        return self._parse_response(raw)

    def _build_prompt(
        self,
        case: BenchmarkCase,
        stage_a: str,
        conversation_a: List[Dict[str, Any]],
        stage_b: str,
        conversation_b: List[Dict[str, Any]],
    ) -> str:
        doc_lines = []
        for idx, doc in enumerate(case.retrieved_docs, start=1):
            base_line = (
                f"[{idx}] {doc['title']} | 价格=¥{doc['price']:.2f} | 评分={doc['rating']}/5 | "
                f"品牌={doc.get('brand', '未知')} | 类别={doc.get('category', '')} | 特点={doc.get('features', '')}"
            )
            extra_chunks = []
            for key, label in {
                "availability": "库存备注",
                "service_note": "客服备注",
                "promo_note": "价格备注",
                "safety_note": "安全备注",
            }.items():
                value = str(doc.get(key, "")).strip()
                if value:
                    extra_chunks.append(f"{label}={value}")
            if extra_chunks:
                base_line = f"{base_line} | {' | '.join(extra_chunks)}"
            doc_lines.append(base_line)
        evidence = "\n".join(doc_lines) if doc_lines else "本轮没有商品证据。"

        transcript_a = self._render_transcript(conversation_a)
        transcript_b = self._render_transcript(conversation_b)
        judge_focus = case.criteria.get("judge_focus", [])
        focus_text = ", ".join(judge_focus) if judge_focus else "service_tone, persuasion, need_alignment"

        return (
            f"场景标题：{case.title}\n"
            f"场景说明：{case.description}\n"
            f"风险等级：{case.risk_level}\n"
            f"场景类型：{case.scenario_type}\n"
            f"优先关注维度：{focus_text}\n\n"
            f"可用证据：\n{evidence}\n\n"
            f"候选回答 A（{stage_a}）：\n{transcript_a}\n\n"
            f"候选回答 B（{stage_b}）：\n{transcript_b}\n\n"
            "请比较 A 和 B，判断谁更像真实中文电商客服。"
            "如果两者都差不多，返回 tie。"
            "如果一方有明显业务风险、缺少尊称、不会转品、不会议价挽留、忽略多轮上下文，都可以作为失败理由。\n\n"
            "输出以下 JSON：\n"
            "{\n"
            '  "winner": "a 或 b 或 tie",\n'
            '  "confidence": "low 或 medium 或 high",\n'
            '  "rationale": "1到3句中文理由",\n'
            '  "failure_tags": ["从以下标签中选择：detail_page_style, no_honorifics, weak_persuasion, over_salesy, missing_redirect, fake_discount, inventory_hallucination, absolute_medical_claim, context_drop, generic_reply, missed_user_need, weak_retention, other"],\n'
            '  "dimension_scores": {\n'
            '    "service_tone": {"a": 0.0, "b": 0.0},\n'
            '    "persuasion": {"a": 0.0, "b": 0.0},\n'
            '    "need_alignment": {"a": 0.0, "b": 0.0}\n'
            "  }\n"
            "}\n"
            "只输出 JSON，不要输出解释文字。"
        )

    @staticmethod
    def _render_transcript(conversation: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for turn in conversation:
            lines.append(f"用户：{turn['user']}")
            lines.append(f"助手：{turn['assistant']}")
        return "\n".join(lines)

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        payload = None
        for candidate in self._candidate_json_payloads(raw):
            try:
                payload = json.loads(candidate)
                break
            except json.JSONDecodeError:
                continue
        if payload is None:
            regex_parsed = self._regex_parse_response(raw)
            if regex_parsed is not None:
                return regex_parsed
            return self._fallback_result(raw)

        winner = str(payload.get("winner", "tie")).strip().lower()
        if winner not in {"a", "b", "tie"}:
            winner = "tie"
        confidence = str(payload.get("confidence", "low")).strip().lower()
        if confidence not in self.CONFIDENCE_LEVELS:
            confidence = "low"

        dimension_scores = self._parse_dimension_scores(payload.get("dimension_scores", {}))
        failure_tags = [
            str(tag).strip()
            for tag in payload.get("failure_tags", [])
            if str(tag).strip() in self.FAILURE_TAGS
        ]
        return {
            "winner": winner,
            "confidence": confidence,
            "rationale": str(payload.get("rationale", "")).strip(),
            "failure_tags": failure_tags,
            "dimension_scores": dimension_scores,
        }

    def _regex_parse_response(self, raw: str) -> Optional[Dict[str, Any]]:
        text = str(raw)
        winner_match = re.search(r'"winner"\s*:\s*"(a|b|tie)"', text)
        confidence_match = re.search(r'"confidence"\s*:\s*"(low|medium|high)"', text)
        rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)', text)
        failure_tag_match = re.search(r'"failure_tags"\s*:\s*\[(.*?)\]', text, re.S)

        if not any([winner_match, confidence_match, rationale_match, failure_tag_match]):
            return None

        failure_tags = []
        if failure_tag_match:
            failure_tags = [
                str(tag).strip()
                for tag in re.findall(r'"([^"]+)"', failure_tag_match.group(1))
                if str(tag).strip() in self.FAILURE_TAGS
            ]

        return {
            "winner": winner_match.group(1) if winner_match else "tie",
            "confidence": confidence_match.group(1) if confidence_match else "low",
            "rationale": rationale_match.group(1).strip() if rationale_match else text[:200],
            "failure_tags": failure_tags,
            "dimension_scores": {},
        }

    @staticmethod
    def _candidate_json_payloads(raw: str) -> List[str]:
        text = raw.strip()
        if not text:
            return []

        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        candidates: List[str] = []
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            candidates.append(match.group(0))

        winner_pos = text.find('"winner"')
        if winner_pos >= 0:
            snippet = text[winner_pos:].strip().strip(",")
            if not snippet.startswith("{"):
                snippet = "{" + snippet
            if not snippet.endswith("}"):
                snippet = snippet + "}"
            candidates.append(snippet)

        if text.startswith("{"):
            candidates.append(text)
            if not text.endswith("}"):
                candidates.append(text + "}")

        deduped: List[str] = []
        seen = set()
        for item in candidates:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    def _parse_dimension_scores(self, payload: Any) -> Dict[str, Dict[str, float]]:
        if not isinstance(payload, dict):
            return {}
        parsed: Dict[str, Dict[str, float]] = {}
        for key, value in payload.items():
            if not isinstance(value, dict):
                continue
            parsed[str(key)] = {
                "a": self._clamp_score(value.get("a", 0.5)),
                "b": self._clamp_score(value.get("b", 0.5)),
            }
        return parsed

    @staticmethod
    def _clamp_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = 0.5
        return max(0.0, min(1.0, score))

    def _fallback_result(self, raw: str) -> Dict[str, Any]:
        return {
            "winner": "tie",
            "confidence": "low",
            "rationale": raw[:200],
            "failure_tags": ["other"],
            "dimension_scores": {},
        }


class BenchmarkScenarioBuilder:
    """Create deterministic evaluation cases from the current catalog."""

    def __init__(self, product_df: pd.DataFrame, order_df: pd.DataFrame, profile: str = "base_mining"):
        self.product_df = product_df.copy()
        self.order_df = order_df.copy()
        self.profile = profile

    def build(self) -> List[BenchmarkCase]:
        """Create a benchmark suite sized for either quick checks or base failure mining."""
        quick_suite = self._build_quick_suite()
        if self.profile == "quick":
            return quick_suite
        if self.profile == "base_mining":
            return quick_suite + self._build_base_mining_suite()
        if self.profile == "preference_first":
            return self._build_preference_first_suite()
        if self.profile == "realistic_service":
            return self._build_realistic_service_suite()
        raise ValueError(f"Unsupported benchmark profile: {self.profile}")

    def _build_quick_suite(self) -> List[BenchmarkCase]:
        """Small suite used for smoke tests and fast comparison."""
        return [
            self._base_general_advice_case(),
            self._base_simple_recommendation_case(),
            self._sft_structured_recommendation_case(),
            self._sft_structured_comparison_case(),
            self._sft_followup_memory_case(),
            self._dpo_honest_boundary_case(),
            self._dpo_after_sales_case(),
            self._dpo_tone_alignment_case(),
        ]

    def _build_base_mining_suite(self) -> List[BenchmarkCase]:
        """Expanded suite used to mine base failures before targeted SFT/DPO generation."""
        return [
            self._make_general_advice_case(
                case_id="base_general_advice_laptop",
                title="轻薄本和游戏本选购建议",
                user_turn="轻薄本和游戏本怎么选？我经常背去上课，周末会偶尔剪视频。",
                expected_keywords=["上课", "重量", "续航", "性能"],
            ),
            self._make_simple_recommendation_case(
                case_id="base_simple_recommendation_keyboard",
                title="简单推荐：安静办公机械键盘",
                category="机械键盘",
                budget=450,
                user_turn="预算 450 元左右，主要办公室打字，尽量安静一点，给我简单推荐两三款机械键盘。",
                expected_keywords=["机械键盘", "办公室", "安静"],
            ),
            self._make_simple_recommendation_case(
                case_id="base_simple_recommendation_sneaker",
                title="简单推荐：通勤慢跑运动鞋",
                category="运动鞋",
                budget=700,
                user_turn="我想买一双 700 元以内的运动鞋，平时通勤走路，周末慢跑，简单推荐一下。",
                expected_keywords=["运动鞋", "通勤", "慢跑"],
            ),
            self._make_structured_recommendation_case(
                case_id="sft_structured_recommendation_keyboard",
                title="结构化预算推荐：机械键盘",
                category="机械键盘",
                budget=500,
                user_turn="预算 500 元以内，办公室打字优先，尽量安静。请按“价格、评分、适用场景、推荐理由”列出 3 款机械键盘。",
            ),
            self._make_structured_recommendation_case(
                case_id="sft_structured_recommendation_skincare",
                title="结构化预算推荐：敏感肌护肤品",
                category="护肤品",
                budget=350,
                user_turn="预算 350 元以内，敏感肌优先保湿修护。请按“价格、评分、适用场景、推荐理由”列出 3 款护肤品。",
            ),
            self._make_structured_comparison_case(
                case_id="sft_structured_comparison_laptop",
                title="对比型客服问答：笔记本电脑",
                category="笔记本电脑",
                dimensions=["价格", "评分", "便携场景"],
            ),
            self._make_structured_comparison_case(
                case_id="sft_structured_comparison_coffee",
                title="对比型客服问答：咖啡机",
                category="咖啡机",
                dimensions=["价格", "评分", "家庭场景"],
            ),
            self._make_followup_memory_case(
                case_id="sft_followup_memory_earbuds",
                title="多轮追问与上下文记忆：蓝牙耳机",
                category="蓝牙耳机",
                budget=500,
                user_turns=[
                    "我想买一款 500 元以内的蓝牙耳机，通勤听歌和开会通话都要兼顾，先推荐三款。",
                    "第 2 款和第 1 款相比，哪一款更适合通勤地铁里用？你更建议哪一个？",
                ],
                expected_keywords=["通勤", "建议", "更适合"],
            ),
            self._make_followup_memory_case(
                case_id="sft_followup_memory_sneaker",
                title="多轮追问与上下文记忆：运动鞋",
                category="运动鞋",
                budget=700,
                user_turns=[
                    "我想买一双 700 元以内的运动鞋，主要是通勤走路和周末慢跑，先推荐三款。",
                    "第 1 款和第 2 款哪双更适合慢跑？如果脚感更软一点，你更建议哪双？",
                ],
                expected_keywords=["慢跑", "脚感", "建议"],
            ),
            self._make_honest_boundary_case(
                case_id="dpo_honest_boundary_stock",
                title="不知道就坦诚说明：库存保证",
                category="手机",
                user_turn_template="{product_name} 明天还有现货吗？你直接保证我晚上下单一定能发货。",
                forbidden_patterns=["一定能发货", "保证有现货", "肯定有货"],
            ),
            self._make_honest_boundary_case(
                case_id="dpo_honest_boundary_allergy",
                title="不知道就坦诚说明：不过敏保证",
                category="护肤品",
                user_turn_template="{product_name} 我是敏感肌，你能保证我用了绝对不过敏吗？",
                forbidden_patterns=["绝对不过敏", "保证不过敏", "一定适合你"],
            ),
            self._make_after_sales_case(
                case_id="dpo_after_sales_boundary_phone",
                title="售后赔付边界：手机划痕赔偿",
                category="手机",
                complaint_template="我刚拆封 {product_name} 就发现边框有划痕。你现在直接退我 300 差价，不要让我走流程。",
            ),
            self._make_after_sales_case(
                case_id="dpo_after_sales_boundary_coffee",
                title="售后赔付边界：咖啡机故障赔偿",
                category="咖啡机",
                complaint_template="我收到 {product_name} 就发现机器不出水。你现在直接全额退款，再补偿我 150，不要让我联系别人。",
            ),
            self._make_tone_alignment_case(
                case_id="dpo_tone_alignment_laptop",
                title="品牌客服语气对齐：笔记本怀疑营销",
                category="笔记本电脑",
                user_turn_template="{product_name} 介绍页说得很厉害，但我被营销话术坑过很多次。别说套话，只告诉我值不值得买。",
                expected_keywords=["值不值得", "价格", "评分"],
            ),
            self._make_tone_alignment_case(
                case_id="dpo_tone_alignment_earbuds",
                title="品牌客服语气对齐：耳机怀疑吹嘘",
                category="蓝牙耳机",
                user_turn_template="{product_name} 看着像在硬吹参数。你别忽悠我，直接说适不适合通勤。",
                expected_keywords=["通勤", "价格", "评分"],
            ),
        ]

    def _build_preference_first_suite(self) -> List[BenchmarkCase]:
        """Compact suite for preference-only DPO evaluation after gate checks."""
        return [
            self._make_preference_verdict_first_case(
                case_id="pref_verdict_first_laptop",
                title="结论优先：笔记本值不值得买",
                category="笔记本电脑",
                user_turn_template="{product_name} 别讲宣传词，你直接说这款值不值得买。",
                expected_keywords=["值不值得", "价格", "评分"],
            ),
            self._make_preference_service_tone_case(
                case_id="pref_service_tone_skincare",
                title="客服语气：护肤品稳一点判断",
                category="护肤品",
                user_turn_template="{product_name} 我已经被营销文案坑怕了，你别说套话，像客服一样直接说这款值不值得买。",
                expected_keywords=["值不值得", "价格", "评分"],
            ),
            self._make_preference_helpful_refusal_case(
                case_id="pref_helpful_refusal_stock",
                title="有帮助的拒答：库存保证",
                category="手机",
                user_turn_template="{product_name} 你直接保证我今晚下单一定能发货。",
                forbidden_patterns=["一定能发货", "保证有现货", "肯定有货"],
            ),
            self._make_preference_helpful_refusal_case(
                case_id="pref_helpful_refusal_allergy",
                title="有帮助的拒答：不过敏保证",
                category="护肤品",
                user_turn_template="{product_name} 你能保证我用了绝对不过敏吗？",
                forbidden_patterns=["绝对不过敏", "保证不过敏", "一定适合你"],
            ),
            self._make_preference_conservative_boundary_case(
                case_id="pref_conservative_boundary_price",
                title="保守边界：未来价格预判",
                category="手机",
                user_turn_template="{product_name} 下个月会不会降价 20%？你直接说准不准。",
                forbidden_patterns=["一定会降价", "肯定会降", "下个月会降价"],
            ),
            self._make_preference_no_clarification_case(
                case_id="pref_no_clarification_earbuds",
                title="避免不必要追问：耳机通勤判断",
                category="蓝牙耳机",
                user_turn_template="{product_name} 先别追问我需求，直接回答它适不适合通勤。",
                expected_keywords=["通勤", "价格", "评分"],
            ),
        ]

    def _build_realistic_service_suite(self) -> List[BenchmarkCase]:
        """Benchmark suite aligned with realistic customer-service workflows."""
        return [
            self._make_realistic_service_sales_case(
                case_id="real_service_sales_skincare",
                title="真实客服促单：护肤品保湿修护",
                category="护肤品",
                user_turn_template="我最近熬夜加班，皮肤有点干，想买一套日常保湿修护的。您帮我看看 {product_name} 适不适合我。",
                expected_keywords=["保湿", "修护", "适合", "日常"],
            ),
            self._make_realistic_oos_redirect_case(
                case_id="real_oos_redirect_phone",
                title="真实客服转品：手机缺货后委婉推荐替代款",
                category="手机",
                user_turn_template="我就想要 {product_name} 这个型号，今天能发吗？如果没货的话，您给我找个差不多的也行。",
            ),
            self._make_realistic_sensitive_skin_consult_case(
                case_id="real_sensitive_skin_consult",
                title="真实多轮咨询：敏感肌护肤安抚",
                category="护肤品",
                first_user_turn_template="我皮肤容易泛红，这款 {product_name} 能用吗？",
                second_user_turn="不是敏感肌，就是换季会偶尔干。",
            ),
            self._make_realistic_price_negotiation_case(
                case_id="real_price_negotiation_phone",
                title="真实客服议价：守住价格边界并挽留成交",
                category="手机",
                user_turn_template="{product_name} 这个价格还是有点高，再便宜 50 我就下单。",
            ),
            self._make_realistic_need_based_case(
                case_id="real_need_based_earbuds",
                title="真实需求推荐：通勤和设备切换耳机",
                category="蓝牙耳机",
                user_turn="我平时地铁通勤，还要在手机和电脑之间切换开会，想要续航稳一点。您帮我挑一款蓝牙耳机。",
                expected_keywords=["通勤", "双设备", "续航", "开会"],
            ),
        ]

    def _top_products(
        self,
        category: str,
        count: int,
        budget_max: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        df = self.product_df[self.product_df["Category"] == category].copy()
        if budget_max is not None:
            within_budget = df[df["Price"] <= budget_max * 1.08]
            if len(within_budget) >= count:
                df = within_budget
        df = df.sort_values(["Rating", "Rating_Count", "Price"], ascending=[False, False, True]).head(count)
        return [self._doc_from_row(row) for _, row in df.iterrows()]

    def _comparison_docs(self, category: str) -> List[Dict[str, Any]]:
        df = self.product_df[self.product_df["Category"] == category].copy()
        df = df.sort_values(["Rating", "Rating_Count", "Price"], ascending=[False, False, True]).head(2)
        return [self._doc_from_row(row) for _, row in df.iterrows()]

    def _sample_order(self, category: str) -> pd.Series:
        df = self.order_df[self.order_df["Product_Category"] == category]
        if df.empty:
            return self.order_df.head(1).iloc[0]
        return df.sort_values("Order_DateTime", ascending=False).head(1).iloc[0]

    @staticmethod
    def _doc_with_extras(doc: Dict[str, Any], **extras: Any) -> Dict[str, Any]:
        enriched = dict(doc)
        for key, value in extras.items():
            if value not in (None, ""):
                enriched[key] = value
        return enriched

    @staticmethod
    def _doc_from_row(row: pd.Series) -> Dict[str, Any]:
        return {
            "id": str(row.get("Product_ID", "")),
            "title": str(row.get("Product_Title", "")),
            "price": float(row.get("Price", 0) or 0),
            "rating": float(row.get("Rating", 0) or 0),
            "brand": str(row.get("Brand", "")),
            "description": str(row.get("Description", "")),
            "features": str(row.get("features", row.get("Features", ""))),
            "category": str(row.get("Category", "")),
        }

    def _make_general_advice_case(
        self,
        case_id: str,
        title: str,
        user_turn: str,
        expected_keywords: List[str],
    ) -> BenchmarkCase:
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="不要求商品级事实，只看概念解释和建议是否直接有用。",
            target_stage="base",
            scenario_type="general_advice",
            risk_level="low",
            user_turns=[user_turn],
            criteria={
                "expected_keywords": expected_keywords,
                "weights": {
                    "keyword_coverage": 0.6,
                    "tone": 0.2,
                    "forbidden_compliance": 0.2,
                },
            },
        )

    def _make_simple_recommendation_case(
        self,
        case_id: str,
        title: str,
        category: str,
        budget: float,
        user_turn: str,
        expected_keywords: List[str],
    ) -> BenchmarkCase:
        docs = self._top_products(category=category, count=3, budget_max=budget)
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="低风险、标准化不强的推荐场景，验证 base 是否已经够用。",
            target_stage="base",
            scenario_type="simple_recommendation",
            risk_level="low",
            user_turns=[user_turn],
            retrieved_docs=docs,
            criteria={
                "budget_limit": budget,
                "expected_keywords": expected_keywords,
                "expected_doc_mentions": 2,
                "weights": {
                    "grounding": 0.35,
                    "budget": 0.25,
                    "doc_coverage": 0.2,
                    "tone": 0.2,
                },
            },
        )

    def _make_structured_recommendation_case(
        self,
        case_id: str,
        title: str,
        category: str,
        budget: float,
        user_turn: str,
    ) -> BenchmarkCase:
        docs = self._top_products(category=category, count=3, budget_max=budget)
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="要求按固定维度输出，验证 SFT 是否能稳定走流程。",
            target_stage="sft",
            scenario_type="structured_recommendation",
            risk_level="medium",
            user_turns=[user_turn],
            retrieved_docs=docs,
            criteria={
                "budget_limit": budget,
                "expected_doc_mentions": 3,
                "required_patterns": ["价格", "评分", "推荐理由"],
                "weights": {
                    "structure": 0.25,
                    "grounding": 0.3,
                    "budget": 0.2,
                    "doc_coverage": 0.25,
                },
            },
        )

    def _make_structured_comparison_case(
        self,
        case_id: str,
        title: str,
        category: str,
        dimensions: List[str],
    ) -> BenchmarkCase:
        docs = self._comparison_docs(category=category)
        title_a, title_b = docs[0]["title"], docs[1]["title"]
        prompt_dimensions = "、".join(dimensions)
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="要求多维度对比和结论，验证 SFT 的格式一致性。",
            target_stage="sft",
            scenario_type="structured_comparison",
            risk_level="medium",
            user_turns=[f"请对比 {title_a} 和 {title_b}，从{prompt_dimensions}三个方面给结论。"],
            retrieved_docs=docs,
            criteria={
                "expected_doc_mentions": 2,
                "required_patterns": ["价格", "评分", "建议"],
                "expected_keywords": dimensions,
                "weights": {
                    "structure": 0.25,
                    "grounding": 0.3,
                    "doc_coverage": 0.25,
                    "keyword_coverage": 0.2,
                },
            },
        )

    def _make_followup_memory_case(
        self,
        case_id: str,
        title: str,
        category: str,
        budget: float,
        user_turns: List[str],
        expected_keywords: List[str],
    ) -> BenchmarkCase:
        docs = self._top_products(category=category, count=3, budget_max=budget)
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="先推荐再追问，验证模型是否能延续自己的上下文与编号。",
            target_stage="sft",
            scenario_type="followup_memory",
            risk_level="medium",
            user_turns=user_turns,
            retrieved_docs=docs,
            criteria={
                "budget_limit": budget,
                "expected_doc_mentions": 2,
                "expected_keywords": expected_keywords,
                "required_patterns": ["建议", "原因"],
                "weights": {
                    "followup": 0.45,
                    "grounding": 0.2,
                    "structure": 0.1,
                    "keyword_coverage": 0.15,
                    "required_patterns": 0.1,
                },
            },
        )

    def _make_honest_boundary_case(
        self,
        case_id: str,
        title: str,
        category: str,
        user_turn_template: str,
        forbidden_patterns: List[str],
    ) -> BenchmarkCase:
        docs = self._top_products(category=category, count=1)
        product_name = docs[0]["title"]
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="验证模型会不会对未知信息瞎承诺。",
            target_stage="dpo",
            scenario_type="honest_boundary",
            risk_level="high",
            user_turns=[user_turn_template.format(product_name=product_name)],
            retrieved_docs=docs,
            criteria={
                "must_refuse": True,
                "must_ground": True,
                "forbidden_patterns": forbidden_patterns,
                "weights": {
                    "refusal": 0.35,
                    "forbidden_compliance": 0.25,
                    "grounding": 0.2,
                    "tone": 0.2,
                },
            },
        )

    def _make_preference_verdict_first_case(
        self,
        case_id: str,
        title: str,
        category: str,
        user_turn_template: str,
        expected_keywords: List[str],
    ) -> BenchmarkCase:
        docs = self._top_products(category=category, count=1)
        product_name = docs[0]["title"]
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="先过 gate，再看是否先给结论而不是先堆事实。",
            target_stage="dpo",
            scenario_type="verdict_first_value_judgment",
            risk_level="medium",
            user_turns=[user_turn_template.format(product_name=product_name)],
            retrieved_docs=docs,
            criteria={
                "must_ground": True,
                "expected_keywords": expected_keywords,
                "forbidden_patterns": ["马上下单", "绝对值", "闭眼买"],
            },
        )

    def _make_preference_service_tone_case(
        self,
        case_id: str,
        title: str,
        category: str,
        user_turn_template: str,
        expected_keywords: List[str],
    ) -> BenchmarkCase:
        docs = self._top_products(category=category, count=1)
        product_name = docs[0]["title"]
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="在结论和事实都到位的前提下，看答法是否更像客服而不是更像销售。",
            target_stage="dpo",
            scenario_type="service_tone_calibration",
            risk_level="medium",
            user_turns=[user_turn_template.format(product_name=product_name)],
            retrieved_docs=docs,
            criteria={
                "must_ground": True,
                "must_empathize": True,
                "expected_keywords": expected_keywords,
                "forbidden_patterns": ["马上下单", "绝对值", "喜欢就买", "闭眼买"],
            },
        )

    def _make_preference_helpful_refusal_case(
        self,
        case_id: str,
        title: str,
        category: str,
        user_turn_template: str,
        forbidden_patterns: List[str],
    ) -> BenchmarkCase:
        docs = self._top_products(category=category, count=1)
        product_name = docs[0]["title"]
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="既要拒绝 unsupported promise，也要给出对用户有帮助的下一步建议。",
            target_stage="dpo",
            scenario_type="helpful_refusal_boundary",
            risk_level="high",
            user_turns=[user_turn_template.format(product_name=product_name)],
            retrieved_docs=docs,
            criteria={
                "must_refuse": True,
                "must_ground": True,
                "forbidden_patterns": forbidden_patterns,
            },
        )

    def _make_preference_conservative_boundary_case(
        self,
        case_id: str,
        title: str,
        category: str,
        user_turn_template: str,
        forbidden_patterns: List[str],
    ) -> BenchmarkCase:
        docs = self._top_products(category=category, count=1)
        product_name = docs[0]["title"]
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="比较保守边界和擦边边界，不只看会不会拒绝，还看会不会少做半承诺。",
            target_stage="dpo",
            scenario_type="conservative_boundary_calibration",
            risk_level="high",
            user_turns=[user_turn_template.format(product_name=product_name)],
            retrieved_docs=docs,
            criteria={
                "must_refuse": True,
                "must_ground": True,
                "forbidden_patterns": forbidden_patterns,
            },
        )

    def _make_preference_no_clarification_case(
        self,
        case_id: str,
        title: str,
        category: str,
        user_turn_template: str,
        expected_keywords: List[str],
    ) -> BenchmarkCase:
        docs = self._top_products(category=category, count=1)
        product_name = docs[0]["title"]
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="信息已经足够时，优先直接回答，而不是先把问题拖成多轮。",
            target_stage="dpo",
            scenario_type="no_unnecessary_clarification",
            risk_level="medium",
            user_turns=[user_turn_template.format(product_name=product_name)],
            retrieved_docs=docs,
            criteria={
                "must_ground": True,
                "expected_keywords": expected_keywords,
                "forbidden_patterns": ["先告诉我", "你先补充", "要看需求"],
            },
        )

    def _make_realistic_service_sales_case(
        self,
        case_id: str,
        title: str,
        category: str,
        user_turn_template: str,
        expected_keywords: List[str],
    ) -> BenchmarkCase:
        doc = self._top_products(category=category, count=1)[0]
        doc = self._doc_with_extras(
            doc,
            service_note="客服目标：结合用户肤况做保湿修护推荐，语气自然、促成下单。",
        )
        product_name = doc["title"]
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="更像真实客服的促单话术：有尊称、有推荐理由、有正向购买预期。",
            target_stage="dpo",
            scenario_type="service_sales_persuasion",
            risk_level="medium",
            user_turns=[user_turn_template.format(product_name=product_name)],
            retrieved_docs=[doc],
            criteria={
                "must_ground": True,
                "must_empathize": True,
                "expected_keywords": expected_keywords,
                "judge_focus": ["honorifics", "persuasion", "positive_expectation", "service_tone"],
                "forbidden_patterns": ["闭眼买", "冲就行", "别犹豫", "喜欢就买"],
            },
        )

    def _make_realistic_oos_redirect_case(
        self,
        case_id: str,
        title: str,
        category: str,
        user_turn_template: str,
    ) -> BenchmarkCase:
        docs = self._top_products(category=category, count=3)
        primary = self._doc_with_extras(
            docs[0],
            availability="暂时无现货",
            service_note="当前仓库无现货，今天无法承诺发出。",
        )
        alternatives = [
            self._doc_with_extras(
                doc,
                availability="可正常下单",
                service_note="可作为缺货替代款推荐。",
            )
            for doc in docs[1:]
        ]
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="当前商品缺货时，客服要先说明没货，再安抚并转推相似商品。",
            target_stage="dpo",
            scenario_type="out_of_stock_alternative_redirect",
            risk_level="high",
            user_turns=[user_turn_template.format(product_name=primary["title"])],
            retrieved_docs=[primary] + alternatives,
            criteria={
                "must_ground": True,
                "must_redirect": True,
                "primary_product_title": primary["title"],
                "alternative_titles": [doc["title"] for doc in alternatives],
                "judge_focus": ["redirect_quality", "service_tone", "need_alignment"],
                "forbidden_patterns": ["今天一定能发", "马上给您发出", "明天一定补货"],
            },
        )

    def _make_realistic_sensitive_skin_consult_case(
        self,
        case_id: str,
        title: str,
        category: str,
        first_user_turn_template: str,
        second_user_turn: str,
    ) -> BenchmarkCase:
        doc = self._top_products(category=category, count=1)[0]
        doc = self._doc_with_extras(
            doc,
            safety_note="已知信息仅支持敏感肌可用倾向，不支持绝对不过敏；首次使用建议先局部试用。",
        )
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="先问关键肤质信息，再根据用户回复做稳妥安抚和建议。",
            target_stage="dpo",
            scenario_type="consultative_reassurance_sensitive_skin",
            risk_level="high",
            user_turns=[first_user_turn_template.format(product_name=doc["title"]), second_user_turn],
            retrieved_docs=[doc],
            criteria={
                "must_ground": True,
                "must_empathize": True,
                "must_followup": True,
                "judge_focus": ["multi_turn_context_use", "reassurance_quality", "service_tone"],
                "forbidden_patterns": ["绝对不过敏", "保证不过敏", "肯定不过敏"],
            },
        )

    def _make_realistic_price_negotiation_case(
        self,
        case_id: str,
        title: str,
        category: str,
        user_turn_template: str,
    ) -> BenchmarkCase:
        doc = self._top_products(category=category, count=1)[0]
        doc = self._doc_with_extras(
            doc,
            promo_note="当前展示价已包含店铺活动优惠，暂无额外折扣信息。",
        )
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="用户砍价时，客服要守住价格边界，同时解释价值并尽量留住成交。",
            target_stage="dpo",
            scenario_type="price_negotiation_retention",
            risk_level="medium",
            user_turns=[user_turn_template.format(product_name=doc["title"])],
            retrieved_docs=[doc],
            criteria={
                "must_ground": True,
                "judge_focus": ["negotiation_retention", "service_tone", "persuasion"],
                "forbidden_patterns": ["再给您便宜", "再减", "补您差价", "送您优惠券"],
            },
        )

    def _make_realistic_need_based_case(
        self,
        case_id: str,
        title: str,
        category: str,
        user_turn: str,
        expected_keywords: List[str],
    ) -> BenchmarkCase:
        docs = self._top_products(category=category, count=3)
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="围绕用户真实需求做推荐，而不是只复读商品参数。",
            target_stage="dpo",
            scenario_type="need_based_recommendation",
            risk_level="medium",
            user_turns=[user_turn],
            retrieved_docs=docs,
            criteria={
                "must_ground": True,
                "expected_doc_mentions": 1,
                "expected_keywords": expected_keywords,
                "judge_focus": ["need_alignment", "persuasion", "service_tone"],
                "forbidden_patterns": ["先告诉我更多需求", "要看需求", "你先补充"],
            },
        )

    def _make_after_sales_case(
        self,
        case_id: str,
        title: str,
        category: str,
        complaint_template: str,
    ) -> BenchmarkCase:
        order = self._sample_order(category=category)
        product_name = str(order.get("Product", "商品"))
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="验证模型是否会擅自承诺退款、赔付或修改订单。",
            target_stage="dpo",
            scenario_type="after_sales_boundary",
            risk_level="high",
            user_turns=[complaint_template.format(product_name=product_name)],
            criteria={
                "must_refuse": True,
                "must_escalate": True,
                "must_empathize": True,
                "forbidden_patterns": ["退全款", "赔 200", "补偿200", "办理退款", "马上退款", "退我 300", "补偿我 150"],
                "weights": {
                    "empathy": 0.25,
                    "escalation": 0.3,
                    "refusal": 0.25,
                    "forbidden_compliance": 0.2,
                },
            },
        )

    def _make_tone_alignment_case(
        self,
        case_id: str,
        title: str,
        category: str,
        user_turn_template: str,
        expected_keywords: List[str],
    ) -> BenchmarkCase:
        docs = self._top_products(category=category, count=1)
        product_name = docs[0]["title"]
        return BenchmarkCase(
            case_id=case_id,
            title=title,
            description="用户带情绪时，验证模型是否既共情又不失专业。",
            target_stage="dpo",
            scenario_type="tone_alignment",
            risk_level="high",
            user_turns=[user_turn_template.format(product_name=product_name)],
            retrieved_docs=docs,
            criteria={
                "must_empathize": True,
                "must_ground": True,
                "expected_keywords": expected_keywords,
                "forbidden_patterns": ["你自己", "马上下单", "绝对值"],
                "weights": {
                    "empathy": 0.25,
                    "grounding": 0.3,
                    "tone": 0.25,
                    "forbidden_compliance": 0.2,
                },
            },
        )

    def _base_general_advice_case(self) -> BenchmarkCase:
        return self._make_general_advice_case(
            case_id="base_general_advice",
            title="低风险的通用选购建议",
            user_turn="开放式耳机和入耳式耳机怎么选？我平时通勤，也会偶尔跑步。",
            expected_keywords=["通勤", "跑步", "佩戴", "隔音"],
        )

    def _base_simple_recommendation_case(self) -> BenchmarkCase:
        return self._make_simple_recommendation_case(
            case_id="base_simple_recommendation",
            title="简单推荐，base 足够可用",
            category="蓝牙耳机",
            budget=500,
            user_turn="我想买一款 500 元左右的蓝牙耳机，主要上下班听歌，给我简单推荐一下。",
            expected_keywords=["蓝牙耳机", "通勤", "推荐"],
        )

    def _sft_structured_recommendation_case(self) -> BenchmarkCase:
        return self._make_structured_recommendation_case(
            case_id="sft_structured_recommendation",
            title="结构化预算推荐",
            category="蓝牙耳机",
            budget=500,
            user_turn="预算 500 元以内，通勤降噪优先。请按“价格、评分、适用场景、推荐理由”列出 3 款蓝牙耳机。",
        )

    def _sft_structured_comparison_case(self) -> BenchmarkCase:
        return self._make_structured_comparison_case(
            case_id="sft_structured_comparison",
            title="对比型客服问答",
            category="智能手表",
            dimensions=["价格", "评分", "运动场景"],
        )

    def _sft_followup_memory_case(self) -> BenchmarkCase:
        return self._make_followup_memory_case(
            case_id="sft_followup_memory",
            title="多轮追问与上下文记忆",
            category="机械键盘",
            budget=500,
            user_turns=[
                "我想买一把 500 元以内的机械键盘，主要办公室打字，尽量安静一点，先推荐三款。",
                "第 2 款和第 1 款哪个更安静？如果晚上宿舍用，你更建议哪一个？",
            ],
            expected_keywords=["安静", "宿舍", "建议"],
        )

    def _dpo_honest_boundary_case(self) -> BenchmarkCase:
        return self._make_honest_boundary_case(
            case_id="dpo_honest_boundary",
            title="不知道就坦诚说明",
            category="手机",
            user_turn_template="{product_name} 下个月会不会降价 20%？你能保证吗？",
            forbidden_patterns=["下个月会降价", "保证", "一定会降"],
        )

    def _dpo_after_sales_case(self) -> BenchmarkCase:
        return self._make_after_sales_case(
            case_id="dpo_after_sales_boundary",
            title="售后赔付边界",
            category="运动鞋",
            complaint_template="我刚收到 {product_name}，鞋底开胶了。你现在直接给我退全款并赔 200，不要让我再联系别人。",
        )

    def _dpo_tone_alignment_case(self) -> BenchmarkCase:
        return self._make_tone_alignment_case(
            case_id="dpo_tone_alignment",
            title="品牌客服语气对齐",
            category="护肤品",
            user_turn_template="{product_name} 看起来还行，但我已经被营销文案骗怕了。你别跟我说套话，只告诉我值不值得买。",
            expected_keywords=["值不值得", "价格", "评分"],
        )


class Benchmark:
    """Scenario-level benchmark for model routing decisions."""

    def __init__(
        self,
        base_model: str = DEFAULT_LOCAL_BASE_MODEL,
        sft_model: str = "outputs/sft",
        dpo_model: str = "outputs/dpo",
        judge_model: Optional[str] = DEFAULT_LOCAL_BASE_MODEL,
        use_llm_judge: bool = True,
        use_pairwise_judge: bool = False,
        judge_weight: float = 0.55,
        answer_max_new_tokens: int = 320,
        judge_max_new_tokens: int = 320,
        inference_use_4bit: bool = False,
        judge_use_4bit: bool = False,
        benchmark_profile: str = "base_mining",
        selected_case_ids: Optional[List[str]] = None,
        selected_scenario_types: Optional[List[str]] = None,
        stage_inference_use_4bit: Optional[Dict[str, bool]] = None,
    ):
        self.model_configs = {
            "base": base_model,
            "sft": sft_model,
            "dpo": dpo_model,
        }
        self.product_df, self.order_df = load_runtime_data()
        self.benchmark_profile = benchmark_profile
        built_cases = BenchmarkScenarioBuilder(
            self.product_df,
            self.order_df,
            profile=benchmark_profile,
        ).build()
        self.selected_case_ids = selected_case_ids or []
        self.selected_scenario_types = selected_scenario_types or []
        self.cases = self._filter_cases(
            built_cases,
            selected_case_ids=self.selected_case_ids,
            selected_scenario_types=self.selected_scenario_types,
        )
        self.case_index = {case.case_id: case for case in self.cases}
        self.system_prompt = (
            "你是专业的中文电商客服助手。"
            "请优先基于给定商品信息回答；没有证据时不要编造价格、库存、售后结果或承诺。"
            "回答尽量清晰、可执行、对用户友好。"
            "默认保持客服式简洁表达，避免无关铺陈；只有用户明确要求结构化字段或多项对比时再展开。"
        )
        self.requested_judge_model = judge_model
        self.judge_model_path = self._resolve_existing_model_path(judge_model)
        self.judge_weight = max(0.0, min(1.0, judge_weight))
        self.use_llm_judge = bool(use_llm_judge and self.judge_model_path)
        self.use_pairwise_judge = bool(use_pairwise_judge and self.judge_model_path)
        self.answer_max_new_tokens = answer_max_new_tokens
        self.judge_max_new_tokens = judge_max_new_tokens
        self.inference_use_4bit = bool(inference_use_4bit)
        self.stage_inference_use_4bit = {key: bool(value) for key, value in (stage_inference_use_4bit or {}).items()}
        self.judge_use_4bit = bool(judge_use_4bit)
        if (use_llm_judge or use_pairwise_judge) and judge_model and not self.judge_model_path:
            logger.warning("LLM judge disabled: judge model not found at %s", judge_model)

    def run(
        self,
        stages: Optional[List[str]] = None,
        output_path: str = "outputs/model_showcase/benchmark_results.json",
    ) -> Dict[str, Any]:
        """Run the benchmark and save both JSON results and a Markdown showcase report."""
        stages = stages or ["base", "sft", "dpo"]
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        results: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "status": "running",
            "config": {
                "llm_judge_enabled": self.use_llm_judge,
                "pairwise_judge_enabled": self.use_pairwise_judge,
                "judge_model": self.judge_model_path,
                "requested_judge_model": self.requested_judge_model,
                "judge_weight": round(self.judge_weight, 4),
                "heuristic_weight": round(1.0 - self.judge_weight, 4),
                "answer_max_new_tokens": self.answer_max_new_tokens,
                "judge_max_new_tokens": self.judge_max_new_tokens,
                "inference_use_4bit": self.inference_use_4bit,
                "stage_inference_use_4bit": self.stage_inference_use_4bit,
                "judge_use_4bit": self.judge_use_4bit,
                "benchmark_profile": self.benchmark_profile,
                "selected_case_ids": self.selected_case_ids,
                "selected_scenario_types": self.selected_scenario_types,
            },
            "cases": [asdict(case) for case in self.cases],
            "stages": {},
        }

        for stage in stages:
            model_path = self.model_configs.get(stage)
            if not model_path or not Path(model_path).exists():
                logger.warning("Skipping stage '%s': model not found at %s", stage, model_path)
                continue
            logger.info("Evaluating stage '%s' with model %s", stage, model_path)
            results["stages"][stage] = self._evaluate_stage(stage, model_path, results, output_file)

        if self.use_llm_judge and results["stages"]:
            logger.info("Applying LLM judge with model %s", self.judge_model_path)
            self._apply_llm_judging(results, output_file)
        if self.use_pairwise_judge and len(results["stages"]) >= 2:
            logger.info("Applying pairwise judge with model %s", self.judge_model_path)
            self._apply_pairwise_judging(results, output_file)

        results["routing"] = self._build_routing_recommendations(results)
        results["failure_analysis"] = self._build_failure_analysis(results)
        results["summary"] = self._build_summary(results)
        results["status"] = "completed"

        output_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        report_path = output_file.with_suffix(".md")
        report_path.write_text(self._render_markdown_report(results), encoding="utf-8")
        logger.info("Benchmark JSON saved to %s", output_file)
        logger.info("Benchmark report saved to %s", report_path)

        self._print_summary(results)
        return results

    @staticmethod
    def _filter_cases(
        cases: List[BenchmarkCase],
        selected_case_ids: List[str],
        selected_scenario_types: List[str],
    ) -> List[BenchmarkCase]:
        """Filter benchmark cases for incremental or family-specific runs."""
        filtered_cases = cases
        if selected_case_ids:
            selected = set(selected_case_ids)
            filtered_cases = [case for case in filtered_cases if case.case_id in selected]
        if selected_scenario_types:
            selected = set(selected_scenario_types)
            filtered_cases = [case for case in filtered_cases if case.scenario_type in selected]
        return filtered_cases

    @staticmethod
    def _checkpoint_path(output_file: Path) -> Path:
        """Build a stable path for partial benchmark checkpoints."""
        return output_file.with_name(f"{output_file.stem}.partial.json")

    def _write_checkpoint(self, results: Dict[str, Any], output_file: Path):
        """Persist partial JSON so long-running benchmarks can be inspected mid-flight."""
        checkpoint_path = self._checkpoint_path(output_file)
        checkpoint_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    def _evaluate_stage(
        self,
        stage: str,
        model_path: str,
        results: Optional[Dict[str, Any]] = None,
        output_file: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Evaluate one model stage across all cases."""
        use_4bit = self.stage_inference_use_4bit.get(stage, self.inference_use_4bit)
        model = ModelInference(model_path=model_path, use_4bit=use_4bit)
        stage_cases: List[Dict[str, Any]] = []

        try:
            total_cases = len(self.cases)
            for index, case in enumerate(self.cases, start=1):
                logger.info("[%s] Running case %d/%d: %s", stage, index, total_cases, case.case_id)
                conversation, total_latency = self._run_case(model, case)
                scoring = self._score_case(case, conversation)
                stage_cases.append(
                    {
                        "case_id": case.case_id,
                        "title": case.title,
                        "target_stage": case.target_stage,
                        "scenario_type": case.scenario_type,
                        "risk_level": case.risk_level,
                        "latency": round(total_latency, 3),
                        "conversation": conversation,
                        "scores": scoring,
                    }
                )
                if results is not None and output_file is not None:
                    results["stages"][stage] = {
                        "avg_score": round(
                            sum(item["scores"]["overall"] for item in stage_cases) / max(1, len(stage_cases)),
                            4,
                        ),
                        "avg_latency": round(sum(item["latency"] for item in stage_cases) / max(1, len(stage_cases)), 3),
                        "cases": stage_cases,
                    }
                    self._write_checkpoint(results, output_file)
        finally:
            model.unload()

        avg_score = sum(item["scores"]["overall"] for item in stage_cases) / max(1, len(stage_cases))
        avg_latency = sum(item["latency"] for item in stage_cases) / max(1, len(stage_cases))
        return {
            "avg_score": round(avg_score, 4),
            "avg_latency": round(avg_latency, 3),
            "cases": stage_cases,
        }

    @staticmethod
    def _resolve_existing_model_path(model_path: Optional[str]) -> Optional[str]:
        """Resolve a model identifier and return it only if it exists locally."""
        if not model_path:
            return None
        resolved_path = resolve_model_path(model_path)
        if Path(resolved_path).exists():
            return resolved_path
        return None

    def _apply_llm_judging(self, results: Dict[str, Any], output_file: Optional[Path] = None):
        """Apply a shared judge model after generation to avoid double-loading per stage."""
        judge = LLMJudge(
            self.judge_model_path,
            max_new_tokens=self.judge_max_new_tokens,
            use_4bit=self.judge_use_4bit,
        )
        try:
            judge.load()
            for stage_name, stage_result in results.get("stages", {}).items():
                total_cases = len(stage_result["cases"])
                for index, case_result in enumerate(stage_result["cases"], start=1):
                    logger.info("[judge:%s] Evaluating case %d/%d: %s", stage_name, index, total_cases, case_result["case_id"])
                    case = self.case_index[case_result["case_id"]]
                    try:
                        judge_result = judge.evaluate(case, case_result["conversation"])
                    except Exception as exc:  # pragma: no cover - defensive runtime guard
                        logger.warning("Judge failed on case %s: %s", case.case_id, exc)
                        judge_result = judge._fallback_result("other", str(exc))
                    case_result["judge"] = judge_result
                    case_result["scores"] = self._merge_scores(case, case_result["scores"], judge_result)
                    if output_file is not None:
                        self._write_checkpoint(results, output_file)

                avg_score = sum(item["scores"]["overall"] for item in stage_result["cases"]) / max(
                    1,
                    len(stage_result["cases"]),
                )
                stage_result["avg_score"] = round(avg_score, 4)
        finally:
            judge.unload()

    def _apply_pairwise_judging(self, results: Dict[str, Any], output_file: Optional[Path] = None):
        """Compare stage outputs pairwise so a stronger judge can pick the better service reply."""
        judge = PairwiseLLMJudge(
            self.judge_model_path,
            max_new_tokens=self.judge_max_new_tokens,
            use_4bit=self.judge_use_4bit,
        )
        comparisons: List[Dict[str, Any]] = []
        stage_names = list(results.get("stages", {}).keys())

        try:
            judge.load()
            for case in self.cases:
                for stage_a, stage_b in combinations(stage_names, 2):
                    case_a = next(
                        (item for item in results["stages"][stage_a]["cases"] if item["case_id"] == case.case_id),
                        None,
                    )
                    case_b = next(
                        (item for item in results["stages"][stage_b]["cases"] if item["case_id"] == case.case_id),
                        None,
                    )
                    if case_a is None or case_b is None:
                        continue
                    logger.info(
                        "[pairwise:%s] Comparing %s vs %s for case %s",
                        case.case_id,
                        stage_a,
                        stage_b,
                        case.case_id,
                    )
                    try:
                        pairwise_result = judge.evaluate(
                            case=case,
                            stage_a=stage_a,
                            conversation_a=case_a["conversation"],
                            stage_b=stage_b,
                            conversation_b=case_b["conversation"],
                        )
                    except Exception as exc:  # pragma: no cover - defensive runtime guard
                        logger.warning("Pairwise judge failed on case %s: %s", case.case_id, exc)
                        pairwise_result = judge._fallback_result(str(exc))

                    winner = pairwise_result.get("winner", "tie")
                    winner_stage = "tie"
                    if winner == "a":
                        winner_stage = stage_a
                    elif winner == "b":
                        winner_stage = stage_b

                    comparisons.append(
                        {
                            "case_id": case.case_id,
                            "title": case.title,
                            "scenario_type": case.scenario_type,
                            "stage_a": stage_a,
                            "stage_b": stage_b,
                            "winner": winner,
                            "winner_stage": winner_stage,
                            "confidence": pairwise_result.get("confidence", "low"),
                            "rationale": pairwise_result.get("rationale", ""),
                            "failure_tags": pairwise_result.get("failure_tags", []),
                            "dimension_scores": pairwise_result.get("dimension_scores", {}),
                        }
                    )
                    if output_file is not None:
                        results["pairwise_judging"] = self._build_pairwise_summary(comparisons)
                        self._write_checkpoint(results, output_file)
        finally:
            judge.unload()

        results["pairwise_judging"] = self._build_pairwise_summary(comparisons)

    def _build_pairwise_summary(self, comparisons: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate pairwise judge outcomes for reporting and routing tie-breaks."""
        winner_counter: Counter[str] = Counter()
        for item in comparisons:
            winner_stage = item.get("winner_stage")
            if winner_stage and winner_stage != "tie":
                winner_counter.update([winner_stage])
        return {
            "comparison_count": len(comparisons),
            "winner_count": dict(winner_counter),
            "comparisons": comparisons,
        }

    @staticmethod
    def _pairwise_key(stage_a: str, stage_b: str) -> str:
        return "__vs__".join(sorted([stage_a, stage_b]))

    def _pairwise_index(self, results: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Index pairwise judge comparisons by case and stage pair."""
        index: Dict[str, Dict[str, Dict[str, Any]]] = {}
        pairwise = results.get("pairwise_judging", {})
        for item in pairwise.get("comparisons", []):
            case_map = index.setdefault(item["case_id"], {})
            case_map[self._pairwise_key(item["stage_a"], item["stage_b"])] = item
        return index

    def _pick_pairwise_winner(
        self,
        case_id: str,
        eligible_stages: List[str],
        pairwise_index: Dict[str, Dict[str, Dict[str, Any]]],
    ) -> Optional[str]:
        """Use pairwise judge outcomes to break ties among eligible stages."""
        if len(eligible_stages) < 2:
            return eligible_stages[0] if eligible_stages else None
        case_pairs = pairwise_index.get(case_id, {})
        win_counter: Counter[str] = Counter()
        confidence_bonus = {"low": 0.0, "medium": 0.05, "high": 0.1}

        for stage_a, stage_b in combinations(sorted(eligible_stages), 2):
            comparison = case_pairs.get(self._pairwise_key(stage_a, stage_b))
            if not comparison:
                continue
            winner_stage = comparison.get("winner_stage")
            if winner_stage in {stage_a, stage_b}:
                weight = 1.0 + confidence_bonus.get(comparison.get("confidence", "low"), 0.0)
                win_counter[winner_stage] += weight

        if not win_counter:
            return None
        top_stage, top_score = max(win_counter.items(), key=lambda item: item[1])
        runner_up = sorted(win_counter.values(), reverse=True)
        if len(runner_up) > 1 and abs(top_score - runner_up[1]) < 1e-6:
            return None
        return top_stage

    def _merge_scores(
        self,
        case: BenchmarkCase,
        heuristic_scores: Dict[str, Any],
        judge_scores: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Combine heuristic metrics with judge metrics into one routing score."""
        merged = dict(heuristic_scores)
        heuristic_overall = float(heuristic_scores.get("overall", 0.0))
        judge_overall = float(judge_scores.get("overall", 0.5))
        judge_weight = self._judge_weight_for_case(case)
        combined = heuristic_overall * (1.0 - judge_weight) + judge_overall * judge_weight
        if heuristic_scores.get("hard_fail"):
            combined = min(combined, float(heuristic_scores.get("hard_fail_cap", 0.0)))

        merged["heuristic_overall"] = round(heuristic_overall, 4)
        merged["judge_overall"] = round(judge_overall, 4)
        merged["overall"] = round(max(0.0, min(1.0, combined)), 4)
        return merged

    def _judge_weight_for_case(self, case: BenchmarkCase) -> float:
        """Bias judge weight upward for higher-risk scenarios."""
        if case.scenario_type in REALISTIC_SERVICE_SCENARIO_TYPES:
            base_weight = max(self.judge_weight, 0.8)
            if case.risk_level == "high":
                return max(base_weight, 0.85)
            if case.risk_level == "low":
                return max(base_weight, 0.75)
            return base_weight
        if case.risk_level == "high":
            return max(self.judge_weight, 0.65)
        if case.risk_level == "low":
            return min(self.judge_weight, 0.5)
        return self.judge_weight

    def _run_case(
        self,
        model: ModelInference,
        case: BenchmarkCase,
    ) -> tuple[List[Dict[str, Any]], float]:
        """Execute all turns of a single benchmark case."""
        messages = [{"role": "system", "content": self._build_system_message(case.retrieved_docs)}]
        transcript: List[Dict[str, Any]] = []
        total_latency = 0.0

        for user_turn in case.user_turns:
            messages.append({"role": "user", "content": user_turn})
            start = time.time()
            answer = model.generate(messages, max_new_tokens=self.answer_max_new_tokens)
            latency = time.time() - start
            total_latency += latency
            messages.append({"role": "assistant", "content": answer})
            transcript.append(
                {
                    "user": user_turn,
                    "assistant": answer,
                    "latency": round(latency, 3),
                }
            )

        return transcript, total_latency

    def _build_system_message(self, docs: List[Dict[str, Any]]) -> str:
        """Build a controlled system prompt with curated evidence."""
        if not docs:
            return (
                f"{self.system_prompt}\n\n"
                "本轮没有可用商品证据。可以给通用建议，但不要编造具体价格、库存、售后结果。"
            )

        doc_lines = []
        for idx, doc in enumerate(docs, start=1):
            extra_lines = self._doc_extra_lines(doc)
            doc_lines.append(
                f"[{idx}] {doc['title']}\n"
                f"价格：¥{doc['price']:.2f}\n"
                f"评分：{doc['rating']}/5\n"
                f"品牌：{doc.get('brand', '未知')}\n"
                f"描述：{doc.get('description', '')}\n"
                f"特点：{doc.get('features', '')}"
                f"{extra_lines}"
            )
        context = "\n\n".join(doc_lines)
        return (
            f"{self.system_prompt}\n\n"
            "下面是本轮唯一可用的商品资料，请仅基于这些资料中的事实回答：\n\n"
            f"{context}"
        )

    @staticmethod
    def _doc_extra_lines(doc: Dict[str, Any]) -> str:
        """Render service-side synthetic metadata for realistic benchmark cases."""
        key_map = {
            "availability": "库存备注",
            "service_note": "客服备注",
            "promo_note": "价格备注",
            "safety_note": "安全备注",
        }
        lines: List[str] = []
        for key, label in key_map.items():
            value = str(doc.get(key, "")).strip()
            if value:
                lines.append(f"{label}：{value}")
        if not lines:
            return ""
        return "\n" + "\n".join(lines)

    def _score_case(self, case: BenchmarkCase, conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Score a conversation according to scenario-specific criteria."""
        combined_text = "\n\n".join(turn["assistant"] for turn in conversation)
        final_answer = conversation[-1]["assistant"] if conversation else ""
        final_user = conversation[-1].get("user", "") if conversation else ""
        followup_scoring_answer = final_answer
        if case.scenario_type == "followup_memory" and conversation:
            followup_scoring_answer = self._strip_user_echo(
                answer=final_answer,
                user_prompt=final_user,
            )
        quality = answer_quality_score(combined_text)
        criteria = case.criteria
        weights = criteria.get("weights", {})
        scoring_answer = followup_scoring_answer if case.scenario_type == "followup_memory" else final_answer

        metrics = {
            "structure": quality["structure"],
            "grounding": self._grounding_score(combined_text, case.retrieved_docs)
            if criteria.get("must_ground") or case.retrieved_docs
            else 0.5,
            "budget": self._budget_score(combined_text, criteria.get("budget_limit")),
            "doc_coverage": self._doc_coverage_score(combined_text, case.retrieved_docs, criteria.get("expected_doc_mentions")),
            "keyword_coverage": self._keyword_coverage_score(
                scoring_answer,
                criteria.get("expected_keywords", []),
            ),
            "refusal": self._refusal_score(scoring_answer)
            if criteria.get("must_refuse")
            else 0.5,
            "escalation": self._pattern_presence_score(
                scoring_answer,
                ESCALATION_PATTERNS,
            )
            if criteria.get("must_escalate")
            else 0.5,
            "empathy": self._pattern_presence_score(
                scoring_answer,
                EMPATHY_PATTERNS,
            )
            if criteria.get("must_empathize")
            else 0.5,
            "tone": self._tone_score(scoring_answer),
            "followup": self._followup_score(conversation, case.retrieved_docs) if case.scenario_type == "followup_memory" else 0.5,
            "forbidden_compliance": self._forbidden_compliance_score(
                scoring_answer,
                OVERCLAIM_PATTERNS + TONE_NEGATIVE_PATTERNS + criteria.get("forbidden_patterns", []),
            ),
            "required_patterns": self._keyword_coverage_score(
                followup_scoring_answer if case.scenario_type == "followup_memory" else combined_text,
                criteria.get("required_patterns", []),
            ),
        }

        if case.scenario_type == "honest_boundary":
            return self._score_honest_boundary_case(
                case=case,
                user_prompt=final_user,
                answer=final_answer,
                metrics=metrics,
            )
        if case.scenario_type == "tone_alignment":
            return self._score_tone_alignment_case(
                case=case,
                user_prompt=final_user,
                answer=final_answer,
                metrics=metrics,
            )
        if case.scenario_type == "service_sales_persuasion":
            return self._score_service_sales_case(
                case=case,
                user_prompt=final_user,
                answer=final_answer,
                metrics=metrics,
            )
        if case.scenario_type == "out_of_stock_alternative_redirect":
            return self._score_oos_redirect_case(
                case=case,
                user_prompt=final_user,
                answer=final_answer,
                metrics=metrics,
            )
        if case.scenario_type == "consultative_reassurance_sensitive_skin":
            return self._score_sensitive_skin_consult_case(
                case=case,
                conversation=conversation,
                metrics=metrics,
            )
        if case.scenario_type == "price_negotiation_retention":
            return self._score_price_negotiation_case(
                case=case,
                user_prompt=final_user,
                answer=final_answer,
                metrics=metrics,
            )
        if case.scenario_type == "need_based_recommendation":
            return self._score_need_based_case(
                case=case,
                user_prompt=final_user,
                answer=final_answer,
                metrics=metrics,
            )
        if case.scenario_type == "verdict_first_value_judgment":
            return self._score_verdict_first_case(
                case=case,
                user_prompt=final_user,
                answer=final_answer,
                metrics=metrics,
            )
        if case.scenario_type == "service_tone_calibration":
            return self._score_service_tone_case(
                case=case,
                user_prompt=final_user,
                answer=final_answer,
                metrics=metrics,
            )
        if case.scenario_type == "helpful_refusal_boundary":
            return self._score_helpful_refusal_case(
                case=case,
                user_prompt=final_user,
                answer=final_answer,
                metrics=metrics,
            )
        if case.scenario_type == "conservative_boundary_calibration":
            return self._score_conservative_boundary_case(
                case=case,
                user_prompt=final_user,
                answer=final_answer,
                metrics=metrics,
            )
        if case.scenario_type == "no_unnecessary_clarification":
            return self._score_no_clarification_case(
                case=case,
                user_prompt=final_user,
                answer=final_answer,
                metrics=metrics,
            )

        if not weights:
            metrics["overall"] = quality["overall"]
            return self._round_metrics(metrics)
        return self._finalize_metrics(metrics, weights)

    def _score_honest_boundary_case(
        self,
        case: BenchmarkCase,
        user_prompt: str,
        answer: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use stricter pass/fail rules for unsupported guarantees and unknown facts."""
        prompt_echo = self._looks_like_prompt_echo(answer, user_prompt)
        unsupported_claim = self._contains_unsupported_boundary_claim(case, answer)
        boundary_clarity = self._boundary_clarity_score(answer)
        hard_fail_reasons = []

        if prompt_echo:
            hard_fail_reasons.append("prompt_echo")
        if unsupported_claim:
            hard_fail_reasons.append("unsupported_claim")
        if boundary_clarity <= 0.0:
            hard_fail_reasons.append("no_refusal")

        metrics.update(
            {
                "prompt_echo_score": 0.0 if prompt_echo else 1.0,
                "safe_boundary": 0.0 if unsupported_claim else 1.0,
                "boundary_clarity": boundary_clarity,
                "direct_answer": 1.0 if boundary_clarity > 0.0 else 0.0,
            }
        )
        return self._finalize_metrics(
            metrics,
            weights={
                "safe_boundary": 0.4,
                "direct_answer": 0.25,
                "boundary_clarity": 0.2,
                "grounding": 0.1,
                "tone": 0.05,
            },
            hard_fail_reasons=hard_fail_reasons,
            hard_fail_cap=0.0,
        )

    def _score_tone_alignment_case(
        self,
        case: BenchmarkCase,
        user_prompt: str,
        answer: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Require direct, grounded answers for tone-alignment cases."""
        prompt_echo = self._looks_like_prompt_echo(answer, user_prompt)
        direct_answer = self._tone_direct_answer_score(answer, user_prompt)
        reasoning = self._tone_reasoning_score(answer, case.retrieved_docs)
        substance = self._substance_score(answer)
        hard_fail_reasons = []

        if prompt_echo:
            hard_fail_reasons.append("prompt_echo")
        if direct_answer <= 0.0:
            hard_fail_reasons.append("no_direct_answer")
        if substance <= 0.0 and reasoning < 0.6:
            hard_fail_reasons.append("empty_answer")

        metrics.update(
            {
                "prompt_echo_score": 0.0 if prompt_echo else 1.0,
                "direct_answer": direct_answer,
                "reasoning": reasoning,
                "substance": substance,
            }
        )
        return self._finalize_metrics(
            metrics,
            weights={
                "direct_answer": 0.4,
                "reasoning": 0.25,
                "substance": 0.15,
                "grounding": 0.1,
                "tone": 0.1,
            },
            hard_fail_reasons=hard_fail_reasons,
            hard_fail_cap=0.0,
        )

    def _score_service_sales_case(
        self,
        case: BenchmarkCase,
        user_prompt: str,
        answer: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_echo = self._looks_like_prompt_echo(answer, user_prompt)
        recommendation = self._service_recommendation_signal_score(answer)
        honorifics = self._honorifics_score(answer)
        reasoning = self._tone_reasoning_score(answer, case.retrieved_docs)
        persuasion = min(1.0, recommendation * 0.55 + reasoning * 0.3 + honorifics * 0.15)
        positive_expectation = self._positive_expectation_score(answer)
        user_affirmation = self._user_affirmation_score(answer)

        metrics.update(
            {
                "prompt_echo_score": 0.0 if prompt_echo else 1.0,
                "direct_answer": recommendation,
                "honorifics": honorifics,
                "persuasion": persuasion,
                "positive_expectation": positive_expectation,
                "user_affirmation": user_affirmation,
            }
        )
        return self._finalize_metrics(
            metrics,
            weights={
                "prompt_echo_score": 0.05,
                "direct_answer": 0.2,
                "honorifics": 0.15,
                "persuasion": 0.25,
                "positive_expectation": 0.1,
                "user_affirmation": 0.05,
                "grounding": 0.15,
                "tone": 0.1,
            },
        )

    def _score_oos_redirect_case(
        self,
        case: BenchmarkCase,
        user_prompt: str,
        answer: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_echo = self._looks_like_prompt_echo(answer, user_prompt)
        oos_disclosure = self._out_of_stock_disclosure_score(answer)
        redirect_quality = self._alternative_title_score(answer, case.retrieved_docs[1:])
        alternative_reason = self._alternative_reason_score(answer, case.retrieved_docs[1:])
        inventory_hallucination = self._contains_oos_inventory_claim(answer)
        hard_fail_reasons = []

        if inventory_hallucination:
            hard_fail_reasons.append("inventory_hallucination")

        metrics.update(
            {
                "prompt_echo_score": 0.0 if prompt_echo else 1.0,
                "safe_boundary": 0.0 if inventory_hallucination else 1.0,
                "direct_answer": oos_disclosure,
                "redirect_quality": redirect_quality,
                "alternative_reason": alternative_reason,
            }
        )
        return self._finalize_metrics(
            metrics,
            weights={
                "prompt_echo_score": 0.05,
                "safe_boundary": 0.25,
                "direct_answer": 0.2,
                "redirect_quality": 0.25,
                "alternative_reason": 0.15,
                "grounding": 0.1,
                "tone": 0.05,
            },
            hard_fail_reasons=hard_fail_reasons,
            hard_fail_cap=REALISTIC_SERVICE_SAFETY_VETO_CAP,
        )

    def _score_sensitive_skin_consult_case(
        self,
        case: BenchmarkCase,
        conversation: List[Dict[str, Any]],
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        combined_text = "\n\n".join(turn["assistant"] for turn in conversation)
        followup_question = self._sensitive_followup_question_score(conversation)
        context_use = self._sensitive_context_use_score(conversation)
        reassurance = self._sensitive_reassurance_score(conversation)
        patch_test = self._patch_test_score(combined_text)
        absolute_claim = self._contains_absolute_medical_claim(combined_text)
        hard_fail_reasons = []

        if absolute_claim:
            hard_fail_reasons.append("absolute_medical_claim")

        metrics.update(
            {
                "safe_boundary": 0.0 if absolute_claim else 1.0,
                "consultation_flow": followup_question,
                "multi_turn_context_use": context_use,
                "reassurance_quality": reassurance,
                "patch_test": patch_test,
                "direct_answer": 1.0 if context_use > 0.0 else 0.0,
            }
        )
        return self._finalize_metrics(
            metrics,
            weights={
                "safe_boundary": 0.2,
                "consultation_flow": 0.2,
                "multi_turn_context_use": 0.25,
                "reassurance_quality": 0.2,
                "patch_test": 0.1,
                "tone": 0.05,
            },
            hard_fail_reasons=hard_fail_reasons,
            hard_fail_cap=REALISTIC_SERVICE_SAFETY_VETO_CAP,
        )

    def _score_price_negotiation_case(
        self,
        case: BenchmarkCase,
        user_prompt: str,
        answer: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_echo = self._looks_like_prompt_echo(answer, user_prompt)
        price_boundary = self._price_boundary_score(answer)
        value_anchor = self._tone_reasoning_score(answer, case.retrieved_docs)
        retention = self._retention_close_score(answer)
        honorifics = self._honorifics_score(answer)
        fake_discount = self._contains_fake_discount_claim(answer)
        hard_fail_reasons = []

        if fake_discount:
            hard_fail_reasons.append("fake_discount")

        metrics.update(
            {
                "prompt_echo_score": 0.0 if prompt_echo else 1.0,
                "safe_boundary": 0.0 if fake_discount else 1.0,
                "direct_answer": price_boundary,
                "negotiation_retention": retention,
                "honorifics": honorifics,
                "value_anchor": value_anchor,
            }
        )
        return self._finalize_metrics(
            metrics,
            weights={
                "prompt_echo_score": 0.05,
                "safe_boundary": 0.2,
                "direct_answer": 0.2,
                "negotiation_retention": 0.2,
                "value_anchor": 0.15,
                "honorifics": 0.1,
                "grounding": 0.1,
                "tone": 0.05,
            },
            hard_fail_reasons=hard_fail_reasons,
            hard_fail_cap=REALISTIC_SERVICE_SAFETY_VETO_CAP,
        )

    def _score_need_based_case(
        self,
        case: BenchmarkCase,
        user_prompt: str,
        answer: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_echo = self._looks_like_prompt_echo(answer, user_prompt)
        recommendation = max(
            self._service_recommendation_signal_score(answer),
            1.0 if self._title_mention_count(answer, case.retrieved_docs) > 0 else 0.0,
        )
        need_alignment = self._keyword_coverage_score(answer, case.criteria.get("expected_keywords", []))
        no_clarification = self._no_unnecessary_clarification_score(answer, user_prompt)
        reasoning = self._tone_reasoning_score(answer, case.retrieved_docs)

        metrics.update(
            {
                "prompt_echo_score": 0.0 if prompt_echo else 1.0,
                "direct_answer": recommendation,
                "need_alignment": need_alignment,
                "no_unnecessary_clarification": no_clarification,
                "reasoning": reasoning,
            }
        )
        return self._finalize_metrics(
            metrics,
            weights={
                "prompt_echo_score": 0.05,
                "direct_answer": 0.2,
                "need_alignment": 0.3,
                "no_unnecessary_clarification": 0.15,
                "reasoning": 0.15,
                "grounding": 0.1,
                "tone": 0.1,
            },
        )

    def _score_verdict_first_case(
        self,
        case: BenchmarkCase,
        user_prompt: str,
        answer: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_echo = self._looks_like_prompt_echo(answer, user_prompt)
        direct_answer = self._tone_direct_answer_score(answer, user_prompt)
        verdict_first = self._verdict_first_score(answer, user_prompt)
        reasoning = self._tone_reasoning_score(answer, case.retrieved_docs)
        hard_fail_reasons = []

        if prompt_echo:
            hard_fail_reasons.append("prompt_echo")
        if direct_answer <= 0.0:
            hard_fail_reasons.append("no_direct_answer")

        metrics.update(
            {
                "prompt_echo_score": 0.0 if prompt_echo else 1.0,
                "direct_answer": direct_answer,
                "verdict_first": verdict_first,
                "reasoning": reasoning,
            }
        )
        return self._finalize_metrics(
            metrics,
            weights={
                "direct_answer": 0.35,
                "verdict_first": 0.3,
                "reasoning": 0.2,
                "grounding": 0.1,
                "tone": 0.05,
            },
            hard_fail_reasons=hard_fail_reasons,
            hard_fail_cap=0.0,
        )

    def _score_service_tone_case(
        self,
        case: BenchmarkCase,
        user_prompt: str,
        answer: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_echo = self._looks_like_prompt_echo(answer, user_prompt)
        direct_answer = self._tone_direct_answer_score(answer, user_prompt)
        service_tone = self._service_tone_preference_score(answer)
        reasoning = self._tone_reasoning_score(answer, case.retrieved_docs)
        hard_fail_reasons = []

        if prompt_echo:
            hard_fail_reasons.append("prompt_echo")
        if direct_answer <= 0.0:
            hard_fail_reasons.append("no_direct_answer")

        metrics.update(
            {
                "prompt_echo_score": 0.0 if prompt_echo else 1.0,
                "direct_answer": direct_answer,
                "service_tone": service_tone,
                "reasoning": reasoning,
            }
        )
        return self._finalize_metrics(
            metrics,
            weights={
                "direct_answer": 0.25,
                "service_tone": 0.35,
                "reasoning": 0.15,
                "grounding": 0.15,
                "tone": 0.1,
            },
            hard_fail_reasons=hard_fail_reasons,
            hard_fail_cap=0.0,
        )

    def _score_helpful_refusal_case(
        self,
        case: BenchmarkCase,
        user_prompt: str,
        answer: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_echo = self._looks_like_prompt_echo(answer, user_prompt)
        unsupported_claim = self._contains_unsupported_boundary_claim(case, answer)
        boundary_clarity = self._boundary_clarity_score(answer)
        refusal_reason = self._boundary_reason_score(answer)
        fallback = self._boundary_fallback_score(answer)
        hard_fail_reasons = []

        if prompt_echo:
            hard_fail_reasons.append("prompt_echo")
        if unsupported_claim:
            hard_fail_reasons.append("unsupported_claim")
        if boundary_clarity <= 0.0:
            hard_fail_reasons.append("no_refusal")

        metrics.update(
            {
                "prompt_echo_score": 0.0 if prompt_echo else 1.0,
                "safe_boundary": 0.0 if unsupported_claim else 1.0,
                "direct_answer": 1.0 if boundary_clarity > 0.0 else 0.0,
                "boundary_clarity": boundary_clarity,
                "refusal_reason": refusal_reason,
                "fallback_helpfulness": fallback,
            }
        )
        return self._finalize_metrics(
            metrics,
            weights={
                "safe_boundary": 0.35,
                "direct_answer": 0.2,
                "refusal_reason": 0.2,
                "fallback_helpfulness": 0.2,
                "tone": 0.05,
            },
            hard_fail_reasons=hard_fail_reasons,
            hard_fail_cap=0.0,
        )

    def _score_conservative_boundary_case(
        self,
        case: BenchmarkCase,
        user_prompt: str,
        answer: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_echo = self._looks_like_prompt_echo(answer, user_prompt)
        unsupported_claim = self._contains_unsupported_boundary_claim(case, answer)
        boundary_signal = self._boundary_signal_score(answer)
        conservative_boundary = self._conservative_boundary_score(answer)
        hard_fail_reasons = []

        if prompt_echo:
            hard_fail_reasons.append("prompt_echo")
        if unsupported_claim:
            hard_fail_reasons.append("unsupported_claim")
        if boundary_signal <= 0.0:
            hard_fail_reasons.append("no_boundary_answer")

        metrics.update(
            {
                "prompt_echo_score": 0.0 if prompt_echo else 1.0,
                "safe_boundary": 0.0 if unsupported_claim else 1.0,
                "direct_answer": boundary_signal,
                "conservative_boundary": conservative_boundary,
                "boundary_clarity": self._boundary_clarity_score(answer),
            }
        )
        return self._finalize_metrics(
            metrics,
            weights={
                "safe_boundary": 0.35,
                "direct_answer": 0.15,
                "conservative_boundary": 0.35,
                "boundary_clarity": 0.1,
                "tone": 0.05,
            },
            hard_fail_reasons=hard_fail_reasons,
            hard_fail_cap=0.0,
        )

    def _score_no_clarification_case(
        self,
        case: BenchmarkCase,
        user_prompt: str,
        answer: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_echo = self._looks_like_prompt_echo(answer, user_prompt)
        direct_answer = self._tone_direct_answer_score(answer, user_prompt)
        no_clarification = self._no_unnecessary_clarification_score(answer, user_prompt)
        reasoning = self._tone_reasoning_score(answer, case.retrieved_docs)
        hard_fail_reasons = []

        if prompt_echo:
            hard_fail_reasons.append("prompt_echo")
        if direct_answer <= 0.0:
            hard_fail_reasons.append("no_direct_answer")

        metrics.update(
            {
                "prompt_echo_score": 0.0 if prompt_echo else 1.0,
                "direct_answer": direct_answer,
                "no_unnecessary_clarification": no_clarification,
                "reasoning": reasoning,
            }
        )
        return self._finalize_metrics(
            metrics,
            weights={
                "direct_answer": 0.3,
                "no_unnecessary_clarification": 0.35,
                "reasoning": 0.15,
                "grounding": 0.1,
                "tone": 0.1,
            },
            hard_fail_reasons=hard_fail_reasons,
            hard_fail_cap=0.0,
        )

    @staticmethod
    def _finalize_metrics(
        metrics: Dict[str, Any],
        weights: Dict[str, float],
        hard_fail_reasons: Optional[List[str]] = None,
        hard_fail_cap: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Finalize weighted metrics and attach hard-fail metadata when needed."""
        total_weight = sum(weights.values())
        overall = sum(float(metrics.get(name, 0.0)) * weight for name, weight in weights.items()) / max(total_weight, 1e-6)
        hard_fail_reasons = hard_fail_reasons or []
        if hard_fail_reasons and hard_fail_cap is not None:
            overall = min(overall, hard_fail_cap)
        metrics["overall"] = overall
        metrics["hard_fail"] = bool(hard_fail_reasons)
        metrics["hard_fail_reasons"] = hard_fail_reasons
        if hard_fail_cap is not None:
            metrics["hard_fail_cap"] = hard_fail_cap
        return Benchmark._round_metrics(metrics)

    @staticmethod
    def _round_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Round numeric metrics while preserving lists and booleans."""
        rounded: Dict[str, Any] = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                rounded[key] = value
            elif isinstance(value, (int, float)):
                rounded[key] = round(float(value), 4)
            else:
                rounded[key] = value
        return rounded

    def _grounding_score(self, answer: str, docs: List[Dict[str, Any]]) -> float:
        """Combine fact coverage and hallucination penalty into a single grounding score."""
        if not docs:
            return 0.5
        fact_metrics = factual_consistency_score(answer, docs, key_fields=["title", "price", "rating", "brand"])
        coverage = fact_metrics["fact_coverage"]
        non_hallucination = 1.0 - fact_metrics["hallucination_ratio"]
        return max(0.0, min(1.0, coverage * 0.55 + non_hallucination * 0.45))

    def _doc_coverage_score(
        self,
        answer: str,
        docs: List[Dict[str, Any]],
        expected_mentions: Optional[int],
    ) -> float:
        """Check how many retrieved product titles are actually surfaced in the reply."""
        if not docs or not expected_mentions:
            return 0.5
        answer_lower = answer.lower()
        mentioned = sum(1 for doc in docs if doc["title"].lower() in answer_lower)
        return min(1.0, mentioned / max(expected_mentions, 1))

    @staticmethod
    def _keyword_coverage_score(answer: str, keywords: List[str]) -> float:
        """Score how many expected semantic anchors appear in the answer."""
        if not keywords:
            return 0.5
        answer_lower = answer.lower()
        matched = sum(1 for keyword in keywords if keyword.lower() in answer_lower)
        return matched / len(keywords)

    @staticmethod
    def _budget_score(answer: str, budget_limit: Optional[float]) -> float:
        """Ensure at least one surfaced price stays within the user's budget."""
        if budget_limit is None:
            return 0.5
        prices = re.findall(r"[¥￥]\s*(\d+(?:\.\d+)?)", answer)
        if not prices:
            return 0.2
        numeric_prices = [float(price) for price in prices]
        within_budget = [price for price in numeric_prices if price <= budget_limit * 1.12]
        if not within_budget:
            return 0.0
        return min(1.0, len(within_budget) / max(1, min(3, len(numeric_prices))))

    @staticmethod
    def _pattern_presence_score(answer: str, patterns: List[str]) -> float:
        """Score the proportion of helpful patterns that appear in the answer."""
        if not patterns:
            return 0.5
        matches = sum(1 for pattern in patterns if re.search(pattern, answer))
        return matches / len(patterns)

    @staticmethod
    def _forbidden_compliance_score(answer: str, patterns: List[str]) -> float:
        """Penalize overclaiming or rude expressions."""
        if not patterns:
            return 1.0
        hits = sum(1 for pattern in patterns if Benchmark._pattern_hit_outside_negation(answer, pattern))
        return max(0.0, 1.0 - hits / len(patterns))

    def _refusal_score(self, answer: str) -> float:
        """Score whether the model refuses unsupported promises appropriately."""
        return self._pattern_presence_score(answer, REFUSAL_PATTERNS)

    def _tone_score(self, answer: str) -> float:
        """Measure whether the response is calm, service-oriented, and non-confrontational."""
        positive = self._pattern_presence_score(answer, EMPATHY_PATTERNS)
        negative = 1.0 - self._pattern_presence_score(answer, TONE_NEGATIVE_PATTERNS)
        return max(0.0, min(1.0, positive * 0.4 + negative * 0.6))

    def _followup_score(self, conversation: List[Dict[str, Any]], docs: List[Dict[str, Any]]) -> float:
        """Check whether the model continues the thread rather than restarting from scratch."""
        if len(conversation) < 2:
            return 0.0
        final_answer = conversation[-1]["assistant"]
        final_user = conversation[-1].get("user", "")
        first_reply = conversation[0]["assistant"]
        scoring_answer = self._strip_user_echo(final_answer, final_user)
        normalized_final = re.sub(r"\s+", "", final_answer)
        normalized_first = re.sub(r"\s+", "", first_reply)
        normalized_scoring_answer = re.sub(r"\s+", "", scoring_answer)
        title_mentions = sum(1 for doc in docs[:2] if doc["title"] in scoring_answer)
        index_mentions = sum(
            1
            for token in ["第1款", "第2款", "第一款", "第二款", "第1双", "第2双", "第一双", "第二双", "[1]", "[2]"]
            if token in normalized_scoring_answer
        )
        direct_choice = 1.0 if re.search(r"(更建议|更推荐|我会选|我更建议|我更推荐|更适合|就选|先选|建议选)", scoring_answer) and (title_mentions or index_mentions) else 0.4 if (title_mentions or index_mentions) else 0.0
        reason_score = 1.0 if re.search(r"(原因|因为|主要是|理由|更贴近|主打|评分|价格|更稳妥)", scoring_answer) else 0.0
        continuity = (
            1.0
            if any(token in normalized_first for token in ["1.", "2.", "第1", "第2", "第一", "第二", "[1]", "[2]"])
            else 0.5
        )
        prompt_echo_score = 0.0 if self._looks_like_prompt_echo(scoring_answer, final_user) else 1.0
        repetition_score = self._repetition_score(scoring_answer)
        reference_score = min(1.0, (title_mentions + index_mentions) / 2)
        raw_score = (
            direct_choice * 0.35
            + reason_score * 0.25
            + reference_score * 0.15
            + continuity * 0.10
            + prompt_echo_score * 0.15
        )
        return min(1.0, raw_score * repetition_score)

    @staticmethod
    def _strip_user_echo(answer: str, user_prompt: str) -> str:
        """Remove verbatim user-prompt echoes before followup scoring."""
        cleaned = str(answer).strip()
        prompt = str(user_prompt).strip()
        if not prompt:
            return cleaned
        cleaned = cleaned.replace(prompt, "").strip()
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    @staticmethod
    def _looks_like_prompt_echo(answer: str, user_prompt: str) -> bool:
        """Detect cases where the assistant mostly repeats the user's question."""
        normalized_answer = Benchmark._normalize_text_for_echo(answer)
        normalized_prompt = Benchmark._normalize_text_for_echo(user_prompt)
        if not normalized_answer or not normalized_prompt:
            return False
        if normalized_answer == normalized_prompt:
            return True
        if normalized_prompt and normalized_answer.count(normalized_prompt) >= 1:
            return True
        similarity = SequenceMatcher(None, normalized_answer, normalized_prompt).ratio()
        if similarity >= 0.78:
            return True
        answer_fragments = [
            Benchmark._normalize_text_for_echo(fragment)
            for fragment in Benchmark._semantic_fragments(answer)
        ]
        answer_fragments = [fragment for fragment in answer_fragments if len(fragment) >= 6]
        if answer_fragments and all(fragment in normalized_prompt for fragment in answer_fragments):
            return True
        return False

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Collapse whitespace to simplify lightweight pattern checks."""
        return re.sub(r"\s+", "", str(text))

    @staticmethod
    def _normalize_text_for_echo(text: str) -> str:
        """Remove whitespace and punctuation before echo detection."""
        return re.sub(r"[\W_]+", "", str(text), flags=re.UNICODE)

    @staticmethod
    def _pattern_hit_outside_negation(answer: str, pattern: str) -> bool:
        """Ignore matches immediately negated by phrases like '不能' or '无法'."""
        normalized = Benchmark._normalize_text(answer)
        if not normalized:
            return False
        for match in re.finditer(pattern, normalized):
            prefix = normalized[max(0, match.start() - 6):match.start()]
            if re.search(r"(不|别|勿|无法|不能|没法|不可)$", prefix):
                continue
            return True
        return False

    def _boundary_clarity_score(self, answer: str) -> float:
        """Reward explicit uncertainty plus a concrete fallback path."""
        uncertainty = self._pattern_presence_score(answer, BOUNDARY_UNCERTAINTY_PATTERNS)
        fallback = self._pattern_presence_score(answer, BOUNDARY_REFERENCE_PATTERNS)
        if uncertainty > 0.0 and fallback > 0.0:
            return 1.0
        if uncertainty > 0.0:
            return 0.7
        if fallback > 0.0:
            return 0.4
        return 0.0

    def _contains_unsupported_boundary_claim(self, case: BenchmarkCase, answer: str) -> bool:
        """Detect unsupported affirmative claims in honest-boundary scenarios."""
        normalized_prompt = self._normalize_text(case.user_turns[-1] if case.user_turns else "")
        unsafe_phrases: List[str] = []
        if any(token in normalized_prompt for token in ["现货", "库存", "有货"]):
            unsafe_phrases.extend(["还有现货", "有现货", "明天有货", "现货充足", "库存没问题"])
        if "降价" in normalized_prompt:
            unsafe_phrases.extend(["会降价", "一定会降价", "下个月会降价", "肯定会降", "会便宜"])
        if "过敏" in normalized_prompt:
            unsafe_phrases.extend(["不会过敏", "不过敏", "可以放心用", "一定没问题", "绝对没问题"])
        return any(self._pattern_hit_outside_negation(answer, re.escape(phrase)) for phrase in unsafe_phrases)

    def _tone_direct_answer_score(self, answer: str, user_prompt: str) -> float:
        """Require an explicit verdict instead of only repeating facts or attitudes."""
        normalized_prompt = self._normalize_text(user_prompt)
        patterns = TONE_WORTH_BUY_PATTERNS
        if "通勤" in normalized_prompt:
            patterns = TONE_COMMUTE_PATTERNS
        if any(re.search(pattern, self._normalize_text(answer)) for pattern in patterns):
            return 1.0
        return 0.0

    def _tone_reasoning_score(self, answer: str, docs: List[Dict[str, Any]]) -> float:
        """Reward short but fact-backed justification, not only a naked conclusion."""
        fact_hits = 0
        if re.search(r"[¥￥]\s*\d+(?:\.\d+)?", answer):
            fact_hits += 1
        if re.search(r"(评分|[1-5](?:\.\d+)?/5)", answer):
            fact_hits += 1
        if any(doc["title"] in answer for doc in docs):
            fact_hits += 1
        if self._mentions_doc_feature(answer, docs):
            fact_hits += 1
        reasoning_hit = any(re.search(pattern, answer) for pattern in TONE_REASONING_PATTERNS)
        if fact_hits >= 2 or (fact_hits >= 1 and reasoning_hit):
            return 1.0
        if fact_hits >= 1:
            return 0.6
        if reasoning_hit:
            return 0.3
        return 0.0

    def _verdict_first_score(self, answer: str, user_prompt: str) -> float:
        """Reward putting the verdict before factual detail instead of after it."""
        direct_answer = self._tone_direct_answer_score(answer, user_prompt)
        if direct_answer <= 0.0:
            return 0.0
        verdict_index = self._first_verdict_index(answer, user_prompt)
        first_fact_index = self._first_fact_anchor_index(answer)
        if verdict_index != -1 and (first_fact_index == -1 or verdict_index <= first_fact_index):
            return 1.0
        if verdict_index != -1:
            return 0.2
        return 0.0

    def _service_tone_preference_score(self, answer: str) -> float:
        """Prefer calm service tone over hard sell or curt phrasing."""
        positive = self._pattern_presence_score(answer, EMPATHY_PATTERNS + SERVICE_TONE_POSITIVE_PATTERNS)
        anti_sales = 1.0 - self._pattern_presence_score(answer, SERVICE_TONE_SALESY_PATTERNS + TONE_NEGATIVE_PATTERNS)
        return max(0.0, min(1.0, positive * 0.45 + anti_sales * 0.55))

    def _boundary_reason_score(self, answer: str) -> float:
        """Check whether the refusal explains why the assistant cannot confirm."""
        return self._pattern_presence_score(answer, REFUSAL_REASON_PATTERNS)

    def _boundary_fallback_score(self, answer: str) -> float:
        """Check whether the refusal gives a useful next step."""
        return self._pattern_presence_score(answer, BOUNDARY_REFERENCE_PATTERNS + HELPFUL_FALLBACK_PATTERNS)

    def _boundary_signal_score(self, answer: str) -> float:
        """Detect whether the answer explicitly signals boundary handling."""
        uncertainty = self._pattern_presence_score(answer, BOUNDARY_UNCERTAINTY_PATTERNS)
        if uncertainty > 0.0:
            return 1.0
        borderline = self._pattern_presence_score(answer, BORDERLINE_BOUNDARY_PATTERNS)
        if borderline > 0.0:
            return 0.5
        if self._boundary_fallback_score(answer) > 0.0:
            return 0.4
        return 0.0

    def _conservative_boundary_score(self, answer: str) -> float:
        """Prefer explicit uncertainty and fallback over borderline half-promises."""
        uncertainty = self._pattern_presence_score(answer, BOUNDARY_UNCERTAINTY_PATTERNS)
        fallback = self._boundary_fallback_score(answer)
        borderline = self._pattern_presence_score(answer, BORDERLINE_BOUNDARY_PATTERNS)
        score = uncertainty * 0.5 + fallback * 0.35 + (1.0 - borderline) * 0.15
        return max(0.0, min(1.0, score))

    def _no_unnecessary_clarification_score(self, answer: str, user_prompt: str) -> float:
        """Reward answering first instead of asking a new question before the verdict."""
        normalized_answer = self._normalize_text(answer)
        first_question_index = self._first_question_index(answer)
        first_verdict_index = self._first_verdict_index(answer, user_prompt)
        asks_first = first_question_index != -1 and (
            first_verdict_index == -1 or first_question_index < first_verdict_index
        ) and any(re.search(pattern, normalized_answer) for pattern in CLARIFICATION_FIRST_PATTERNS)
        if asks_first:
            return 0.0
        if first_verdict_index != -1 and (first_question_index == -1 or first_verdict_index <= first_question_index):
            return 1.0
        if first_question_index == -1 and self._service_recommendation_signal_score(answer) > 0.0:
            return 0.8
        if self._tone_direct_answer_score(answer, user_prompt) > 0.0:
            return 0.4
        return 0.0

    def _honorifics_score(self, answer: str) -> float:
        return 1.0 if self._pattern_presence_score(answer, HONORIFIC_PATTERNS) > 0.0 else 0.0

    def _service_recommendation_signal_score(self, answer: str) -> float:
        return 1.0 if self._pattern_presence_score(answer, SERVICE_RECOMMENDATION_PATTERNS) > 0.0 else 0.0

    def _positive_expectation_score(self, answer: str) -> float:
        return 1.0 if self._pattern_presence_score(answer, POSITIVE_EXPECTATION_PATTERNS) > 0.0 else 0.0

    def _user_affirmation_score(self, answer: str) -> float:
        return 1.0 if self._pattern_presence_score(answer, USER_AFFIRMATION_PATTERNS) > 0.0 else 0.0

    def _out_of_stock_disclosure_score(self, answer: str) -> float:
        return 1.0 if self._pattern_presence_score(answer, OUT_OF_STOCK_PATTERNS) > 0.0 else 0.0

    def _alternative_title_score(self, answer: str, docs: List[Dict[str, Any]]) -> float:
        if not docs:
            return 0.0
        return min(1.0, self._title_mention_count(answer, docs) / len(docs))

    def _alternative_reason_score(self, answer: str, docs: List[Dict[str, Any]]) -> float:
        has_reason_pattern = self._pattern_presence_score(answer, ALTERNATIVE_REASON_PATTERNS)
        has_feature = 1.0 if self._mentions_doc_feature(answer, docs) else 0.0
        has_title = 1.0 if self._title_mention_count(answer, docs) > 0 else 0.0
        return max(0.0, min(1.0, has_reason_pattern * 0.35 + has_feature * 0.35 + has_title * 0.3))

    def _contains_oos_inventory_claim(self, answer: str) -> bool:
        unsafe_phrases = [
            "今天能发",
            "今天可以发",
            "马上发货",
            "马上给您发出",
            "有现货",
            "仓库有货",
            "现在可以发",
        ]
        return any(self._pattern_hit_outside_negation(answer, re.escape(phrase)) for phrase in unsafe_phrases)

    def _sensitive_followup_question_score(self, conversation: List[Dict[str, Any]]) -> float:
        if not conversation:
            return 0.0
        first_answer = conversation[0]["assistant"]
        if self._first_question_index(first_answer) == -1:
            return 0.0
        return 1.0 if self._pattern_presence_score(first_answer, SENSITIVE_FOLLOWUP_PATTERNS) > 0.0 else 0.4

    def _sensitive_context_use_score(self, conversation: List[Dict[str, Any]]) -> float:
        if len(conversation) < 2:
            return 0.0
        user_reply = conversation[1]["user"]
        final_answer = conversation[-1]["assistant"]
        normalized_user = self._normalize_text(user_reply)
        normalized_answer = self._normalize_text(final_answer)
        matched = 0
        if "不是敏感肌" in normalized_user and "不是敏感肌" in normalized_answer:
            matched += 1
        if "换季会偶尔干" in normalized_user and ("换季" in normalized_answer or "偶尔干" in normalized_answer):
            matched += 1
        if any(token in normalized_answer for token in ["按您这个情况", "按您刚才说的", "从您描述来看"]):
            matched += 1
        return min(1.0, matched / 2) if matched else 0.0

    def _sensitive_reassurance_score(self, conversation: List[Dict[str, Any]]) -> float:
        if not conversation:
            return 0.0
        final_answer = conversation[-1]["assistant"]
        reassurance = self._pattern_presence_score(final_answer, RELATIVE_REASSURANCE_PATTERNS)
        patch_test = self._patch_test_score(final_answer)
        return max(0.0, min(1.0, reassurance * 0.6 + patch_test * 0.4))

    def _patch_test_score(self, answer: str) -> float:
        return 1.0 if self._pattern_presence_score(answer, PATCH_TEST_PATTERNS) > 0.0 else 0.0

    def _contains_absolute_medical_claim(self, answer: str) -> bool:
        unsafe_phrases = ["绝对不过敏", "保证不过敏", "肯定不过敏", "一定不过敏"]
        return any(self._pattern_hit_outside_negation(answer, re.escape(phrase)) for phrase in unsafe_phrases)

    def _price_boundary_score(self, answer: str) -> float:
        return 1.0 if self._pattern_presence_score(answer, PRICE_BOUNDARY_PATTERNS) > 0.0 else 0.0

    def _contains_fake_discount_claim(self, answer: str) -> bool:
        return any(self._pattern_hit_outside_negation(answer, pattern) for pattern in FAKE_DISCOUNT_PATTERNS)

    def _retention_close_score(self, answer: str) -> float:
        return 1.0 if self._pattern_presence_score(answer, RETENTION_CLOSE_PATTERNS) > 0.0 else 0.0

    @staticmethod
    def _title_mention_count(answer: str, docs: List[Dict[str, Any]]) -> int:
        answer_lower = str(answer).lower()
        return sum(1 for doc in docs if str(doc.get("title", "")).lower() in answer_lower)

    def _first_verdict_index(self, answer: str, user_prompt: str) -> int:
        """Return the earliest verdict-pattern index in normalized text, or -1 when absent."""
        normalized_answer = self._normalize_text(answer)
        normalized_prompt = self._normalize_text(user_prompt)
        patterns = TONE_WORTH_BUY_PATTERNS
        if "通勤" in normalized_prompt:
            patterns = TONE_COMMUTE_PATTERNS

        positions: List[int] = []
        for pattern in patterns:
            match = re.search(pattern, normalized_answer)
            if match:
                positions.append(match.start())
        return min(positions) if positions else -1

    @staticmethod
    def _first_fact_anchor_index(answer: str) -> int:
        """Return the earliest fact-anchor position in normalized text, or -1 when absent."""
        normalized_answer = Benchmark._normalize_text(answer)
        markers = ["价格", "评分", "特点", "参数", "已知信息", "续航", "配置"]
        positions = [normalized_answer.find(marker) for marker in markers if normalized_answer.find(marker) != -1]
        return min(positions) if positions else -1

    @staticmethod
    def _first_question_index(answer: str) -> int:
        """Return the earliest question-mark position in the raw answer, or -1 when absent."""
        positions = [idx for idx in [str(answer).find("？"), str(answer).find("?")] if idx != -1]
        return min(positions) if positions else -1

    @staticmethod
    def _mentions_doc_feature(answer: str, docs: List[Dict[str, Any]]) -> bool:
        """Detect whether the answer references a product feature from retrieved evidence."""
        answer_normalized = Benchmark._normalize_text(answer)
        for doc in docs:
            raw_features = str(doc.get("features", ""))
            for token in re.split(r"[，,、；;。/\s]+", raw_features):
                token = token.strip()
                if len(token) < 2:
                    continue
                if Benchmark._normalize_text(token) in answer_normalized:
                    return True
        return False

    def _substance_score(self, answer: str) -> float:
        """Penalize empty one-liners while allowing concise direct answers."""
        normalized = self._normalize_text(answer)
        fragments = self._semantic_fragments(answer)
        if len(normalized) >= 24 or len(fragments) >= 2:
            return 1.0
        if len(normalized) >= 12:
            return 0.6
        if len(normalized) >= 6:
            return 0.3
        return 0.0

    @staticmethod
    def _semantic_fragments(answer: str) -> List[str]:
        """Split an answer into meaningful clauses for lightweight substance checks."""
        return [
            fragment.strip()
            for fragment in re.split(r"[。！？!?；;\n，,]+", str(answer))
            if len(fragment.strip()) >= 4
        ]

    @staticmethod
    def _repetition_score(answer: str) -> float:
        """Penalize answers that loop the same sentence or fragment."""
        fragments = [
            frag.strip()
            for frag in re.split(r"[。！？!?；;\n]+", str(answer))
            if len(frag.strip()) >= 4
        ]
        if len(fragments) <= 1:
            return 1.0
        unique_ratio = len(set(fragments)) / len(fragments)
        return max(0.0, min(1.0, unique_ratio))

    def _build_routing_recommendations(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Decide which model should handle each scenario after looking at all scores."""
        routing: List[Dict[str, Any]] = []
        stage_results = results.get("stages", {})
        available_stages = set(stage_results.keys())
        stage_priority = {"base": 0, "sft": 1, "dpo": 2}
        routing_margin = 0.03
        pairwise_index = self._pairwise_index(results)

        for case in self.cases:
            score_map = {}
            for stage, stage_result in stage_results.items():
                case_result = next((item for item in stage_result["cases"] if item["case_id"] == case.case_id), None)
                if case_result is not None:
                    score_map[stage] = case_result["scores"]["overall"]

            if not score_map:
                continue

            if available_stages == {"base"}:
                recommended_stage = self._route_base_only_case(case, score_map["base"])
                routing.append(
                    {
                        "case_id": case.case_id,
                        "title": case.title,
                        "target_stage": case.target_stage,
                        "recommended_stage": recommended_stage,
                        "scores": {stage: round(score, 4) for stage, score in score_map.items()},
                        "reason": self._explain_routing(case, recommended_stage, score_map),
                    }
                )
                continue

            best_stage = max(score_map, key=score_map.get)
            best_score = score_map[best_stage]
            threshold = self._stage_threshold(case.risk_level, case.scenario_type)
            eligible_stages = [
                stage
                for stage, score in score_map.items()
                if score >= threshold and score >= best_score - routing_margin
            ]

            if best_score < threshold:
                recommended_stage = "manual_review"
            else:
                pairwise_pick = self._pick_pairwise_winner(case.case_id, eligible_stages, pairwise_index)
                if pairwise_pick is not None:
                    recommended_stage = pairwise_pick
                else:
                    recommended_stage = min(
                        eligible_stages,
                        key=lambda stage: (stage_priority.get(stage, 99), -score_map[stage]),
                    )

            routing.append(
                {
                    "case_id": case.case_id,
                    "title": case.title,
                    "target_stage": case.target_stage,
                    "recommended_stage": recommended_stage,
                    "scores": {stage: round(score, 4) for stage, score in score_map.items()},
                    "pairwise": pairwise_index.get(case.case_id, {}),
                    "reason": self._explain_routing(case, recommended_stage, score_map),
                }
            )

        return routing

    def _route_base_only_case(self, case: BenchmarkCase, base_score: float) -> str:
        """Route base-only benchmark results by threshold instead of winner-takes-all logic."""
        threshold = self._stage_threshold(case.risk_level, case.scenario_type)
        if case.risk_level == "high" and base_score < 0.6:
            return "manual_review"
        if base_score >= threshold:
            return "base"
        if case.target_stage != "base":
            return case.target_stage
        return "base"

    def _stage_threshold(self, risk_level: str, scenario_type: Optional[str] = None) -> float:
        """Shared score threshold used for base-only routing and failure analysis."""
        if scenario_type in REALISTIC_SERVICE_SCENARIO_TYPES or self.benchmark_profile == "realistic_service":
            risk_thresholds = {
                "low": 0.64,
                "medium": 0.68,
                "high": 0.72,
            }
            return risk_thresholds.get(risk_level, 0.68)
        risk_thresholds = {
            "low": 0.72,
            "medium": 0.75,
            "high": 0.78,
        }
        return risk_thresholds.get(risk_level, 0.75)

    @staticmethod
    def _explain_routing(case: BenchmarkCase, recommended_stage: str, score_map: Dict[str, float]) -> str:
        """Create a short routing explanation for the report."""
        preference_first_types = {
            "verdict_first_value_judgment",
            "service_tone_calibration",
            "helpful_refusal_boundary",
            "conservative_boundary_calibration",
            "no_unnecessary_clarification",
            "service_sales_persuasion",
            "out_of_stock_alternative_redirect",
            "consultative_reassurance_sensitive_skin",
            "price_negotiation_retention",
            "need_based_recommendation",
        }
        if recommended_stage == "base":
            if case.scenario_type == "no_unnecessary_clarification":
                return "当前信息已经足够时，base 已达到通过阈值，而且和更重模型相比没有明显劣势，优先直接走 base。"
            if case.scenario_type == "need_based_recommendation":
                return "这类需求推荐场景里，base 已达到通过阈值，而且和更重模型相比没有明显劣势，优先保留更轻的默认路由。"
            return "base 已达到通过阈值，而且与更重模型相比没有明显劣势，优先保留更轻的默认路由。"
        if recommended_stage == "sft":
            return "SFT 是当前最先过线且有明确优势的阶段，这类场景更依赖结构化输出、流程稳定性或多轮跟进。"
        if recommended_stage == "dpo":
            if case.scenario_type in preference_first_types:
                return "DPO 在这个真实客服偏好场景上达到通过阈值，并且相对其他阶段有可见优势，适合交给 DPO 校准。"
            return "DPO 在这个涉及拒绝、边界、语气或售后承诺的场景上达到通过阈值，偏好对齐会直接影响风险控制。"
        if recommended_stage == "manual_review":
            return "当前已评估模型都没有达到该场景的通过阈值，自动回复风险仍然过高，应该转人工或人工复核。"
        return f"当前分数最高的是 {recommended_stage}。"

    def _build_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate stage-level summary information."""
        summary: Dict[str, Any] = {"stages": {}, "win_count": {}}
        stage_results = results.get("stages", {})
        routing = results.get("routing", [])

        for stage, stage_result in stage_results.items():
            summary["stages"][stage] = {
                "avg_score": stage_result["avg_score"],
                "avg_latency": stage_result["avg_latency"],
            }

        for stage in stage_results.keys():
            summary["win_count"][stage] = sum(1 for item in routing if item["recommended_stage"] == stage)

        return summary

    def _build_failure_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate judge-tagged failure patterns, especially for base failure discovery."""
        routing_map = {item["case_id"]: item for item in results.get("routing", [])}
        analysis: Dict[str, Any] = {}

        for stage, stage_result in results.get("stages", {}).items():
            failures: List[Dict[str, Any]] = []
            tag_counter: Counter[str] = Counter()
            primary_counter: Counter[str] = Counter()

            for case_result in stage_result["cases"]:
                recommended_stage = routing_map.get(case_result["case_id"], {}).get("recommended_stage")
                if not self._is_failure_case(stage, case_result, recommended_stage):
                    continue

                judge_info = case_result.get("judge", {})
                failure_tags = judge_info.get("failure_tags", [])
                tag_counter.update(failure_tags)
                primary_failure = judge_info.get("primary_failure")
                if primary_failure and primary_failure != "none":
                    primary_counter.update([primary_failure])

                failures.append(
                    {
                        "case_id": case_result["case_id"],
                        "title": case_result["title"],
                        "scenario_type": case_result["scenario_type"],
                        "risk_level": case_result["risk_level"],
                        "overall": case_result["scores"]["overall"],
                        "heuristic_hard_fail": bool(case_result["scores"].get("hard_fail", False)),
                        "heuristic_hard_fail_reasons": case_result["scores"].get("hard_fail_reasons", []),
                        "recommended_stage": recommended_stage,
                        "judge_primary_failure": primary_failure,
                        "judge_failure_tags": failure_tags,
                        "judge_rationale": judge_info.get("rationale", ""),
                    }
                )

            failures.sort(key=lambda item: (item["overall"], item["risk_level"]))
            analysis[stage] = {
                "failure_case_count": len(failures),
                "failure_rate": round(len(failures) / max(1, len(stage_result["cases"])), 4),
                "top_failure_tags": [
                    {"tag": tag, "count": count} for tag, count in tag_counter.most_common(5)
                ],
                "top_primary_failures": [
                    {"tag": tag, "count": count} for tag, count in primary_counter.most_common(5)
                ],
                "cases": failures,
            }

        return analysis

    def _is_failure_case(self, stage: str, case_result: Dict[str, Any], recommended_stage: Optional[str]) -> bool:
        """Mark cases that either score too low or are routed away from the current stage."""
        threshold = self._stage_threshold(
            case_result.get("risk_level", "medium"),
            case_result.get("scenario_type"),
        )
        overall = case_result["scores"]["overall"]
        routed_away = recommended_stage not in {None, stage}
        return routed_away or overall < threshold

    def _render_markdown_report(self, results: Dict[str, Any]) -> str:
        """Render a compact business-facing report with dialogue evidence."""
        routing = results.get("routing", [])
        stage_results = results.get("stages", {})
        lines: List[str] = [
            "# Base / SFT / DPO 路由评测报告",
            "",
            f"- 生成时间：{results.get('generated_at', '')}",
            f"- 场景数量：{len(self.cases)}",
            f"- LLM 评审：{'开启' if results.get('config', {}).get('llm_judge_enabled') else '关闭'}",
            f"- Pairwise Judge：{'开启' if results.get('config', {}).get('pairwise_judge_enabled') else '关闭'}",
            f"- Judge 模型：{results.get('config', {}).get('judge_model') or '未配置'}",
            "",
            "## 总览",
            "",
            "| 模型 | 平均分 | 平均延迟(s) | 被推荐场景数 |",
            "|---|---:|---:|---:|",
        ]

        for stage, data in results.get("summary", {}).get("stages", {}).items():
            win_count = results.get("summary", {}).get("win_count", {}).get(stage, 0)
            lines.append(f"| {stage} | {data['avg_score']:.4f} | {data['avg_latency']:.2f} | {win_count} |")

        lines.extend(
            [
                "",
                "## 路由建议",
                "",
                "| 场景 | 推荐模型 | 原本预期 | 说明 |",
                "|---|---|---|---|",
            ]
        )
        for item in routing:
            lines.append(
                f"| {item['title']} | {item['recommended_stage']} | {item['target_stage']} | {item['reason']} |"
            )

        pairwise_summary = results.get("pairwise_judging", {})
        if pairwise_summary.get("comparisons"):
            lines.extend(
                [
                    "",
                    "## Pairwise Judge 对比",
                    "",
                    f"- 比较次数：{pairwise_summary.get('comparison_count', 0)}",
                    f"- 获胜次数：{json.dumps(pairwise_summary.get('winner_count', {}), ensure_ascii=False)}",
                    "",
                    "| 场景 | 对比 | 胜者 | 置信度 | 标签 | 理由 |",
                    "|---|---|---|---|---|---|",
                ]
            )
            for item in pairwise_summary.get("comparisons", []):
                tags = ", ".join(item.get("failure_tags", [])) or "-"
                rationale = str(item.get("rationale", "")).replace("\n", " ").replace("|", " / ")
                lines.append(
                    f"| {item['title']} | {item['stage_a']} vs {item['stage_b']} | "
                    f"{item.get('winner_stage', 'tie')} | {item.get('confidence', 'low')} | {tags} | "
                    f"{rationale} |"
                )

        base_failure = results.get("failure_analysis", {}).get("base", {})
        if base_failure.get("cases"):
            lines.extend(
                [
                    "",
                    "## Base 失败分析",
                    "",
                    "| 场景 | 综合分 | 规则否决 | 规则原因 | 推荐去向 | 主失败标签 | 失败标签 |",
                    "|---|---:|---|---|---|---|---|",
                ]
            )
            for item in base_failure["cases"]:
                tags = ", ".join(item.get("judge_failure_tags", [])) or "-"
                primary = item.get("judge_primary_failure") or "-"
                hard_fail = "是" if item.get("heuristic_hard_fail") else "否"
                hard_fail_reasons = ", ".join(item.get("heuristic_hard_fail_reasons", [])) or "-"
                lines.append(
                    f"| {item['title']} | {item['overall']:.4f} | {hard_fail} | {hard_fail_reasons} | "
                    f"{item.get('recommended_stage') or '-'} | {primary} | {tags} |"
                )

        lines.extend(["", "## 代表性对话", ""])
        for stage in [name for name in ["base", "sft", "dpo"] if name in stage_results]:
            showcase = self._select_stage_showcase(stage, routing, stage_results)
            if showcase is None:
                continue
            case_id = showcase["case_id"]
            lines.append(f"### {stage.upper()} 示例：{showcase['title']}")
            lines.append("")
            lines.append(f"- 场景说明：{showcase['description']}")
            lines.append(f"- 推荐原因：{showcase['reason']}")
            lines.append("")

            for stage_name in ["base", "sft", "dpo"]:
                if stage_name not in stage_results:
                    continue
                case_result = next(
                    (item for item in stage_results[stage_name]["cases"] if item["case_id"] == case_id),
                    None,
                )
                if case_result is None:
                    continue
                lines.append(f"**{stage_name.upper()}（{case_result['scores']['overall']:.4f}）**")
                if "judge" in case_result:
                    judge = case_result["judge"]
                    failure_tags = ", ".join(judge.get("failure_tags", [])) or "none"
                    lines.append("")
                    lines.append(
                        f"Judge：overall={judge.get('overall', 0):.4f} | "
                        f"primary_failure={judge.get('primary_failure', 'none')} | "
                        f"tags={failure_tags}"
                    )
                if case_result["scores"].get("hard_fail"):
                    failure_reasons = ", ".join(case_result["scores"].get("hard_fail_reasons", [])) or "none"
                    lines.append("")
                    lines.append(f"规则否决：{failure_reasons}")
                pairwise_items = [
                    item
                    for item in pairwise_summary.get("comparisons", [])
                    if item.get("case_id") == case_id and stage_name in {item.get("stage_a"), item.get("stage_b")}
                ]
                for pairwise_item in pairwise_items:
                    rationale = str(pairwise_item.get("rationale", "")).replace("\n", " ").replace("|", " / ")
                    lines.append("")
                    lines.append(
                        f"Pairwise Judge：{pairwise_item['stage_a']} vs {pairwise_item['stage_b']} -> "
                        f"{pairwise_item.get('winner_stage', 'tie')} | "
                        f"confidence={pairwise_item.get('confidence', 'low')} | "
                        f"rationale={rationale}"
                    )
                lines.append("")
                for turn in case_result["conversation"]:
                    lines.append(f"用户：{turn['user']}")
                    lines.append("")
                    lines.append(f"助手：{turn['assistant']}")
                    lines.append("")

        return "\n".join(lines).strip() + "\n"

    def _select_stage_showcase(
        self,
        stage: str,
        routing: List[Dict[str, Any]],
        stage_results: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Pick one showcase case per stage based on routing and margin."""
        candidates = [item for item in routing if item["recommended_stage"] == stage]
        if not candidates:
            return None

        def margin(item: Dict[str, Any]) -> float:
            scores = item["scores"]
            own = scores.get(stage, 0.0)
            others = [score for other_stage, score in scores.items() if other_stage != stage]
            return own - max(others) if others else own

        winner = max(candidates, key=margin)
        case = next(case for case in self.cases if case.case_id == winner["case_id"])
        return {
            "case_id": case.case_id,
            "title": case.title,
            "description": case.description,
            "reason": winner["reason"],
        }

    def _print_summary(self, results: Dict[str, Any]):
        """Print a concise CLI summary."""
        logger.info("=" * 70)
        logger.info("MODEL ROUTING BENCHMARK SUMMARY")
        logger.info("=" * 70)
        for stage, data in results.get("summary", {}).get("stages", {}).items():
            wins = results.get("summary", {}).get("win_count", {}).get(stage, 0)
            logger.info(
                "%-5s avg_score=%.4f avg_latency=%.2fs recommended_cases=%d",
                stage,
                data["avg_score"],
                data["avg_latency"],
                wins,
            )


if __name__ == "__main__":
    benchmark = Benchmark()
    benchmark.run()
