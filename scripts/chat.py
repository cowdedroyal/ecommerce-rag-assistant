#!/usr/bin/env python3
"""
Interactive Chinese multi-turn chat interface for the E-commerce RAG assistant.
Supports model switching, retrieval mode selection, and rich console output.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.assistant import ECommerceRAG
from src.config import Settings
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class ChatSession:
    """Interactive chat session with the RAG assistant."""

    def __init__(self, use_llm: bool = False, retrieval_mode: str = "hybrid"):
        settings = Settings()
        self.assistant = ECommerceRAG(
            product_dataset_path=str(settings.PRODUCT_DATA_PATH),
            order_dataset_path=str(settings.ORDER_DATA_PATH),
            retrieval_mode=retrieval_mode,
        )
        self.customer_id = None
        self.use_llm = use_llm
        self.retrieval_mode = retrieval_mode

    def process_input(self, user_input: str) -> str:
        """Process user input and return response."""
        user_input = user_input.strip()

        if not user_input:
            return "请输入您的问题。"

        # Handle commands
        if user_input.lower() in ["quit", "exit", "bye", "退出", "再见"]:
            return "__EXIT__"

        if user_input.startswith("set customer") or user_input.startswith("设置客户"):
            return self._set_customer(user_input)

        if user_input in ["clear", "清空", "新对话"]:
            self.assistant.clear_conversation()
            return "对话历史已清空。"

        if user_input.startswith("mode ") or user_input.startswith("模式 "):
            return self._set_mode(user_input)

        # Process query
        try:
            result = self.assistant.process_query(
                query=user_input,
                customer_id=self.customer_id,
                use_llm=self.use_llm,
                retrieval_mode=self.retrieval_mode,
            )

            answer = result["answer"]

            # Show retrieval info
            process = result.get("retrieval_process", {})
            if process:
                answer += "\n\n---\n"
                bm25 = process.get("bm25_results", [])
                dense = process.get("dense_results", [])
                reranked = process.get("reranked_results", [])

                if bm25:
                    answer += f"[BM25 Top-3] "
                    answer += ", ".join(
                        f"{r.get('title', r['id'])}({r['score']:.3f})"
                        for r in bm25[:3]
                    )
                    answer += "\n"
                if dense:
                    answer += f"[Dense Top-3] "
                    answer += ", ".join(
                        f"{r.get('title', r['id'])}({r['score']:.3f})"
                        for r in dense[:3]
                    )
                    answer += "\n"
                if reranked:
                    answer += f"[Rerank Top-3] "
                    answer += ", ".join(
                        f"{r.get('title', r.get('id', ''))}({r.get('score', 0):.3f})"
                        for r in reranked[:3]
                    )

            return answer

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"抱歉，处理您的请求时出错：{str(e)}"

    def _set_customer(self, text: str) -> str:
        """Set customer ID."""
        try:
            parts = text.split()
            self.customer_id = int(parts[-1])
            return f"客户ID已设置为：{self.customer_id}"
        except (ValueError, IndexError):
            return "请输入有效的客户ID，如：设置客户 12345"

    def _set_mode(self, text: str) -> str:
        """Set retrieval mode."""
        parts = text.split()
        if len(parts) < 2:
            return f"当前模式：{self.retrieval_mode}。可选：hybrid / dense / bm25"
        mode = parts[-1].lower()
        if mode in ["hybrid", "dense", "bm25"]:
            self.retrieval_mode = mode
            return f"检索模式已切换为：{mode}"
        return "无效模式。可选：hybrid / dense / bm25"


def main():
    """Main chat loop."""
    print("=" * 50)
    print("  电商智能助手 - 中文多轮对话")
    print("=" * 50)
    print("\n正在初始化...")

    # Parse args
    use_llm = "--llm" in sys.argv
    mode = "hybrid"
    for arg in sys.argv:
        if arg.startswith("--mode="):
            mode = arg.split("=")[1]

    session = ChatSession(use_llm=use_llm, retrieval_mode=mode)

    print("\n初始化完成！\n")
    print("功能说明：")
    print("  1. 直接输入问题进行商品查询/推荐")
    print("  2. 支持多轮对话追问")
    print("  3. '设置客户 <ID>' - 设置客户ID查询订单")
    print("  4. '模式 <hybrid/dense/bm25>' - 切换检索模式")
    print("  5. '清空' - 清空对话历史")
    print("  6. '退出' - 退出程序")
    if use_llm:
        print("  [LLM 模式已启用]")
    print(f"  [检索模式: {mode}]")
    print("-" * 50)

    while True:
        try:
            user_input = input("\n👤 你: ").strip()
            response = session.process_input(user_input)

            if response == "__EXIT__":
                print("\n🤖 再见！祝您购物愉快！")
                break

            print(f"\n🤖 助手: {response}")

        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except EOFError:
            break


if __name__ == "__main__":
    main()
