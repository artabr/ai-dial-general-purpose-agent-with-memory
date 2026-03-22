import json
from typing import Any

from task.tools.base import BaseTool
from task.tools.memory._models import MemoryData
from task.tools.memory.memory_store import LongTermMemoryStore
from task.tools.models import ToolCallParams


class SearchMemoryTool(BaseTool):
    """
    Tool for searching long-term memories about the user.

    Performs semantic search over stored memories to find relevant information.
    """

    def __init__(self, memory_store: LongTermMemoryStore):
        self.memory_store = memory_store


    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return ("Search long-term memories about the user using semantic similarity. "
                "Use this tool to recall previously stored information about the user's preferences, personal details, goals, or context. "
                "WHEN TO USE: At the START of conversations to retrieve relevant user context, or when user asks about something you might have remembered. "
                "CRITICAL: Always search memories BEFORE responding to user queries to personalize responses. "
                "The search uses semantic matching, so you can use natural language queries like 'What are the user's preferences?' or 'user's work information'. "
                "Returns the most relevant memories ranked by similarity.")

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Can be a question or keywords to find relevant memories"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of most relevant memories to return.",
                    "minimum": 1,
                    "maximum": 20,
                    "default": 5
                }
            },
            "required": ["query"]
        }


    async def _execute(self, tool_call_params: ToolCallParams) -> str:
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        
        query = arguments["query"]
        top_k = arguments.get("top_k", 5)
        
        results = await self.memory_store.search_memories(
            api_key=tool_call_params.api_key,
            query=query,
            top_k=top_k
        )
        
        if not results:
            final_result = "No memories found."
        else:
            memory_lines = []
            for idx, memory in enumerate(results, 1):
                memory_text = f"**Memory {idx}:**\n- **Content:** {memory.content}\n- **Category:** {memory.category}"
                if memory.topics:
                    memory_text += f"\n- **Topics:** {', '.join(memory.topics)}"
                memory_lines.append(memory_text)
            
            final_result = "\n\n".join(memory_lines)
        
        tool_call_params.stage.append_content(final_result)
        
        return final_result
