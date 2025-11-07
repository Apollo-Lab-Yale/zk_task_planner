from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI

llm_interface = LLMInterfaceOpenAI()
output = llm_interface.get_task_decomposition("clean the kitchen", None, None)

print(output)