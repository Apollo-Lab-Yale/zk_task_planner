from cognitive_bt_framework.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv
from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI

if __name__ == '__main__':
    sim = AI2ThorSimEnv(8)

    context, states = sim.get_context(4)
    llm = LLMInterfaceOpenAI()
    print(llm.get_task_decomposition_ordered_context("Bring a mug of coffee to the table.", sim.get_object_names(), context))