from cognitive_bt_framework.src.llm_interface.llm_interface_openai import LLMInterfaceOpenAI
from cognitive_bt_framework.src.htn_planner.htn_planner import HTNPlanner
from cognitive_bt_framework.src.cbt_planner.cbtf import CognitiveBehaviorTreeFramework


def iterative_refinement_process(task_name, llm_interface, planner):
    max_iterations = 5
    iteration = 0
    user_satisfied = False

    while iteration < max_iterations and not user_satisfied:
        decomposition = planner.decompose_task(task_name)
        print(f"\nIteration {iteration + 1}: Proposed Decomposition for '{task_name}':\n{decomposition}")

        # Simulate asking for user feedback
        user_feedback = input("Please provide your feedback (or type 'ok' if satisfied): ")
        if user_feedback.lower() == 'ok':
            user_satisfied = True
            print("Decomposition accepted.")
            break

        # Refine the decomposition based on feedback
        planner.refine_decomposition_with_feedback(task_name, user_feedback)
        iteration += 1

    if not user_satisfied:
        print("Maximum iterations reached. Please review the final decomposition.")

def task_learning_sessions(planner):
    print("Interactive Task Learning Session\n")
    while True:
        task_name = input("\nEnter a new task (or type 'exit' to end): ")
        if task_name.lower() == 'exit':
            break

        iterative_refinement_process(task_name, planner.llm_interface, planner)

        # After each session, feedback could be analyzed, and decompositions adjusted
        # For simplicity, this step is conceptual. Implement based on your feedback analysis strategy.
        analyze_feedback_and_update_model(planner)