from transformers import BartForConditionalGeneration, BartTokenizer


class LLMInterface:
    def __init__(self, model_name="facebook/bart-large"):
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)

    def generate_prompt(self, task):
        # In the case of BART, you may not need to modify the task for a prompt
        # but you can add specific instructions or context here if needed.
        return task

    def query_llm(self, prompt):
        # Tokenize the prompt
        inputs = self.tokenizer.encode("Decompose task: " + prompt, return_tensors="pt", max_length=1024,
                                       truncation=True)

        # Generate output sequence
        outputs = self.model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                      early_stopping=True)

        # Decode the output
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return decoded_output

    def get_task_decomposition(self, task):
        prompt = self.generate_prompt(task)
        decomposition = self.query_llm(prompt)
        return decomposition

if __name__ == "__main__":
    # Example usage
    llm_interface = LLMInterface()
    task = "Organize a small conference"
    decomposition = llm_interface.get_task_decomposition(task)
    print(decomposition)
