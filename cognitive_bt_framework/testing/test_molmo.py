import litserve as ls
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image

class MolmoAPI(ls.LitAPI):
    def setup(self, device):
        # Load the processor
        self.processor = AutoProcessor.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

        # load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            'allenai/Molmo-7B-D-0924',
            trust_remote_code=True,
            torch_dtype='auto',
            device_map='auto'
        )

    def decode_request(self, request):
        # Extract file from request
        return (request["content"].file, request["prompt"])

    def predict(self, input):
        image = Image.open(input[0]).convert('RGB')
        inputs = self.processor.process(
            images=[image],
            text=input[1]
        )
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}
        output = self.model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
            tokenizer=self.processor.tokenizer
        )

        generated_tokens = output[0,inputs['input_ids'].size(1):]
        return self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def encode_response(self, result):
        return result

# Starting the server
if __name__ == "__main__":
    api = MolmoAPI()
    server = ls.LitServer(api, timeout=False)
    server.run(port=8000)