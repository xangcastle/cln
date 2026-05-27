from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
msgs = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hello!"}]

try:
    ret = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=False)
    print("return_dict=False:", type(ret))
    
    ret2 = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    print("default:", type(ret2))
    if hasattr(ret2, "input_ids"):
        print("has input_ids")
except Exception as e:
    print("Exception:", type(e), str(e))
